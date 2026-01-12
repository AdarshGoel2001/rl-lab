"""Original 2018 World Models workflow placeholder.

This file is intentionally left as a scaffold so we can gradually rebuild the
exact VAE + MDN-RNN + CMA-ES pipeline described by Ha & Schmidhuber (2018).
Treat the TODOs like homework checkpoints: fill them in one by one as features
come online.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import time
import numpy as np
import torch
import torch.nn.functional as F

from .base import Batch, CollectResult, PhaseConfig, WorldModelWorkflow
from .context import WorkflowContext


class OriginalWorldModelsWorkflow(WorldModelWorkflow):
    """Scaffold for the OG World Models agent (V, M, C modules)."""

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Workflow lifecycle hooks
    # ------------------------------------------------------------------
    def initialize(self, context: WorkflowContext) -> None:
        """Bind orchestrator resources and reset episode tracking."""
        self._bind_context(context)
        self._reset_episode_tracking(self.num_envs, clear_history=True)
        self._reset_rollout_state(
            initial_obs=context.initial_observation,
            initial_dones=context.initial_dones,
        )

    def on_context_update(self, context: WorkflowContext) -> None:
        """Refresh context-derived handles if orchestrator updates them."""
        # TODO: Update optimizers/contexts after checkpoint load or phase transition
        pass

    def collect_step(
        self,
        step: int,
        *,
        phase: PhaseConfig,
    ) -> Optional[CollectResult]:
        """Gather on-environment experience for training V and M."""
        obs_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        num_episodes = 0
        start_time = time.time()
        with torch.no_grad():
          for _ in range(self.collect_length):
            obs_tensor = torch.as_tensor(self.current_obs, device=self.device, dtype=torch.float32) / 255.0
            obs_tensor_reshaped = obs_tensor.permute(0, 3, 1, 2)
            features = self.vae.observe(obs_tensor_reshaped)
            action_tensor = self.controller.act(features["latent"], self.hidden_state) 
            #self.hidden_state is a tuple of (hidden, cell)
            self.prev_action = action_tensor.detach()
            result = self.mdn_rnn.observe(latent = features["latent"], action = self.prev_action, dones = self.current_dones)
            #latent = result["next_latent"]
            self.hidden_state = result["hidden"]
            action = action_tensor.detach().cpu().numpy()
            next_obs, reward, done, info = self.environment.step(action)
            if np.any(done):
              num_episodes += int(np.sum(done))
            self._update_episode_stats(reward, done, info)
            obs_list.append(self.current_obs.copy())
            self.current_obs = next_obs
            self.current_dones = done
            dones_list.append(self.current_dones.copy())
            actions_list.append(action.copy())
            rewards_list.append(reward.copy())
        trajectory = {
            "observations": np.stack(obs_list, axis=0),
            "actions": np.stack(actions_list, axis=0),
            "rewards": np.stack(rewards_list, axis=0),
            "dones": np.stack(dones_list, axis=0),
        }

        return CollectResult(
          steps=self.collect_length * self.num_envs,  
          episodes=num_episodes,
          metrics = { 
            "collect/mean_step_reward": float(np.mean(trajectory["rewards"])),
            "collect/duration": time.time() - start_time,
          },
          trajectory = trajectory)

    def update_world_model(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        # Normalize observations to [0, 1] for BCE loss
        phase_name = phase['name']
        obs_normalized = batch["observations"].float() / 255.0
        encoded_features = self.vae.observe_sequence(obs_normalized)
        if phase_name == 'converge_vae':
         decoded_features = self.vae.decode(encoded_features["latent"])
         reconstruction_loss = F.binary_cross_entropy_with_logits(decoded_features, obs_normalized, reduction='none')
         reconstruction_loss = reconstruction_loss.sum(dim=[2,3,4]).mean()
         kl_dim = 0.5 * (encoded_features["mean"]**2 + encoded_features["logvar"].exp() - 1.0 - encoded_features["logvar"])
         kl_sum = kl_dim.sum(dim=-1)
         kl_loss = torch.clamp(kl_sum, min = self.kl_free_bits).mean()
         #kl_loss = torch.clamp(kl_loss, min = self.kl_free_bits)
         progress = min(1.0, phase['updates_done'] / phase['duration_updates'])
         beta = self.beta * progress
         vae_loss = reconstruction_loss + beta * (kl_loss)
         total_loss = vae_loss

        elif phase_name == 'converge_mdn':
         encoded_features["latent"] = encoded_features["latent"].detach()
         latents = self.mdn_rnn.observe_sequence(encoded_features["latent"][:, :-1], batch["actions"][:, :-1], batch["dones"][:, :-1])
         pi_probs = F.softmax(latents["pi_logits"], dim=-1)
         targets = encoded_features["latent"][:, 1:]
         targets_expanded = targets.unsqueeze(2)
         var = torch.exp(latents["logvar"])
         log_prob = -0.5 * (np.log(2* np.pi) + latents["logvar"] + ((targets_expanded - latents["mu"])**2) / var)
         log_prob = log_prob.sum(dim=-1)
         log_pi = torch.log(pi_probs + 1e-8)
         log_weighted = log_prob + log_pi
         max_log_weighted, _ = torch.max(log_weighted, dim=-1, keepdim=True)
         log_sum_exp = max_log_weighted + torch.log(torch.sum(torch.exp(log_weighted - max_log_weighted), dim=-1, keepdim=True))
         log_sum_exp =  log_sum_exp.squeeze(-1)
         nll = -log_sum_exp
         latents_loss = nll.mean()
         reward_targets = batch["rewards"][:, 1:]
         reward_predictions = latents["reward_preds"]
         reward_loss = F.mse_loss(reward_predictions.squeeze(-1), reward_targets)
         done_targets = batch["dones"][:, 1:].float()
         done_predictions = latents["done_logits"]
         done_loss = F.binary_cross_entropy_with_logits(done_predictions.squeeze(-1), done_targets)
         dynamics_loss = latents_loss + reward_loss + done_loss
         total_loss = dynamics_loss
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        self.world_model_optimizer.step()
        if phase_name == 'converge_vae':
         return {"world_model/reconstruction_loss": reconstruction_loss.item(),
         "world_model/kl_loss": kl_loss.item(),
         "world_model/total_loss": vae_loss.item()}
        if phase_name == 'converge_mdn':
         return {"world_model/latents_loss": latents_loss.item(),
         "world_model/reward_loss": reward_loss.item(),
         "world_model/done_loss": done_loss.item(),
         "world_model/dynamics_loss": dynamics_loss.item(),
         "world_model/total_loss": total_loss.item()}
    


    def update_controller(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        """Train the CMA-ES controller inside dream rollouts."""
        
        raise NotImplementedError("update_controller homework pending.")

    # ------------------------------------------------------------------
    # Helper wiring utilities
    # ------------------------------------------------------------------
    def _bind_context(self, context: WorkflowContext) -> None:
        """Extract reusable handles from the WorkflowContext."""
        self.config = context.config
        self.device = context.device
        self.environment = context.train_environment
        self.eval_environment = context.eval_environment
        self.buffers = dict(context.buffers)
        self.controller_manager = context.controller_manager
        self.temperature = float(getattr(self.config.algorithm, "temperature", 1.0))
        self.beta = float(getattr(self.config.algorithm, "beta", 1.0))  # Beta for KL loss scaling
        self.kl_free_bits = float(getattr(self.config.algorithm, "kl_free_bits", 0.0))
        components = context.components
        # Single VAE component for both encoding and decoding
        self.vae = getattr(components, "vae", None)
        self.mdn_rnn = getattr(components, "dynamics_model", None)
        self.controller = (
            self.controller_manager.get("actor") if self.controller_manager and "actor" in self.controller_manager else None
        )

        optimizers = context.optimizers or {}
        self.world_model_optimizer = optimizers.get("world_model")
        self.controller_optimizer = optimizers.get("actor")

        dims_cfg = getattr(self.config, "_dims", None)
        self.action_dim = int(getattr(dims_cfg, "action", 0) or 0)
        self.z_dim = int(getattr(dims_cfg, "representation", 0) or 0)
        self.num_envs = int(getattr(self.environment, "num_envs", 1) or 1)

        if self.vae is None or self.mdn_rnn is None:
            raise RuntimeError("OriginalWorldModelsWorkflow requires 'vae' and 'mdn_rnn' components.")
        if self.controller is None:
            raise RuntimeError("OriginalWorldModelsWorkflow requires 'actor' controller.")
        if self.world_model_optimizer is None:
            raise RuntimeError("OriginalWorldModelsWorkflow requires 'world_model' optimizer.")
        # controller_optimizer is optional (not needed for random policy)
        # TODO: Pull rollout hyperparameters (collect length, dream horizon, etc.) from config

    def _reset_rollout_state(
        self,
        *,
        initial_obs: Optional[Any],
        initial_dones: Optional[Any],
    ) -> None:
        """Helper to sync current observation/done buffers with the environment."""
        if initial_obs is None or initial_dones is None:
            raise ValueError("WorkflowContext missing initial observation/done snapshots.")
        obs_array = np.asarray(initial_obs, dtype=np.uint8)
        dones_array = np.asarray(initial_dones, dtype=bool)
        self.current_obs = obs_array
        self.current_dones = dones_array
        self.collect_length = max(1, int(getattr(self.config.algorithm, "collect_length", 1)))
        self.hidden_state = self.mdn_rnn.reset_state(batch_size=self.num_envs)["hidden"]
        self.prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)


    def imagine(
        self,
        *,
        observations: Any = None,
        latent: Any = None,
        horizon: Optional[int] = None,
        deterministic: bool = False,
        controller_role: Optional[str] = None,
        action_sequence: Any = None,
    ) -> Dict[str, Any]:
        """Delegate to MDN-RNN to sample dream trajectories."""
        # TODO: Implement MDN-RNN rollout logic once the module and latent interface exist
        raise NotImplementedError("imagine homework pending.")

    # ------------------------------------------------------------------
    # Checkpoint state (researcher-defined custom state)
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Return custom state for checkpointing (episode tracking)."""
        return {
            "total_episodes": self.total_episodes,
            "episode_returns": list(self.episode_returns),
            "episode_lengths": list(self.episode_lengths),
            "vector_episode_returns": self.vector_episode_returns.tolist(),
            "vector_episode_lengths": self.vector_episode_lengths.tolist(),
            "num_envs": self.num_envs,
            "episode_history_len": self._episode_history_len,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore custom state from checkpoint."""
        if not state:
            return
        # Use the existing episode tracking restoration helper
        tracking_state = {
            "total_episodes": state.get("total_episodes", 0),
            "returns": state.get("episode_returns", []),
            "lengths": state.get("episode_lengths", []),
            "vector_returns": np.array(state.get("vector_episode_returns", []), dtype=np.float32),
            "vector_lengths": np.array(state.get("vector_episode_lengths", []), dtype=np.int32),
            "num_envs": state.get("num_envs", self.num_envs),
            "history_len": state.get("episode_history_len", self._episode_history_len),
        }
        self._restore_episode_tracking(tracking_state)
