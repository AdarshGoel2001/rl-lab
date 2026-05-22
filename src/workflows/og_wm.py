"""Original 2018 World Models workflow placeholder.

This file is intentionally left as a scaffold so we can gradually rebuild the
exact VAE + MDN-RNN + CMA-ES pipeline described by Ha & Schmidhuber (2018).
Treat the TODOs like homework checkpoints: fill them in one by one as features
come online.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import time
import numpy as np
import torch
import torch.nn.functional as F

from .utils.base import Batch, CollectResult, PhaseConfig, WorldModelWorkflow
from .utils.context import WorkflowContext


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
         metrics = {"world_model/reconstruction_loss": reconstruction_loss.item(),
         "world_model/kl_loss": kl_loss.item(),
         "world_model/total_loss": vae_loss.item()}

        elif phase_name == 'converge_mdn':
         encoded_features["latent"] = encoded_features["latent"].detach()
         latents = self.mdn_rnn.observe_sequence(
             encoded_features["latent"][:, :-1],
             batch["actions"][:, :-1],
             batch["dones"][:, :-1],
             temperature=self.temperature,
         )
         targets = encoded_features["latent"][:, 1:]
         targets_expanded = targets.unsqueeze(2)
         logvar = torch.clamp(latents["logvar"], min=-10.0, max=10.0)
         var = torch.exp(logvar)
         log_prob = -0.5 * (np.log(2* np.pi) + logvar + ((targets_expanded - latents["mu"])**2) / var)
         log_prob = log_prob.sum(dim=-1)
         log_pi = F.log_softmax(latents["pi_logits"], dim=-1)
         nll = -torch.logsumexp(log_prob + log_pi, dim=-1)
         latents_loss = nll.mean()
         reward_targets = batch["rewards"][:, 1:]
         reward_predictions = latents["reward_preds"]
         reward_loss = F.mse_loss(reward_predictions.squeeze(-1), reward_targets)
         done_targets = batch["dones"][:, 1:].float()
         done_predictions = latents["done_logits"]
         done_loss = F.binary_cross_entropy_with_logits(done_predictions.squeeze(-1), done_targets)
         dynamics_loss = latents_loss + reward_loss + done_loss
         total_loss = dynamics_loss
         metrics = {"world_model/latents_loss": latents_loss.item(),
         "world_model/reward_loss": reward_loss.item(),
         "world_model/done_loss": done_loss.item(),
         "world_model/dynamics_loss": dynamics_loss.item(),
         "world_model/total_loss": total_loss.item()}
        else:
         raise ValueError(f"Unknown OG World Models phase: {phase_name}")
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        self.world_model_optimizer.step()
        self._record_metrics(phase_name, phase, metrics)
        if phase_name == 'converge_vae':
         self._maybe_save_reconstruction_grid(obs_normalized, decoded_features, phase)
        return metrics
    


    def update_controller(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        """Intended behavior: evolve a small controller over imagined MDN-RNN rollouts."""
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
        self.artifact_dir = Path(context.checkpoint_manager.experiment_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv_path = self.artifact_dir / "metrics.csv"
        self.loss_curve_path = self.artifact_dir / "loss_curves.png"
        self.reconstruction_grid_path = self.artifact_dir / "reconstruction_grid.png"
        self._metrics_rows = []
        self._saved_reconstruction_grid = False
        self._publish_latest_run_path()

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
        """Intended behavior: roll latent state forward with MDN-RNN and a controller policy."""
        # TODO: Implement MDN-RNN rollout logic once the module and latent interface exist
        raise NotImplementedError("imagine homework pending.")

    def _publish_latest_run_path(self) -> None:
        """Write a stable pointer used by smoke tests and demos."""
        try:
            from hydra.core.hydra_config import HydraConfig

            root = Path(HydraConfig.get().runtime.cwd)
        except Exception:
            root = Path.cwd()

        runs_dir = root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "latest_og_wm_run.txt").write_text(str(self.artifact_dir.resolve()))

    def _record_metrics(self, phase_name: str, phase: PhaseConfig, metrics: Dict[str, float]) -> None:
        row = {
            "step": int(getattr(self, "_artifact_step", 0)),
            "phase": phase_name,
            "updates_done": int(phase.get("updates_done", 0)) + 1,
            "vae_recon_loss": metrics.get("world_model/reconstruction_loss", "") if phase_name == "converge_vae" else "",
            "vae_kl_loss": metrics.get("world_model/kl_loss", "") if phase_name == "converge_vae" else "",
            "vae_total_loss": metrics.get("world_model/total_loss", "") if phase_name == "converge_vae" else "",
            "mdn_latent_nll": metrics.get("world_model/latents_loss", "") if phase_name == "converge_mdn" else "",
            "mdn_reward_loss": metrics.get("world_model/reward_loss", "") if phase_name == "converge_mdn" else "",
            "mdn_done_loss": metrics.get("world_model/done_loss", "") if phase_name == "converge_mdn" else "",
            "mdn_total_loss": metrics.get("world_model/dynamics_loss", "") if phase_name == "converge_mdn" else "",
        }
        self._artifact_step = row["step"] + 1
        self._metrics_rows.append(row)

        fieldnames = list(row.keys())
        write_header = not self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._save_loss_curves()

    def _save_loss_curves(self) -> None:
        if not self._metrics_rows:
            return
        os.environ.setdefault("MPLCONFIGDIR", str(self.artifact_dir / ".matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(self.artifact_dir / ".cache"))
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        numeric_keys = [
            "vae_recon_loss",
            "vae_kl_loss",
            "vae_total_loss",
            "mdn_latent_nll",
            "mdn_reward_loss",
            "mdn_done_loss",
            "mdn_total_loss",
        ]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        steps = [row["step"] for row in self._metrics_rows]
        for key in numeric_keys:
            values = [row[key] for row in self._metrics_rows]
            xy = [(step, float(value)) for step, value in zip(steps, values) if value != ""]
            if not xy:
                continue
            xs, ys = zip(*xy)
            ax.plot(xs, ys, marker="o", label=key)
        ax.set_xlabel("world model update")
        ax.set_ylabel("loss")
        ax.set_title("OG World Models smoke losses")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(self.loss_curve_path, dpi=150)
        plt.close(fig)

    def _maybe_save_reconstruction_grid(
        self,
        obs_normalized: torch.Tensor,
        decoded_logits: torch.Tensor,
        phase: PhaseConfig,
    ) -> None:
        duration_updates = int(phase.get("duration_updates", 0) or 0)
        updates_done = int(phase.get("updates_done", 0) or 0) + 1
        if self._saved_reconstruction_grid or duration_updates <= 0 or updates_done < duration_updates:
            return

        os.environ.setdefault("MPLCONFIGDIR", str(self.artifact_dir / ".matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", str(self.artifact_dir / ".cache"))
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        originals = obs_normalized.detach().cpu().reshape(-1, *obs_normalized.shape[2:])
        reconstructions = torch.sigmoid(decoded_logits.detach()).cpu().reshape(
            -1, *decoded_logits.shape[2:]
        )
        count = min(8, originals.shape[0])
        if count == 0:
            return

        fig, axes = plt.subplots(2, count, figsize=(count * 1.4, 3.0))
        if count == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for idx in range(count):
            original = originals[idx].numpy()
            reconstruction = reconstructions[idx].numpy()
            if original.shape[-1] == 1:
                original = original[..., 0]
                reconstruction = reconstruction[..., 0]
                cmap = "gray"
            else:
                cmap = None
            axes[0, idx].imshow(np.clip(original, 0.0, 1.0), cmap=cmap)
            axes[1, idx].imshow(np.clip(reconstruction, 0.0, 1.0), cmap=cmap)
            axes[0, idx].axis("off")
            axes[1, idx].axis("off")
        axes[0, 0].set_ylabel("original", fontsize=8)
        axes[1, 0].set_ylabel("recon", fontsize=8)
        fig.suptitle("VAE reconstruction smoke grid", fontsize=10)
        fig.tight_layout()
        fig.savefig(self.reconstruction_grid_path, dpi=150)
        plt.close(fig)
        self._saved_reconstruction_grid = True

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
