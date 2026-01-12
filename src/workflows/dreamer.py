"""Dreamer workflow implementation for orchestrated world-model training."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Distribution, kl_divergence
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ..utils.config import Config
from .utils.base import Batch, CollectResult, PhaseConfig, WorldModelWorkflow
from .utils.context import WorkflowContext, WorldModelComponents
from .utils.controllers import ControllerManager
from ..components.representation_learners import (
    LatentState,
    LatentStep,
    LatentSequence,
)

logger = logging.getLogger(__name__)


class DreamerWorkflow(WorldModelWorkflow):
    """Skeleton workflow that will gradually replace the legacy Dreamer trainer."""

    def __init__(self, *, config: Optional[Config] = None) -> None:
        super().__init__()
        self._override_config = config
        self._bound_config: Optional[Config] = None
        self.device: Optional[str] = None
        self.buffers: Dict[str, Any] = {}
        self.environment = None
        self.eval_environment = None
        self.controller_manager: Optional[ControllerManager] = None
        self.actor_controller: Optional[torch.nn.Module] = None
        self.critic_controller: Optional[torch.nn.Module] = None

        self.encoder: Optional[torch.nn.Module] = None
        self.rssm: Optional[torch.nn.Module] = None
        self.dynamics_model: Optional[torch.nn.Module] = None
        self.reward_predictor: Optional[torch.nn.Module] = None
        self.observation_decoder: Optional[torch.nn.Module] = None

        self.world_model_optimizer: Optional[torch.optim.Optimizer] = None

        self.actor_warmup_updates = 0
        self.critic_warmup_updates = 0
        self.world_model_updates = 0
        self.max_grad_norm = None

        self.current_obs = None
        self.current_dones = None
        self.is_vectorized = False
        self.num_envs = 1
        self.start_time: Optional[float] = None
        self._rssm_state = None
        self._prev_actions_model: Optional[torch.Tensor] = None
        self.action_dim = 0
        self.discrete_actions = False
        self.collect_length = 1
        self.gamma = 0.99
        self.lambda_return = 0.95
        self.entropy_coef = 0.0
        self.imagination_horizon = 15
        self.free_nats = 0.0
        self.kl_scale = 1.0

    @property
    def config(self) -> Config:
        if self._override_config is not None:
            return self._override_config
        if self._bound_config is None:
            raise RuntimeError("DreamerWorkflow accessed config before initialization.")
        return self._bound_config

    def initialize(self, context: WorkflowContext) -> None:
        logger.info("Initializing Dreamer workflow from orchestrator context")
        self._bind_context(context)

        self._reset_episode_tracking(self.num_envs, clear_history=True)
        self._reset_rollout_state(
            initial_obs=context.initial_observation,
            initial_dones=context.initial_dones,
        )

        training_cfg = self.config.training
        policy_warmup = int(getattr(training_cfg, "policy_warmup_updates", 0) or 0)
        actor_warmup = int(getattr(training_cfg, "actor_warmup_updates", 0) or 0)
        critic_warmup = int(getattr(training_cfg, "critic_warmup_updates", 0) or 0)

        self.actor_warmup_updates = max(actor_warmup, policy_warmup, 0)
        self.critic_warmup_updates = max(critic_warmup, policy_warmup, 0)
        self.world_model_updates = 0
        self.max_grad_norm = getattr(self.config.algorithm, "max_grad_norm", None)
        self.start_time = time.time()
        self.collect_length = int(
            getattr(self.config.algorithm, "collect_length", 0)
            or getattr(self.config.buffer, "sequence_length", 1)
        )
        algo_cfg = self.config.algorithm
        self.gamma = float(getattr(algo_cfg, "gamma", 0.99))
        self.lambda_return = float(getattr(algo_cfg, "lambda_return", 0.95))
        self.entropy_coef = float(getattr(algo_cfg, "entropy_coef", 0.0))
        self.imagination_horizon = int(getattr(algo_cfg, "imagination_horizon", 15))
        self.free_nats = float(getattr(algo_cfg, "free_nats", 0.0))
        self.kl_scale = float(getattr(algo_cfg, "kl_scale", 1.0))

    # ------------------------------------------------------------------
    # Internal setup helpers
    # ------------------------------------------------------------------
    def _bind_context(self, context: WorkflowContext) -> None:
        self._bound_config = context.config
        self.device = context.device
        self.buffers = dict(context.buffers)
        self.environment = context.train_environment
        self.eval_environment = context.eval_environment

        components = context.components

        self.encoder = components.encoder
        self.rssm = components.representation_learner
        self.dynamics_model = components.dynamics_model
        self.reward_predictor = components.reward_predictor
        self.observation_decoder = components.observation_decoder

        if context.controller_manager is None:
            raise RuntimeError("DreamerWorkflow requires a controller manager in the workflow context.")
        self.controller_manager = context.controller_manager
        self.actor_controller = self.controller_manager.get("actor")
        self.critic_controller = self.controller_manager.get("critic")
        if self.actor_controller is None or self.critic_controller is None:
            raise RuntimeError("DreamerWorkflow requires 'actor' and 'critic' controllers in the workflow context.")

        dims_cfg = getattr(self.config, "_dims", None)
        action_dim = getattr(dims_cfg, "action", None) if dims_cfg is not None else None
        if action_dim is None:
            raise RuntimeError("Workflow config missing '_dims.action'; cannot infer action dimension.")
        self.action_dim = int(action_dim)

        controllers_cfg = getattr(self.config, "controllers", None)
        actor_cfg = getattr(controllers_cfg, "actor", None) if controllers_cfg is not None else None
        discrete_actions = False
        if actor_cfg is not None:
            if isinstance(actor_cfg, Mapping):
                discrete_actions = bool(actor_cfg.get("discrete_actions", False))
            else:
                discrete_actions = bool(getattr(actor_cfg, "discrete_actions", False))
        self.discrete_actions = discrete_actions

        self.is_vectorized = getattr(self.environment, "is_vectorized", False)
        self.num_envs = int(getattr(self.environment, "num_envs", 1) or 1)

        optimizers = context.optimizers or {}
        self.world_model_optimizer = optimizers.get("world_model")
        if self.world_model_optimizer is None:
            raise RuntimeError("DreamerWorkflow requires a 'world_model' optimizer supplied via workflow context.")

        param_groups = getattr(self.world_model_optimizer, "param_groups", None)
        if not param_groups or not param_groups[0].get("params"):
            raise RuntimeError("World-model optimizer supplied to workflow does not manage any parameters.")

        self.collect_length = max(1, int(self.collect_length))

    def _reset_rollout_state(
        self,
        *,
        initial_obs: Optional[Any] = None,
        initial_dones: Optional[Any] = None,
    ) -> None:
        if self.environment is None:
            raise RuntimeError("Environment not bound before resetting rollout state.")

        if initial_obs is None:
            raise ValueError("Initial observation snapshot missing from workflow context.")
        if isinstance(initial_obs, tuple):
            initial_obs = initial_obs[0]

        obs_array = np.asarray(initial_obs, dtype=np.float32)
        if obs_array.shape[0] != self.num_envs:
            raise ValueError(
                f"Environment reset returned batch dimension {obs_array.shape[0]}, expected {self.num_envs}."
            )
        self.current_obs = obs_array

        if initial_dones is not None:
            dones_array = np.asarray(initial_dones, dtype=bool)
        else:
            raise ValueError("Initial done mask missing from workflow context.")
        if dones_array.shape[0] != self.num_envs:
            raise ValueError(
                f"Initial done mask provided batch dimension {dones_array.shape[0]}, expected {self.num_envs}."
            )
        self.current_dones = dones_array

        self._reset_episode_tracking(self.num_envs, clear_history=False)

        self._prev_actions_model = torch.zeros(
            self.num_envs,
            self.action_dim,
            device=self.device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _collect_metrics(self, rewards: np.ndarray, duration: float) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "reward/mean": float(np.mean(rewards)),
            "collect/duration": float(duration),
            "collect/throughput_fps": float((rewards.size / duration) if duration > 0 else 0.0),
            "episodes/total": float(self.total_episodes),
        }
        if self.episode_returns:
            metrics["return/rolling_mean"] = float(np.mean(self.episode_returns))
        if self.episode_lengths:
            metrics["episode_length/rolling_mean"] = float(np.mean(self.episode_lengths))
        return metrics

    def _metrics_state(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "workflow/world_model_updates": float(self.world_model_updates),
            "workflow/episodes_total": float(self.total_episodes),
        }
        if self.episode_returns:
            metrics["workflow/return_mean"] = float(np.mean(self.episode_returns))
            metrics["workflow/return_std"] = float(np.std(self.episode_returns))
            metrics["workflow/return_max"] = float(np.max(self.episode_returns))
            metrics["workflow/return_min"] = float(np.min(self.episode_returns))
        if self.episode_lengths:
            metrics["workflow/episode_length_mean"] = float(np.mean(self.episode_lengths))
        if self.start_time is not None:
            metrics["workflow/runtime_seconds"] = float(time.time() - self.start_time)
        return metrics

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: float,
        lam: float,
    ) -> torch.Tensor:
        horizon = rewards.shape[1]
        returns = torch.zeros_like(rewards)
        next_return = bootstrap
        for t in reversed(range(horizon)):
            value = values[:, t]
            next_return = rewards[:, t] + discount * ((1 - lam) * value + lam * next_return)
            returns[:, t] = next_return
        return returns

    def collect_step(
        self,
        step: int,
        *,
        phase: PhaseConfig,
    ) -> Optional[CollectResult]:
        if self.environment is None or self.encoder is None or self.rssm is None or self.actor_controller is None:
            raise RuntimeError("DreamerWorkflow not fully initialised before collect_step.")

        collect_len = int(phase.get("collect_length", self.collect_length) or self.collect_length)
        deterministic = bool(phase.get("deterministic_policy", False))
        start_time = time.time()

        obs_list: list[np.ndarray] = []
        next_obs_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        rewards_list: list[np.ndarray] = []
        dones_list: list[np.ndarray] = []
        values_list: list[np.ndarray] = []
        log_probs_list: list[np.ndarray] = []

        rewards_for_metrics = []

        for _ in range(collect_len):
            obs_np = np.asarray(self.current_obs, dtype=np.float32)
            obs_list.append(obs_np.copy())

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32)
                reset_mask = None
                if self.current_dones is not None:
                    reset_mask = torch.as_tensor(self.current_dones, device=self.device, dtype=torch.bool)

                # Encode observations to features before passing to RSSM
                features = self.encoder(obs_tensor)

                latent_step: LatentStep = self.rssm.observe(
                    features,
                    prev_action=self._prev_actions_model,
                    reset_mask=reset_mask,
                    detach_posteriors=True,
                )
                latent_tensor = latent_step.posterior.to_tensor()

                actor_dist: Distribution = self.actor_controller.forward(latent_tensor)

                if self.discrete_actions:
                    if deterministic:
                        action_indices = torch.argmax(actor_dist.probs, dim=-1)
                    else:
                        action_indices = actor_dist.sample()
                    action_tensor = F.one_hot(action_indices, num_classes=self.action_dim).to(
                        dtype=torch.float32
                    )
                    env_actions = action_indices.detach().cpu().numpy()
                    log_prob = actor_dist.log_prob(action_indices)
                else:
                    action_tensor = actor_dist.mean if deterministic else actor_dist.rsample()
                    env_actions = action_tensor.detach().cpu().numpy()
                    log_prob = actor_dist.log_prob(action_tensor)

                value_pred = self.critic_controller.forward(latent_tensor).detach()

            next_obs, reward, done, infos = self.environment.step(env_actions)

            self._update_episode_stats(reward, done, infos)

            rewards_for_metrics.append(np.asarray(reward, dtype=np.float32))
            rewards_list.append(np.asarray(reward, dtype=np.float32))
            dones_list.append(np.asarray(done, dtype=bool))
            values_list.append(value_pred.cpu().numpy())
            log_probs_list.append(log_prob.detach().cpu().numpy())
            actions_list.append(action_tensor.detach().cpu().numpy())
            next_obs_list.append(np.asarray(next_obs, dtype=np.float32))

            self.current_obs = np.asarray(next_obs, dtype=np.float32)
            if self.current_obs.shape[0] != self.num_envs:
                raise ValueError(
                    f"Environment step returned batch dimension {self.current_obs.shape[0]}, expected {self.num_envs}."
                )
            self.current_dones = np.asarray(done, dtype=bool)
            if self.current_dones.shape[0] != self.num_envs:
                raise ValueError(
                    f"Environment step returned done batch dimension {self.current_dones.shape[0]}, expected {self.num_envs}."
                )

            self._prev_actions_model = action_tensor.detach()

        duration = max(time.time() - start_time, 1e-9)
        rewards_stack = np.stack(rewards_for_metrics, axis=0)
        metrics = self._collect_metrics(rewards_stack, duration)

        trajectory = {
            "observations": np.stack(obs_list, axis=0),
            "next_observations": np.stack(next_obs_list, axis=0),
            "actions": np.stack(actions_list, axis=0),
            "rewards": np.stack(rewards_list, axis=0),
            "dones": np.stack(dones_list, axis=0),
            "values": np.stack(values_list, axis=0),
            "log_probs": np.stack(log_probs_list, axis=0),
        }

        return CollectResult(
            episodes=0,
            steps=int(collect_len * self.num_envs),
            metrics=metrics,
            trajectory=trajectory,
        )

    def update_world_model(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        if self.world_model_optimizer is None or self.encoder is None or self.rssm is None:
            raise RuntimeError("World-model components not ready before update.")

        device = self.device or "cpu"
        tensor_batch: Dict[str, torch.Tensor] = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device))
            for key, value in batch.items()
        }

        observations = tensor_batch["observations"].to(torch.float32)  # (B, T, ...)
        actions = tensor_batch.get("actions")
        dones = tensor_batch.get("dones")

        if observations.dim() < 3:
            raise ValueError("Expected batched observation sequences with shape (B, T, ...).")

        batch_size, horizon = observations.shape[0], observations.shape[1]
        obs_flat = observations.reshape(batch_size * horizon, *observations.shape[2:])
        features_flat = self.encoder(obs_flat)
        feature_dim = features_flat.shape[-1]
        features = features_flat.reshape(batch_size, horizon, feature_dim)

        if dones is not None:
            dones = dones.to(torch.bool)

        sequence: LatentSequence = self.rssm.observe_sequence(features, actions=actions, dones=dones)
        posterior = sequence.posterior
        prior = sequence.prior

        latent_feat = posterior.to_tensor()
        latent_flat = latent_feat.reshape(batch_size * horizon, -1)

        losses: Dict[str, torch.Tensor] = {}

        if self.observation_decoder is not None:
            target = obs_flat
            recon = self.observation_decoder(latent_flat)
            losses["reconstruction"] = F.mse_loss(recon, target)

        if self.reward_predictor is not None and "rewards" in tensor_batch:
            rewards = tensor_batch["rewards"].to(torch.float32).reshape(batch_size * horizon, -1)
            pred_rewards = self.reward_predictor(latent_flat)
            losses["reward"] = F.mse_loss(pred_rewards, rewards)

        prior_dist = sequence.prior_dist if prior is not None else None
        posterior_dist = sequence.posterior_dist
        if prior_dist is None:
            raise RuntimeError("RSSM observe_sequence did not return prior states.")
        kl_values = kl_divergence(posterior_dist, prior_dist)
        if self.free_nats > 0.0:
            kl_values = torch.clamp(kl_values - self.free_nats, min=0.0)
        kl_loss = kl_values.mean()
        losses["kl"] = kl_loss * self.kl_scale

        total_loss = sum(losses.values())

        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.world_model_optimizer.param_groups[0]["params"], self.max_grad_norm)
        self.world_model_optimizer.step()
        self.world_model_updates += 1

        metrics = {
            "world_model/total_loss": float(total_loss.item()),
            "world_model/kl_loss": float(kl_loss.item()),
        }
        if "reconstruction" in losses:
            metrics["world_model/recon_loss"] = float(losses["reconstruction"].item())
        if "reward" in losses:
            metrics["world_model/reward_loss"] = float(losses["reward"].item())
        return metrics

    def update_controller(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        if self.world_model_updates < self.actor_warmup_updates:
            return {"controller/skipped": 1.0}

        device = self.device or "cpu"
        tensor_batch: Dict[str, torch.Tensor] = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device))
            for key, value in batch.items()
        }

        observations = tensor_batch["observations"].to(torch.float32)
        actions = tensor_batch.get("actions")
        dones = tensor_batch.get("dones")

        batch_size, horizon = observations.shape[0], observations.shape[1]
        obs_flat = observations.reshape(batch_size * horizon, *observations.shape[2:])
        with torch.no_grad():
            features_flat = self.encoder(obs_flat)
            feature_dim = features_flat.shape[-1]
            features = features_flat.reshape(batch_size, horizon, feature_dim)
            dones_seq = dones.to(torch.bool) if dones is not None else None
            seq = self.rssm.observe_sequence(features, actions=actions, dones=dones_seq)
            start_state = seq.last_posterior.detach()

        imagination_horizon = int(phase.get("imagination_horizon", self.imagination_horizon))
        rollout = self.imagine(latent=start_state, horizon=imagination_horizon, deterministic=False)

        states_feat = rollout["states"]  # (B, H, latent_dim)
        rewards = rollout["rewards"]  # (B, H)
        values = rollout["values"]  # (B, H)
        log_probs = rollout["log_probs"]  # (B, H)
        entropies = rollout["entropies"]  # (B, H)
        bootstrap = rollout["bootstrap"]  # (B,)

        returns = self._compute_lambda_returns(
            rewards,
            values.detach(),
            bootstrap.detach(),
            self.gamma,
            self.lambda_return,
        )
        advantages = returns - values.detach()

        actor_loss = -(advantages.detach() * log_probs).mean() - self.entropy_coef * entropies.mean()
        actor_optimizer = self.actor_controller.optimizer
        actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_grad_clip = getattr(self.actor_controller, "grad_clip", None)
        if actor_grad_clip is not None:
            clip_grad_norm_(self.actor_controller.parameters(), actor_grad_clip)
        actor_optimizer.step()

        critic_loss = F.mse_loss(values, returns.detach())
        critic_optimizer = self.critic_controller.optimizer
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_clip = getattr(self.critic_controller, "grad_clip", None)
        if critic_grad_clip is not None:
            clip_grad_norm_(self.critic_controller.parameters(), critic_grad_clip)
        critic_optimizer.step()

        metrics = {
            "controller/actor_loss": float(actor_loss.item()),
            "controller/critic_loss": float(critic_loss.item()),
            "controller/entropy": float(entropies.mean().item()),
        }
        return metrics

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
        if self.rssm is None or self.actor_controller is None:
            raise RuntimeError("RSSM or actor controller not initialised for imagination.")

        horizon = int(horizon or self.imagination_horizon)
        if horizon <= 0:
            raise ValueError("Imagination horizon must be positive.")

        if latent is None and observations is None:
            raise ValueError("Either latent or observations must be provided for imagination.")

        if latent is None:
            obs_tensor = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
            prev_action = self._prev_actions_model if self._prev_actions_model is not None else None
            latent_step = self.rssm.observe(obs_tensor, prev_action=prev_action, detach_posteriors=True)
            latent_state = latent_step.posterior.detach()
        elif isinstance(latent, LatentState):
            latent_state = latent.to(self.device)
        else:
            raise TypeError(f"Unsupported latent type for imagination: {type(latent)}")

        batch_size = latent_state.to_tensor().shape[0]

        states_feat = []
        rewards = []
        values = []
        log_probs = []
        entropies = []
        actions = []

        state = latent_state
        for _ in range(horizon):
            latent_tensor = state.to_tensor()
            actor_dist: Distribution = self.actor_controller.forward(latent_tensor)
            if self.discrete_actions:
                if deterministic:
                    action_indices = torch.argmax(actor_dist.probs, dim=-1)
                else:
                    action_indices = actor_dist.sample()
                action = F.one_hot(action_indices, num_classes=self.action_dim).to(
                    dtype=torch.float32
                )
                log_probs.append(actor_dist.log_prob(action_indices))
            else:
                action = actor_dist.mean if deterministic else actor_dist.rsample()
                log_probs.append(actor_dist.log_prob(action))

            critic_value = self.critic_controller.forward(latent_tensor)
            values.append(critic_value)
            entropy = actor_dist.entropy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)
            entropies.append(entropy)
            actions.append(action)
            states_feat.append(latent_tensor)

            reward_pred = (
                self.reward_predictor(latent_tensor)
                if self.reward_predictor is not None
                else torch.zeros(batch_size, device=self.device)
            )
            rewards.append(reward_pred.squeeze(-1))

            state = self.rssm.imagine_step(state, action, deterministic=deterministic)

        bootstrap_value = self.critic_controller.forward(state.to_tensor())

        states_tensor = torch.stack(states_feat, dim=1)
        rewards_tensor = torch.stack(rewards, dim=1)
        values_tensor = torch.stack([v.view(batch_size) for v in values], dim=1)
        log_probs_tensor = torch.stack([lp.view(batch_size) for lp in log_probs], dim=1)
        entropies_tensor = torch.stack([e.view(batch_size) for e in entropies], dim=1)

        return {
            "states": states_tensor,
            "actions": torch.stack(actions, dim=1),
            "rewards": rewards_tensor,
            "values": values_tensor,
            "log_probs": log_probs_tensor,
            "entropies": entropies_tensor,
            "bootstrap": bootstrap_value.view(batch_size),
        }

    def state_dict(self, *, mode: str = "checkpoint") -> Dict[str, Any]:
        if mode == "metrics":
            return self._metrics_state()
        if mode != "checkpoint":
            raise ValueError(f"DreamerWorkflow does not support state_dict mode '{mode}'.")
        if self.world_model_optimizer is None:
            raise RuntimeError("World-model optimizer not initialised before saving state.")
        payload = {
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "world_model_updates": self.world_model_updates,
        }
        if self.controller_manager is not None:
            payload["controllers"] = self.controller_manager.state_dict()
        return payload

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        if self.world_model_optimizer and state.get("world_model_optimizer"):
            self.world_model_optimizer.load_state_dict(state["world_model_optimizer"])
        self.world_model_updates = int(state.get("world_model_updates", 0))
        controller_snapshot = state.get("controllers")
        if controller_snapshot and self.controller_manager is not None:
            self.controller_manager.load_state_dict(controller_snapshot)

   
