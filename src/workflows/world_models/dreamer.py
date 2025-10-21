"""Dreamer workflow implementation for orchestrated world-model training."""

from __future__ import annotations

import logging
import time
import random
from collections import deque
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Distribution, kl_divergence
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ...utils.config import Config
from .base import Batch, CollectResult, PhaseConfig, WorldModelWorkflow
from .context import WorkflowContext, WorldModelComponents
from .controllers import ControllerManager
from ...components.world_models.representation_learners import (
    RSSMState,
    LatentStep,
    LatentSequence,
)

logger = logging.getLogger(__name__)


class DreamerWorkflow(WorldModelWorkflow):
    """Skeleton workflow that will gradually replace the legacy Dreamer trainer."""

    def __init__(self, *, config: Optional[Config] = None) -> None:
        self._override_config = config

        self.context: Optional[WorkflowContext] = None
        self.device: Optional[str] = None
        self.components: Optional[WorldModelComponents] = None
        self.buffer = None
        self.environment = None
        self.eval_environment = None
        self.controllers: Dict[str, Any] = {}
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

        self.episode_returns: deque[float] = deque(maxlen=100)
        self.episode_lengths: deque[int] = deque(maxlen=100)
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        self.vector_episode_returns: Optional[np.ndarray] = None
        self.vector_episode_lengths: Optional[np.ndarray] = None
        self.total_episodes = 0
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
        if self.context is None:
            raise RuntimeError("DreamerWorkflow accessed config before initialization.")
        return self.context.config

    def initialize(self, context: WorkflowContext) -> None:
        logger.info("Initializing Dreamer workflow from orchestrator context")
        self._bind_context(context)

        self._prepare_optimizers()
        self._set_seeds(self.config.experiment.seed)
        self._reset_rollout_state()

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
        self.context = context
        self.device = context.device
        self.components = context.components
        self.buffer = context.buffer
        self.environment = context.train_environment
        self.eval_environment = context.eval_environment

        self.encoder = self.components.encoder
        self.rssm = self.components.representation_learner
        self.dynamics_model = self.components.dynamics_model
        self.reward_predictor = self.components.reward_predictor
        self.observation_decoder = self.components.observation_decoder

        self.controllers = context.controllers or {}
        self.controller_manager = context.controller_manager
        self.actor_controller = self._resolve_controller(["actor", "policy"])
        self.critic_controller = self._resolve_controller(["critic", "value", "value_function"])

        self.is_vectorized = getattr(self.environment, "is_vectorized", False)
        self.num_envs = int(getattr(self.environment, "num_envs", 1) or 1)

        specs = getattr(self.components, "specs", {}) or {}
        if self.actor_controller is not None:
            self.action_dim = getattr(self.actor_controller, "action_dim", specs.get("action_dim", 0))
            self.discrete_actions = bool(getattr(self.actor_controller, "discrete_actions", specs.get("discrete_actions", False)))
        else:
            self.action_dim = int(specs.get("action_dim", 0))
            self.discrete_actions = bool(specs.get("discrete_actions", False))

        if not self.action_dim and hasattr(self.environment, "action_space"):
            action_space = self.environment.action_space
            if getattr(action_space, "discrete", False):
                self.discrete_actions = True
                self.action_dim = int(getattr(action_space, "n", 1))
            else:
                shape = getattr(action_space, "shape", None) or (1,)
                self.action_dim = int(np.prod(shape))

        self.collect_length = max(1, int(self.collect_length))

    def _prepare_optimizers(self) -> None:
        params = []
        for module in (
            self.encoder,
            self.rssm,
            self.reward_predictor,
            self.observation_decoder,
        ):
            if module is None:
                continue
            params.extend(list(module.parameters()))

        if not params:
            raise RuntimeError("DreamerWorkflow could not gather parameters for world-model optimizer.")

        lr = float(getattr(self.config.algorithm, "world_model_lr", 2e-4))
        betas = getattr(self.config.algorithm, "world_model_betas", (0.9, 0.999))
        weight_decay = float(getattr(self.config.algorithm, "world_model_weight_decay", 0.0))
        self.world_model_optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

    def _set_seeds(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(self.environment, "reset"):
            try:
                self.environment.reset(seed=seed)
            except TypeError:
                # Some envs expect seed via config, ignore silently
                pass

    def _reset_rollout_state(self) -> None:
        if self.environment is None:
            raise RuntimeError("Environment not bound before resetting rollout state.")
        initial_obs = self.environment.reset()
        if isinstance(initial_obs, tuple):
            initial_obs = initial_obs[0]
        self.current_obs = np.asarray(initial_obs, dtype=np.float32)
        if self.current_obs.ndim == 1:
            self.current_obs = np.expand_dims(self.current_obs, axis=0)
        self.current_dones = np.zeros(self.current_obs.shape[0], dtype=bool)
        self._prev_actions_model = torch.zeros(
            self.current_obs.shape[0],
            self.action_dim if self.action_dim else 1,
            device=self.device,
            dtype=torch.float32,
        )
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        if self.is_vectorized:
            self.vector_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self.vector_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)

    def _resolve_controller(self, roles: list[str]) -> Optional[torch.nn.Module]:
        if self.controller_manager is None:
            for role in roles:
                controller = self.controllers.get(role)
                if controller is not None:
                    return controller
            return None
        for role in roles:
            try:
                return self.controller_manager.get(role)
            except KeyError:
                continue
        return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _latent_to_tensor(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deterministic, state.stochastic], dim=-1)

    def _prepare_action_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        if actions is None:
            return torch.zeros(1, self.action_dim, device=self.device)
        if actions.dtype.is_floating_point():
            return actions
        # assume discrete indices
        one_hot = F.one_hot(actions.to(torch.long), num_classes=self.action_dim).to(torch.float32)
        return one_hot

    def _action_to_env(self, action: torch.Tensor) -> np.ndarray:
        if self.discrete_actions:
            if action.dim() > 1:
                return torch.argmax(action, dim=-1).cpu().numpy()
            return action.long().cpu().numpy()
        return action.cpu().numpy()

    def _action_to_rssm(self, action: torch.Tensor) -> torch.Tensor:
        if self.discrete_actions:
            if action.dtype.is_floating_point() and action.shape[-1] == self.action_dim:
                return action.to(torch.float32)
            return F.one_hot(action.long(), num_classes=self.action_dim).to(torch.float32)
        if action.dim() == 1:
            return action.unsqueeze(-1)
        return action

    def _prepare_action_sequence(self, actions: torch.Tensor) -> torch.Tensor:
        if self.discrete_actions:
            if actions.dim() == 2:
                return F.one_hot(actions.long(), num_classes=self.action_dim).to(torch.float32)
            if actions.dim() == 3 and actions.shape[-1] == 1:
                actions = actions.squeeze(-1)
                return F.one_hot(actions.long(), num_classes=self.action_dim).to(torch.float32)
        if actions.dim() == 2:
            return actions.unsqueeze(-1).to(torch.float32)
        return actions.to(torch.float32)

    def _sample_action(self, dist: Distribution, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            if hasattr(dist, "mode"):
                mode = dist.mode
                return mode() if callable(mode) else mode
            if hasattr(dist, "mean"):
                mean = dist.mean
                return mean() if callable(mean) else mean
        if getattr(dist, "has_rsample", False):
            return dist.rsample()
        return dist.sample()

    def _update_episode_stats(self, rewards: np.ndarray, dones: np.ndarray, infos: list[dict[str, Any]]) -> None:
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        if self.is_vectorized:
            assert self.vector_episode_returns is not None
            assert self.vector_episode_lengths is not None
            self.vector_episode_returns += rewards
            self.vector_episode_lengths += 1
            for idx, done in enumerate(dones):
                if done:
                    ep_return = float(self.vector_episode_returns[idx])
                    ep_length = int(self.vector_episode_lengths[idx])
                    self.episode_returns.append(ep_return)
                    self.episode_lengths.append(ep_length)
                    self.total_episodes += 1
                    self.vector_episode_returns[idx] = 0.0
                    self.vector_episode_lengths[idx] = 0
        else:
            reward = float(rewards.item())
            done = bool(dones.item())
            self.current_episode_return += reward
            self.current_episode_length += 1
            if done:
                self.episode_returns.append(self.current_episode_return)
                self.episode_lengths.append(self.current_episode_length)
                self.total_episodes += 1
                self.current_episode_return = 0.0
                self.current_episode_length = 0

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

    def _prepare_batch(self, batch: Batch) -> Dict[str, torch.Tensor]:
        device = self.device or "cpu"
        tensor_batch: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                tensor_batch[key] = value.to(device)
            else:
                tensor_batch[key] = torch.as_tensor(value, device=device)
        return tensor_batch

    def _world_model_parameters(self):
        modules = [
            self.encoder,
            self.rssm,
            self.reward_predictor,
            self.observation_decoder,
        ]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                if param.requires_grad:
                    yield param

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

            obs_tensor = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32)
            reset_mask = None
            if self.current_dones is not None:
                reset_mask = torch.as_tensor(self.current_dones, device=self.device, dtype=torch.bool)

            latent_step: LatentStep = self.rssm.observe(
                obs_tensor,
                prev_action=self._prev_actions_model,
                reset_mask=reset_mask,
                detach_posteriors=True,
            )
            latent_tensor = self._latent_to_tensor(latent_step.posterior)

            actor_dist: Distribution = self.actor_controller.forward(latent_tensor)
            action_tensor = self._sample_action(actor_dist, deterministic=deterministic)
            action_env = self._action_to_env(action_tensor)

            value_pred = self.critic_controller.forward(latent_tensor).detach() if self.critic_controller else torch.zeros(
                latent_tensor.shape[0], device=self.device
            )
            log_prob = actor_dist.log_prob(action_tensor)

            next_obs, reward, done, infos = self.environment.step(action_env)

            self._update_episode_stats(reward, done, infos)

            rewards_for_metrics.append(np.asarray(reward, dtype=np.float32))
            rewards_list.append(np.asarray(reward, dtype=np.float32))
            dones_list.append(np.asarray(done, dtype=bool))
            values_list.append(value_pred.cpu().numpy())
            log_probs_list.append(log_prob.detach().cpu().numpy())
            actions_list.append(np.asarray(action_env))
            next_obs_list.append(np.asarray(next_obs, dtype=np.float32))

            self.current_obs = np.asarray(next_obs, dtype=np.float32)
            if self.current_obs.ndim == 1:
                self.current_obs = np.expand_dims(self.current_obs, axis=0)
            self.current_dones = np.asarray(done, dtype=bool)

            self._prev_actions_model = self._action_to_rssm(action_tensor.detach()).to(self.device)

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
            extras={"replay": {"trajectory": trajectory}},
        )

    def update_world_model(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        if self.world_model_optimizer is None or self.encoder is None or self.rssm is None:
            raise RuntimeError("World-model components not ready before update.")

        tensor_batch = self._prepare_batch(batch)
        observations = tensor_batch["observations"].to(torch.float32)  # (B, T, ...)
        actions = tensor_batch.get("actions")
        dones = tensor_batch.get("dones")

        if observations.dim() < 3:
            raise ValueError("Expected batched observation sequences with shape (B, T, ...).")

        batch_size, horizon = observations.shape[0], observations.shape[1]
        obs_flat = observations.reshape(batch_size * horizon, *observations.shape[2:])
        features_flat = self.encoder(obs_flat)
        if features_flat.dim() == 1:
            features_flat = features_flat.unsqueeze(-1)
        feature_dim = features_flat.shape[-1]
        features = features_flat.reshape(batch_size, horizon, feature_dim)

        actions_seq = None
        if actions is not None:
            actions_seq = self._prepare_action_sequence(actions)
        if dones is not None:
            dones = dones.to(torch.bool)

        sequence: LatentSequence = self.rssm.observe_sequence(features, actions=actions_seq, dones=dones)
        posterior = sequence.posterior
        prior = sequence.prior

        latent_feat = torch.cat([posterior.deterministic, posterior.stochastic], dim=-1)
        latent_flat = latent_feat.reshape(batch_size * horizon, -1)

        losses: Dict[str, torch.Tensor] = {}

        recon_loss = torch.tensor(0.0, device=self.device)
        if self.observation_decoder is not None:
            target = obs_flat.reshape(batch_size * horizon, -1)
            recon = self.observation_decoder(latent_flat)
            recon_loss = F.mse_loss(recon, target)
            losses["reconstruction"] = recon_loss

        reward_loss = torch.tensor(0.0, device=self.device)
        if self.reward_predictor is not None and "rewards" in tensor_batch:
            rewards = tensor_batch["rewards"].to(torch.float32).reshape(batch_size * horizon, -1)
            pred_rewards = self.reward_predictor(latent_flat)
            reward_loss = F.mse_loss(pred_rewards, rewards)
            losses["reward"] = reward_loss

        prior_dist = prior.distribution() if prior is not None else None
        posterior_dist = posterior.distribution()
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
            clip_grad_norm_(self._world_model_parameters(), self.max_grad_norm)
        self.world_model_optimizer.step()
        self.world_model_updates += 1

        metrics = {
            "world_model/total_loss": float(total_loss.item()),
            "world_model/kl_loss": float(kl_loss.item()),
        }
        if "reconstruction" in losses:
            metrics["world_model/recon_loss"] = float(recon_loss.item())
        if "reward" in losses:
            metrics["world_model/reward_loss"] = float(reward_loss.item())
        return metrics

    def update_controller(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        if self.actor_controller is None or self.critic_controller is None:
            return {}
        if self.world_model_updates < self.actor_warmup_updates:
            return {"controller/skipped": 1.0}

        tensor_batch = self._prepare_batch(batch)
        observations = tensor_batch["observations"].to(torch.float32)
        actions = tensor_batch.get("actions")
        dones = tensor_batch.get("dones")

        batch_size, horizon = observations.shape[0], observations.shape[1]
        obs_flat = observations.reshape(batch_size * horizon, *observations.shape[2:])
        with torch.no_grad():
            features_flat = self.encoder(obs_flat)
            feature_dim = features_flat.shape[-1]
            features = features_flat.reshape(batch_size, horizon, feature_dim)
            actions_seq = self._prepare_action_sequence(actions) if actions is not None else None
            dones_seq = dones.to(torch.bool) if dones is not None else None
            seq = self.rssm.observe_sequence(features, actions=actions_seq, dones=dones_seq)
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

        actor_optimizer = getattr(self.actor_controller, "optimizer", None)
        critic_optimizer = getattr(self.critic_controller, "optimizer", None)

        actor_loss = -(advantages.detach() * log_probs).mean() - self.entropy_coef * entropies.mean()
        if actor_optimizer is not None:
            actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_grad_clip = getattr(self.actor_controller, "grad_clip", None)
        if actor_grad_clip is not None:
            clip_grad_norm_(self.actor_controller.parameters(), actor_grad_clip)
        if actor_optimizer is not None:
            actor_optimizer.step()

        critic_loss = F.mse_loss(values, returns.detach())
        if critic_optimizer is not None:
            critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_clip = getattr(self.critic_controller, "grad_clip", None)
        if critic_grad_clip is not None:
            clip_grad_norm_(self.critic_controller.parameters(), critic_grad_clip)
        if critic_optimizer is not None:
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
        elif isinstance(latent, RSSMState):
            latent_state = latent.to(self.device)
        else:
            raise TypeError(f"Unsupported latent type for imagination: {type(latent)}")

        batch_size = latent_state.deterministic.shape[0]

        states_feat = []
        rewards = []
        values = []
        log_probs = []
        entropies = []
        actions = []

        state = latent_state
        for _ in range(horizon):
            latent_tensor = self._latent_to_tensor(state)
            actor_dist: Distribution = self.actor_controller.forward(latent_tensor)
            action = self._sample_action(actor_dist, deterministic=deterministic)
            rssm_action = self._action_to_rssm(action)

            critic_value = (
                self.critic_controller.forward(latent_tensor)
                if self.critic_controller is not None
                else torch.zeros(batch_size, device=self.device)
            )
            values.append(critic_value)
            log_probs.append(actor_dist.log_prob(action))
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

            state = self.rssm.imagine_step(state, rssm_action, deterministic=deterministic)

        bootstrap_value = (
            self.critic_controller.forward(self._latent_to_tensor(state))
            if self.critic_controller is not None
            else torch.zeros(batch_size, device=self.device)
        )

        states_tensor = torch.stack(states_feat, dim=1)
        rewards_tensor = torch.stack(rewards, dim=1)
        values_tensor = torch.stack([v.view(batch_size) for v in values], dim=1)
        log_probs_tensor = torch.stack([lp.view(batch_size) for lp in log_probs], dim=1)
        entropies_tensor = torch.stack([e.view(batch_size) for e in entropies], dim=1)

        return {
            "states": states_tensor,
            "actions": torch.stack([self._action_to_rssm(a) for a in actions], dim=1),
            "rewards": rewards_tensor,
            "values": values_tensor,
            "log_probs": log_probs_tensor,
            "entropies": entropies_tensor,
            "bootstrap": bootstrap_value.view(batch_size),
        }

    def log_metrics(self, step: int, writer: Any) -> None:
        if hasattr(self.buffer, "log_diagnostics"):
            self.buffer.log_diagnostics(writer, step)

    def state_dict(self) -> Dict[str, Any]:
        payload = {
            "world_model_optimizer": self._maybe_state_dict(self.world_model_optimizer),
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

   
