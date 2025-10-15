"""Dreamer workflow implementation for orchestrated world-model training."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Distribution

from ...components.world_models.latents import LatentBatch
from ...utils.config import Config
from .base import Batch, CollectResult, PhaseConfig, WorldModelWorkflow
from .context import WorkflowContext, WorldModelComponents
from .controllers import ControllerManager

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

        self.world_model_optimizer: Optional[torch.optim.Optimizer] = None
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.value_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None

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

    def collect_step(
        self,
        step: int,
        *,
        phase: PhaseConfig,
    ) -> Optional[CollectResult]:
        phase = phase or {}
        phase_type = str(phase.get("type", ""))
        deterministic = bool(phase.get("deterministic", phase_type == "eval_only"))
        is_eval_phase = phase_type == "eval_only"

        rollout_steps = max(
            self.buffer.batch_size,
            int(getattr(self.config.buffer, "sequence_length", 1) or 1),
        )

        collected_steps = 0
        episodes = 0
        trajectories = []
        start_total_episodes = self.total_episodes

        if self.current_obs is None:
            self.current_obs = self.environment.reset()
            if self.is_vectorized:
                self.current_dones = np.zeros(self.num_envs, dtype=bool)

        while collected_steps < rollout_steps and step + collected_steps < self.config.training.total_timesteps:
            if self.is_vectorized:
                trajectory, completed = self._collect_vector_step(deterministic=deterministic)
                collected_steps += self.num_envs
            else:
                trajectory, completed = self._collect_single_step(deterministic=deterministic)
                collected_steps += 1

            trajectories.append({"trajectory": trajectory})
            episodes += completed

        metrics = {
            "collect/episodes": float(episodes),
            "collect/steps": float(collected_steps),
        }

        extras: Dict[str, Any] = {"replay": trajectories} if trajectories else {}

        if is_eval_phase:
            new_episodes = self.total_episodes - start_total_episodes
            if new_episodes > 0:
                recent_returns = list(self.episode_returns)[-new_episodes:]
                recent_lengths = list(self.episode_lengths)[-new_episodes:]
                extras["evaluation"] = {
                    "returns": [float(value) for value in recent_returns],
                    "lengths": [float(value) for value in recent_lengths],
                }
                metrics["evaluation/episodes"] = float(new_episodes)

        return CollectResult(
            episodes=episodes,
            steps=collected_steps,
            metrics=metrics,
            extras=extras,
        )

    def update_world_model(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        raise NotImplementedError(
            "DreamerWorkflow.update_world_model no longer provides a default loss. "
            "Implement your world-model losses here using `self.components`."
        )

    def update_controller(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        if self.controller_manager:
            return self.controller_manager.learn_all(batch, phase=phase)
        return {}

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
        if self.components is None:
            raise RuntimeError("DreamerWorkflow must be initialized before calling imagine().")

        if observations is not None and latent is not None:
            raise ValueError("Specify either observations or latent, not both.")
        if observations is None and latent is None:
            raise ValueError("DreamerWorkflow.imagine requires observations or latent input.")

        if latent is not None:
            latent_batch = self._coerce_latent_batch(latent)
        else:
            obs_tensor = self._observation_to_tensor(observations)
            # TODO: Implement encode_observation() helper that uses self.components directly
            # Should: features = self.components.encoder(obs_tensor)
            #         latent = self.components.representation_learner.encode(features, sample=not deterministic)
            raise NotImplementedError("DreamerWorkflow.imagine needs encoding implementation")

        batch_size = int(latent_batch.latent.shape[0])
        action_sequence_tensor = (
            self._coerce_action_sequence(action_sequence, batch_size)
            if action_sequence is not None
            else None
        )

        controller = self._resolve_controller(controller_role)
        if controller is not None:
            imagine_hook = getattr(controller, "imagine", None)
            if callable(imagine_hook):
                return imagine_hook(
                    latent_batch=latent_batch,
                    dynamics_model=self.components.dynamics_model,
                    value_function=self.components.value_function,
                    policy_head=self.components.policy_head,
                    horizon=horizon,
                    deterministic=deterministic,
                    action_sequence=action_sequence_tensor,
                )

        rollout = self._default_imagination_rollout(
            latent_batch=latent_batch,
            horizon=horizon,
            deterministic=deterministic,
            action_sequence=action_sequence_tensor,
        )
        rollout["latent_initial"] = latent_batch.latent
        return rollout

    def log_metrics(self, step: int, writer: Any) -> None:
        if hasattr(self.buffer, "log_diagnostics"):
            self.buffer.log_diagnostics(writer, step)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "world_model_optimizer": self._maybe_state_dict(self.world_model_optimizer),
            "actor_optimizer": self._maybe_state_dict(self.actor_optimizer),
            "value_optimizer": self._maybe_state_dict(self.value_optimizer),
            "world_model_updates": self.world_model_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state:
            return
        if self.world_model_optimizer and state.get("world_model_optimizer"):
            self.world_model_optimizer.load_state_dict(state["world_model_optimizer"])
        if self.actor_optimizer and state.get("actor_optimizer"):
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if self.value_optimizer and state.get("value_optimizer"):
            self.value_optimizer.load_state_dict(state["value_optimizer"])
        self.world_model_updates = int(state.get("world_model_updates", 0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def on_context_update(self, context: WorkflowContext) -> None:
        self._bind_context(context)

    def _bind_context(self, context: WorkflowContext) -> None:
        self.context = context
        self.device = context.device
        components = context.components
        if components is None:
            raise RuntimeError("Workflow context missing world model components.")
        if hasattr(components, "to"):
            components.to(self.device)
        self.components = components
        self.buffer = context.buffer
        self.environment = context.train_environment
        self.eval_environment = context.eval_environment
        self.controllers = context.controllers or {}
        self.controller_manager = context.controller_manager or (
            ControllerManager(self.controllers) if self.controllers else None
        )
        self.is_vectorized = getattr(self.environment, "is_vectorized", False)
        self.num_envs = int(getattr(self.environment, "num_envs", 1))

    def _prepare_optimizers(self) -> None:
        alg_cfg = self.config.algorithm
        if self.components is None:
            raise RuntimeError("DreamerWorkflow components not bound.")
        world_params = []
        world_params.extend(self.components.encoder.parameters())
        world_params.extend(self.components.representation_learner.parameters())
        world_params.extend(self.components.dynamics_model.parameters())
        if getattr(self.components, "observation_decoder", None) is not None:
            world_params.extend(self.components.observation_decoder.parameters())
        if getattr(self.components, "reward_predictor", None) is not None:
            world_params.extend(self.components.reward_predictor.parameters())

        self.world_model_optimizer = (
            torch.optim.Adam(
                world_params,
                lr=alg_cfg.world_model_lr,
            )
            if world_params
            else None
        )

        self.actor_optimizer = torch.optim.Adam(
            self.components.policy_head.parameters(),
            lr=alg_cfg.actor_lr,
        )

        self.value_optimizer = torch.optim.Adam(
            self.components.value_function.parameters(),
            lr=alg_cfg.value_lr,
        )
        self.critic_optimizer = self.value_optimizer

    @staticmethod
    def _maybe_state_dict(optimizer: Optional[torch.optim.Optimizer]) -> Optional[Dict[str, Any]]:
        if optimizer is None:
            return None
        return optimizer.state_dict()

    def _clip_gradients(self, actor_active: bool, critic_active: bool) -> None:
        def _clip(optimizer: Optional[torch.optim.Optimizer]) -> None:
            if optimizer is None:
                return
            params = [
                p
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            ]
            if params:
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        if self.world_model_optimizer:
            _clip(self.world_model_optimizer)
        if actor_active:
            _clip(self.actor_optimizer)
        if critic_active:
            _clip(self.critic_optimizer)

    def _set_seeds(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        try:
            self.environment.seed(seed)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Environment seed failed: %s", exc)

    def _reset_rollout_state(self) -> None:
        self.current_obs = None
        self.current_dones = None
        self.total_episodes = 0
        self.episode_returns.clear()
        self.episode_lengths.clear()
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        if self.is_vectorized:
            self.vector_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self.vector_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        else:
            self.vector_episode_returns = None
            self.vector_episode_lengths = None

    def _select_action(self, action_dist: Distribution, *, deterministic: bool) -> torch.Tensor:
        if deterministic:
            if hasattr(action_dist, "mode"):
                action = action_dist.mode
            elif hasattr(action_dist, "mean"):
                action = action_dist.mean
            else:
                action = action_dist.sample()
        else:
            if getattr(action_dist, "has_rsample", False):
                action = action_dist.rsample()
            else:
                action = action_dist.sample()
        return action

    def _collect_single_step(self, *, deterministic: bool) -> tuple[Dict[str, np.ndarray], int]:
        obs = self.current_obs
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)

        # TODO: Replace with direct component usage - encode observation then call policy/planner
        # latent_batch = self.encode_observation(obs_tensor, sample=not deterministic)
        # action_dist = self.components.policy_head(latent_batch.latent) or planner
        raise NotImplementedError("DreamerWorkflow._collect_single_step needs component refactor")
        action_tensor = self._select_action(action_dist, deterministic=deterministic)
        action = action_tensor.cpu().numpy().squeeze(0)
        env_action = action.item() if action.shape == () else action

        next_obs, reward, done, info = self.environment.step(env_action)
        log_prob = action_dist.log_prob(action_tensor).detach().cpu().numpy()

        trajectory = {
            "observations": np.expand_dims(np.asarray(obs), axis=0),
            "actions": np.expand_dims(np.asarray(action), axis=0),
            "log_probs": np.expand_dims(np.asarray(log_prob), axis=0),
            "next_observations": np.expand_dims(np.asarray(next_obs), axis=0),
            "rewards": np.asarray([reward], dtype=np.float32),
            "dones": np.asarray([done], dtype=bool),
        }

        self.current_obs = next_obs if not done else self.environment.reset()

        self.current_episode_return += float(reward)
        self.current_episode_length += 1
        if done:
            self.episode_returns.append(self.current_episode_return)
            self.episode_lengths.append(self.current_episode_length)
            self.total_episodes += 1
            self.current_episode_return = 0.0
            self.current_episode_length = 0

            reported = {}
            if isinstance(info, dict):
                reported = info.get("episode", info)
            if "r" in reported:
                self.episode_returns[-1] = float(reported["r"])
            if "l" in reported:
                self.episode_lengths[-1] = int(reported["l"])

        return trajectory, int(done)

    def _collect_vector_step(self, *, deterministic: bool) -> tuple[Dict[str, np.ndarray], int]:
        obs = self.current_obs
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

        # TODO: Replace with direct component usage - encode observation then call policy/planner
        # latent_batch = self.encode_observation(obs_tensor, sample=not deterministic)
        # action_dist = self.components.policy_head(latent_batch.latent) or planner
        raise NotImplementedError("DreamerWorkflow._collect_vector_step needs component refactor")
        action_tensor = self._select_action(action_dist, deterministic=deterministic)
        actions = action_tensor.cpu().numpy()

        env_actions = actions
        next_obs, rewards, dones, infos = self.environment.step(env_actions)

        if isinstance(rewards, list):
            rewards_np = np.asarray(rewards, dtype=np.float32)
        else:
            rewards_np = rewards
        dones_np = np.asarray(dones, dtype=bool)

        log_probs = action_dist.log_prob(action_tensor).detach().cpu().numpy()

        trajectory = {
            "observations": np.expand_dims(np.asarray(obs), axis=0),
            "actions": np.expand_dims(np.asarray(actions), axis=0),
            "log_probs": np.expand_dims(np.asarray(log_probs), axis=0),
            "next_observations": np.expand_dims(np.asarray(next_obs), axis=0),
            "rewards": np.expand_dims(np.asarray(rewards_np, dtype=np.float32), axis=0),
            "dones": np.expand_dims(dones_np, axis=0),
        }

        self.current_obs = next_obs
        self.current_dones = dones_np

        if self.vector_episode_returns is None or self.vector_episode_lengths is None:
            raise RuntimeError("Vector episode trackers not initialized for vectorized environment.")

        self.vector_episode_returns += rewards_np
        self.vector_episode_lengths += 1

        done_indices = np.where(dones_np)[0]
        appended_indices: Dict[int, int] = {}
        for idx in done_indices:
            episodic_return = self.vector_episode_returns[idx]
            episodic_length = self.vector_episode_lengths[idx]
            self.episode_returns.append(float(episodic_return))
            self.episode_lengths.append(int(episodic_length))
            self.total_episodes += 1
            appended_indices[idx] = len(self.episode_returns) - 1
            self.vector_episode_returns[idx] = 0.0
            self.vector_episode_lengths[idx] = 0

        if hasattr(infos, "__iter__"):
            for env_idx, (done_flag, info) in enumerate(zip(dones_np, infos)):
                if done_flag and isinstance(info, dict) and env_idx in appended_indices:
                    ret = info.get("episode_return", info.get("episode", {}).get("r"))
                    length = info.get("episode_length", info.get("episode", {}).get("l"))
                    pos = appended_indices[env_idx]
                    if ret is not None:
                        self.episode_returns[pos] = float(ret)
                    if length is not None:
                        self.episode_lengths[pos] = int(length)

        episodes_completed = int(dones_np.sum())
        return trajectory, episodes_completed

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _resolve_controller(self, role: Optional[str]):
        if role is None or not role or self.controller_manager is None:
            return None
        try:
            return self.controller_manager.get(role)
        except KeyError:
            logger.warning("Controller role '%s' not registered; falling back to policy head.", role)
            return None

    def _observation_to_tensor(self, observation: Any) -> Any:
        if isinstance(observation, Mapping):
            return {
                key: self._ensure_batch(torch.as_tensor(value, dtype=torch.float32, device=self.device))
                for key, value in observation.items()
            }
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        return self._ensure_batch(tensor)

    def _ensure_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor.reshape(1, 1)
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor.unsqueeze(0) if tensor.dim() > 1 and tensor.shape[0] != 1 else tensor

    def _coerce_latent_batch(self, latent: Any) -> LatentBatch:
        if isinstance(latent, LatentBatch):
            return latent
        latent_tensor = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        if latent_tensor.dim() == 1:
            latent_tensor = latent_tensor.unsqueeze(0)
        return LatentBatch(latent=latent_tensor, features=None)

    def _coerce_action_sequence(self, action_sequence: Any, batch_size: int) -> torch.Tensor:
        tensor = torch.as_tensor(action_sequence, device=self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 2 and tensor.shape[0] not in (batch_size, 1):
            tensor = tensor.unsqueeze(0)
        if tensor.dim() > 3 or tensor.dim() < 2:
            raise ValueError(
                "Action sequence must have shape (horizon,), (horizon, action_dim), "
                "(batch, horizon) or (batch, horizon, action_dim)."
            )
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, *tensor.shape[1:])
        return tensor

    def _default_imagination_rollout(
        self,
        *,
        latent_batch: LatentBatch,
        horizon: Optional[int],
        deterministic: bool,
        action_sequence: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.components is None:
            raise RuntimeError("DreamerWorkflow components not initialized.")

        # TODO: Get imagination_length from config or component config
        horizon = horizon or 15
        current_state = latent_batch.latent
        policy_head = self.components.policy_head
        dynamics_model = self.components.dynamics_model
        value_function = self.components.value_function
        reward_predictor = self.components.reward_predictor

        states = []
        next_states = []
        actions = []
        rewards = []
        values = []
        next_values = []
        log_probs = []
        entropies = []

        discrete_actions = getattr(policy_head, "discrete_actions", False)
        action_dim = getattr(policy_head, "action_dim", None)

        for step in range(horizon):
            action_dist = policy_head(current_state)

            if action_sequence is not None:
                action = action_sequence[:, step]
                if discrete_actions:
                    action = action.long()
            else:
                if deterministic:
                    action = getattr(action_dist, "mean", None)
                    if action is None:
                        action = getattr(action_dist, "mode", None)
                    if action is None:
                        action = action_dist.sample()
                else:
                    if getattr(action_dist, "has_rsample", False):
                        action = action_dist.rsample()
                    else:
                        action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(dim=-1)

            entropy = action_dist.entropy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            state_value = value_function(current_state)
            if state_value.dim() > 1:
                state_value = state_value.squeeze(-1)

            action_for_model = action
            if discrete_actions:
                if action_dim is None:
                    raise ValueError("Discrete actions require policy head to define action_dim")
                action_for_model = torch.nn.functional.one_hot(action.long(), num_classes=action_dim).float()

            next_state_dist = dynamics_model(current_state, action_for_model)
            if deterministic:
                next_state = getattr(next_state_dist, "mean", None)
                if next_state is None:
                    next_state = next_state_dist.sample()
            else:
                if getattr(next_state_dist, "has_rsample", False):
                    next_state = next_state_dist.rsample()
                else:
                    next_state = next_state_dist.sample()

            next_value = value_function(next_state)
            if next_value.dim() > 1:
                next_value = next_value.squeeze(-1)

            if reward_predictor is not None:
                reward_pred = reward_predictor(next_state)
                if reward_pred.dim() > 1:
                    reward_pred = reward_pred.squeeze(-1)
            else:
                reward_pred = torch.zeros(current_state.shape[0], device=current_state.device)

            states.append(current_state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward_pred)
            values.append(state_value)
            next_values.append(next_value)
            log_probs.append(log_prob)
            entropies.append(entropy)

            current_state = next_state

        stacked_states = torch.stack(states, dim=1)
        stacked_next_states = torch.stack(next_states, dim=1)
        stacked_actions = torch.stack(actions, dim=1)
        stacked_rewards = torch.stack(rewards, dim=1)
        stacked_values = torch.stack(values, dim=1)
        stacked_next_values = torch.stack(next_values, dim=1)
        stacked_log_probs = torch.stack(log_probs, dim=1)
        stacked_entropies = torch.stack(entropies, dim=1)

        bootstrap_value = stacked_next_values[:, -1]

        return {
            "states": stacked_states,
            "next_states": stacked_next_states,
            "actions": stacked_actions,
            "rewards": stacked_rewards,
            "values": stacked_values,
            "next_values": stacked_next_values,
            "log_probs": stacked_log_probs,
            "entropies": stacked_entropies,
            "bootstrap": bootstrap_value,
        }
