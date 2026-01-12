"""TD-MPC workflow implementing model-predictive control with TD value learning."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .utils.base import Batch, CollectResult, PhaseConfig, WorldModelWorkflow
from .utils.context import WorkflowContext
from .utils.controllers import ControllerManager


class TDMPCWorkflow(WorldModelWorkflow):
    """TD-MPC style workflow with deterministic latent dynamics and MPC planning."""

    def __init__(self, *, plan_horizon: int = 12, gamma: float = 0.99, config: Optional[Any] = None) -> None:
        super().__init__()
        self.plan_horizon = plan_horizon
        self.gamma = gamma
        self.device: torch.device | str = "cpu"

        self._override_config = config
        self._bound_config: Optional[Any] = None

        self.environment: Any = None
        self.eval_environment: Any = None
        self.encoder: Optional[torch.nn.Module] = None
        self.dynamics_model: Optional[torch.nn.Module] = None
        self.reward_predictor: Optional[torch.nn.Module] = None
        self.controller_manager: Optional[ControllerManager] = None
        self.planner: Optional[Any] = None
        self.value_function: Optional[torch.nn.Module] = None

        self.world_model_optimizer: Optional[torch.optim.Optimizer] = None
        self.value_optimizer: Optional[torch.optim.Optimizer] = None

        self.current_obs: Optional[np.ndarray] = None
        self.current_dones: Optional[np.ndarray] = None
        self.is_vectorized = False
        self.num_envs = 1
        self.action_dim = 0
        self.collect_length = 1

        self._action_low: Optional[Tensor] = None
        self._action_high: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # WorldModelWorkflow API
    # ------------------------------------------------------------------
    @property
    def config(self) -> Any:
        if self._override_config is not None:
            return self._override_config
        if self._bound_config is None:
            raise RuntimeError("TDMPCWorkflow accessed config before initialization.")
        return self._bound_config

    def initialize(self, context: WorkflowContext) -> None:
        self._bind_context(context)
        self._reset_episode_tracking(self.num_envs, clear_history=True)
        self._reset_rollout_state(
            initial_obs=context.initial_observation,
            initial_dones=context.initial_dones,
        )

    def collect_step(
        self,
        step: int,
        *,
        phase: PhaseConfig,
    ) -> Optional[CollectResult]:
        if self.environment is None or self.encoder is None or self.planner is None:
            raise RuntimeError("TDMPCWorkflow is not fully initialised before collect_step.")

        collect_len = int(phase.get("collect_length", self.collect_length) or self.collect_length)
        deterministic = bool(phase.get("deterministic_policy", False))

        obs_list: list[np.ndarray] = []
        next_obs_list: list[np.ndarray] = []
        actions_list: list[np.ndarray] = []
        rewards_list: list[np.ndarray] = []
        dones_list: list[np.ndarray] = []

        for _ in range(collect_len):
            obs_np = np.asarray(self.current_obs, dtype=np.float32)
            obs_list.append(obs_np.copy())

            obs_tensor = torch.as_tensor(obs_np, device=self.device, dtype=torch.float32)
            latent = self.encoder(obs_tensor)

            planner_kwargs = {"workflow": self, "deterministic": deterministic}
            action_tensor = self.planner.act(latent, **planner_kwargs)
            action_np = action_tensor.detach().cpu().numpy()

            next_obs, reward, done, info = self.environment.step(action_np)

            self._update_episode_stats(reward, done, info)

            actions_list.append(action_np.copy())
            rewards_list.append(np.asarray(reward, dtype=np.float32))
            dones_list.append(np.asarray(done, dtype=bool))
            next_obs_list.append(np.asarray(next_obs, dtype=np.float32))

            self.current_obs = np.asarray(next_obs, dtype=np.float32)
            self.current_dones = np.asarray(done, dtype=bool)

        trajectory = {
            "observations": np.stack(obs_list, axis=0),
            "next_observations": np.stack(next_obs_list, axis=0),
            "actions": np.stack(actions_list, axis=0),
            "rewards": np.stack(rewards_list, axis=0),
            "dones": np.stack(dones_list, axis=0),
        }

        return CollectResult(
            episodes=0,
            steps=int(collect_len * self.num_envs),
            extras={"replay": {"trajectory": trajectory}},
        )

    def update_world_model(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        if self.encoder is None or self.dynamics_model is None or self.reward_predictor is None:
            raise RuntimeError("World model components are missing for TD-MPC update.")
        if self.world_model_optimizer is None:
            raise RuntimeError("World-model optimizer not available for TD-MPC update.")

        obs = batch["observations"].to(self.device, dtype=torch.float32)
        next_obs = batch["next_observations"].to(self.device, dtype=torch.float32)
        actions = batch["actions"].to(self.device, dtype=torch.float32)
        rewards = batch["rewards"].to(self.device, dtype=torch.float32)

        batch_shape = obs.shape
        flat_size = math.prod(batch_shape[:-1])

        obs_flat = obs.reshape(flat_size, -1)
        next_obs_flat = next_obs.reshape(flat_size, -1)
        actions_flat = actions.reshape(flat_size, -1)
        rewards_flat = rewards.reshape(flat_size, -1)

        latent = self.encoder(obs_flat)
        with torch.no_grad():
            next_latent_target = self.encoder(next_obs_flat)

        next_latent_pred = self.dynamics_model(latent, actions_flat)
        dynamics_loss = F.mse_loss(next_latent_pred, next_latent_target)

        reward_pred = self.reward_predictor(latent)
        reward_loss = F.mse_loss(reward_pred, rewards_flat)

        total_loss = dynamics_loss + reward_loss

        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        self.world_model_optimizer.step()

        return {
            "tdmpc/dynamics_loss": float(dynamics_loss.item()),
            "tdmpc/reward_loss": float(reward_loss.item()),
        }

    def update_controller(
        self,
        batch: Batch,
        *,
        phase: PhaseConfig,
    ) -> Dict[str, float]:
        if self.encoder is None or self.value_function is None or self.value_optimizer is None:
            return {}

        obs = batch["observations"].to(self.device, dtype=torch.float32)
        next_obs = batch["next_observations"].to(self.device, dtype=torch.float32)
        rewards = batch["rewards"].to(self.device, dtype=torch.float32)
        dones = batch.get("dones")

        batch_shape = obs.shape
        flat_size = math.prod(batch_shape[:-1])

        obs_flat = obs.reshape(flat_size, -1)
        next_obs_flat = next_obs.reshape(flat_size, -1)
        rewards_flat = rewards.reshape(flat_size, -1)
        dones_flat = (
            dones.to(self.device, dtype=torch.float32).reshape(flat_size, -1)
            if dones is not None
            else torch.zeros_like(rewards_flat)
        )

        latent = self.encoder(obs_flat)
        next_latent = self.encoder(next_obs_flat)

        with torch.no_grad():
            next_value = self.value_function(next_latent)
            td_target = rewards_flat + self.gamma * (1.0 - dones_flat) * next_value

        value_pred = self.value_function(latent)
        value_loss = F.mse_loss(value_pred, td_target)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return {"tdmpc/value_loss": float(value_loss.item())}

    def imagine(
        self,
        *,
        latent: Tensor,
        horizon: Optional[int] = None,
        action_sequence: Optional[Tensor] = None,
        **_: Any,
    ) -> Dict[str, Tensor]:
        if self.dynamics_model is None or self.reward_predictor is None or self.value_function is None:
            raise RuntimeError("TDMPCWorkflow imagination requires dynamics, reward predictor, and value function.")

        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        latent = latent.to(self.device)

        horizon = int(horizon or self.plan_horizon)
        if action_sequence is None:
            raise ValueError("TD-MPC imagination requires an explicit action_sequence.")

        action_sequence = action_sequence.to(self.device)
        if action_sequence.dim() == 2:
            action_sequence = action_sequence.unsqueeze(0)

        batch_size = latent.shape[0]

        with torch.no_grad():
            states = [latent]
            rewards: list[Tensor] = []
            state = latent
            for t in range(horizon):
                action_t = action_sequence[:, t]
                reward = self.reward_predictor(state).reshape(batch_size, -1)
                rewards.append(reward)
                state = self.dynamics_model(state, action_t)
                states.append(state)

            bootstrap = self.value_function(state).reshape(batch_size, -1)
            rewards_tensor = torch.stack(rewards, dim=1)
            stacked_states = torch.stack(states, dim=1)

        return {
            "states": stacked_states,
            "rewards": rewards_tensor,
            "bootstrap": bootstrap,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _bind_context(self, context: WorkflowContext) -> None:
        self._bound_config = context.config
        self.device = context.device
        self.environment = context.train_environment
        self.eval_environment = context.eval_environment

        components = context.components
        self.encoder = getattr(components, "encoder", None)
        self.dynamics_model = getattr(components, "dynamics_model", None)
        self.reward_predictor = getattr(components, "reward_predictor", None)

        if context.controller_manager is None:
            raise RuntimeError("TDMPCWorkflow requires a controller manager.")

        self.controller_manager = context.controller_manager
        self.planner = self.controller_manager.get("planner")
        self.value_function = self.controller_manager.get("critic")

        optimizers = context.optimizers or {}
        self.world_model_optimizer = optimizers.get("world_model")
        self.value_optimizer = optimizers.get("critic")

        algo_cfg = getattr(self.config, "algorithm", None)
        if algo_cfg is not None:
            self.plan_horizon = int(getattr(algo_cfg, "plan_horizon", self.plan_horizon))
            self.gamma = float(getattr(algo_cfg, "gamma", self.gamma))

        self.collect_length = int(getattr(self.config.algorithm, "collect_length", 1))

        if hasattr(self.environment, "is_vectorized"):
            self.is_vectorized = bool(self.environment.is_vectorized)
        self.num_envs = int(getattr(self.environment, "num_envs", 1) or 1)

        action_space = getattr(self.environment, "action_space", None)
        if action_space is not None and hasattr(action_space, "shape"):
            self.action_dim = int(np.prod(action_space.shape))
            try:
                low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
                high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
            except Exception:
                low = torch.full((self.action_dim,), -1.0, device=self.device)
                high = torch.full((self.action_dim,), 1.0, device=self.device)
        else:
            self.action_dim = getattr(self.config._dims, "action", 0)
            low = torch.full((self.action_dim,), -1.0, device=self.device)
            high = torch.full((self.action_dim,), 1.0, device=self.device)

        self._action_low = low
        self._action_high = high

    def _reset_rollout_state(
        self,
        *,
        initial_obs: Optional[Any],
        initial_dones: Optional[Any],
    ) -> None:
        if initial_obs is None:
            raise RuntimeError("Initial observation snapshot missing for TDMPC workflow.")
        obs = np.asarray(initial_obs, dtype=np.float32)
        if obs.shape[0] != self.num_envs:
            raise ValueError(
                f"Environment reset returned batch {obs.shape[0]}, expected {self.num_envs}."
            )
        self.current_obs = obs

        if initial_dones is None:
            initial_dones = np.zeros(self.num_envs, dtype=bool)
        dones = np.asarray(initial_dones, dtype=bool)
        if dones.shape[0] != self.num_envs:
            raise ValueError(
                f"Initial done mask batch {dones.shape[0]}, expected {self.num_envs}."
            )
        self.current_dones = dones

    def get_action_bounds(self) -> tuple[Tensor, Tensor]:
        if self._action_low is None or self._action_high is None:
            raise RuntimeError("Action bounds are not initialised.")
        return self._action_low, self._action_high
