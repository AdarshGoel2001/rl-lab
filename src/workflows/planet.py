"""Tiny PlaNet-style workflow for testing RSSM training and latent planning."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.nn.utils import clip_grad_norm_

from ..components.representation_learners import LatentState, RSSMState
from .utils.base import Batch, CollectResult, PhaseConfig, WorldModelWorkflow
from .utils.context import WorkflowContext

logger = logging.getLogger(__name__)


class PlaNetWorkflow(WorldModelWorkflow):
    """Small PlaNet recipe: RSSM world model plus CEM-compatible imagination."""

    def __init__(self) -> None:
        super().__init__()
        self.device: str = "cpu"
        self.config = None
        self.environment = None
        self.eval_environment = None
        self.buffers: Dict[str, Any] = {}
        self.controllers: Dict[str, Any] = {}
        self.rssm: Optional[torch.nn.Module] = None
        self.reward_predictor: Optional[torch.nn.Module] = None
        self.continue_predictor: Optional[torch.nn.Module] = None
        self.observation_predictor: Optional[torch.nn.Module] = None
        self.world_model_optimizer: Optional[torch.optim.Optimizer] = None
        self.current_obs: Optional[np.ndarray] = None
        self.current_dones: Optional[np.ndarray] = None
        self.collect_length = 1
        self.action_dim = 0
        self.gamma = 0.99
        self.free_nats = 0.0
        self.kl_scale = 1.0
        self.reward_loss_scale = 1.0
        self.continue_loss_scale = 1.0
        self.observation_loss_scale = 1.0
        self.exploration_noise = 0.0
        self.max_grad_norm: Optional[float] = None
        self.action_low: Optional[torch.Tensor] = None
        self.action_high: Optional[torch.Tensor] = None
        self.world_model_updates = 0
        self.rollout_state: Optional[LatentState] = None
        self.prev_action: Optional[torch.Tensor] = None

    def initialize(self, context: WorkflowContext) -> None:
        self.config = context.config
        self.device = context.device
        self.environment = context.train_environment
        self.eval_environment = context.eval_environment
        self.buffers = dict(context.buffers)
        self.controllers = dict(context.controllers or {})
        self.rssm = context.components.representation_learner
        self.reward_predictor = context.components.reward_predictor
        self.continue_predictor = getattr(context.components, "continue_predictor", None)
        self.observation_predictor = getattr(context.components, "observation_predictor", None)
        self.world_model_optimizer = (context.optimizers or {}).get("world_model")
        if self.world_model_optimizer is None:
            raise RuntimeError("PlaNetWorkflow requires a world_model optimizer.")

        dims = getattr(self.config, "_dims", None)
        self.action_dim = int(getattr(dims, "action"))
        algorithm = getattr(self.config, "algorithm", None)
        self.collect_length = int(getattr(algorithm, "collect_length", 1))
        self.gamma = float(getattr(algorithm, "gamma", 0.99))
        self.free_nats = float(getattr(algorithm, "free_nats", 0.0))
        self.kl_scale = float(getattr(algorithm, "kl_scale", 1.0))
        self.reward_loss_scale = float(getattr(algorithm, "reward_loss_scale", 1.0))
        self.continue_loss_scale = float(getattr(algorithm, "continue_loss_scale", 1.0))
        self.observation_loss_scale = float(getattr(algorithm, "observation_loss_scale", 1.0))
        self.exploration_noise = float(getattr(algorithm, "exploration_noise", 0.0))
        self.max_grad_norm = getattr(algorithm, "max_grad_norm", None)

        self.num_envs = int(getattr(self.environment, "num_envs", 1) or 1)
        self._reset_episode_tracking(self.num_envs, clear_history=True)
        self.current_obs = np.asarray(context.initial_observation, dtype=np.float32)
        self.current_dones = np.asarray(context.initial_dones, dtype=bool)
        self.rollout_state = None
        self.prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        actor = self.controllers.get("actor")
        low = getattr(actor, "action_low", torch.zeros(self.action_dim))
        high = getattr(actor, "action_high", torch.ones(self.action_dim))
        self.action_low = torch.as_tensor(low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(high, dtype=torch.float32, device=self.device)

    def collect_step(self, step: int, *, phase: PhaseConfig) -> Optional[CollectResult]:
        if self.environment is None or self.current_obs is None:
            raise RuntimeError("PlaNetWorkflow not initialized before collect_step.")
        controller_role = str(phase.get("controller", "actor"))
        controller = self.controllers.get(controller_role)
        if controller is None:
            raise RuntimeError(f"PlaNetWorkflow requires controller '{controller_role}' for collection.")

        collect_len = int(phase.get("collect_length", self.collect_length) or self.collect_length)
        start = time.time()
        observations, actions, rewards, dones = [], [], [], []

        for _ in range(collect_len):
            obs = np.asarray(self.current_obs, dtype=np.float32)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                if controller_role == "planner":
                    if self.rssm is None:
                        raise RuntimeError("Planner-guided collection requires an RSSM.")
                    prev_action = self.prev_action
                    if prev_action is None or prev_action.shape[0] != obs_tensor.shape[0]:
                        prev_action = torch.zeros(obs_tensor.shape[0], self.action_dim, device=self.device)
                    reset_mask = torch.as_tensor(self.current_dones, dtype=torch.bool, device=self.device)
                    latent_step = self.rssm.observe(
                        obs_tensor,
                        prev_action=prev_action,
                        reset_mask=reset_mask,
                        detach_posteriors=True,
                    )
                    self.rollout_state = latent_step.posterior
                    action_tensor = controller.act(self.rollout_state.to_tensor(), workflow=self)
                    exploration_noise = float(phase.get("exploration_noise", self.exploration_noise) or 0.0)
                    if exploration_noise > 0.0:
                        action_tensor = action_tensor + torch.randn_like(action_tensor) * exploration_noise
                        low, high = self.get_action_bounds()
                        action_tensor = torch.clamp(action_tensor, min=low.to(self.device), max=high.to(self.device))
                else:
                    action_tensor = controller.act(obs_tensor)
            next_obs, reward, done, info = self.environment.step(action_tensor)

            self._update_episode_stats(reward, done, info)
            done_arr = np.asarray(done, dtype=bool)
            observations.append(obs.copy())
            actions.append(action_tensor.detach().cpu().numpy().copy())
            rewards.append(np.asarray(reward, dtype=np.float32).copy())
            dones.append(done_arr.copy())
            next_obs_arr = np.asarray(next_obs, dtype=np.float32)
            self.current_obs = self._reset_single_env_if_done(next_obs_arr, done_arr)
            self.current_dones = done_arr
            self.prev_action = action_tensor.detach()

        trajectory = {
            "observations": np.stack(observations, axis=0),
            "actions": np.stack(actions, axis=0),
            "rewards": np.stack(rewards, axis=0),
            "dones": np.stack(dones, axis=0),
        }
        return CollectResult(
            steps=collect_len * self.num_envs,
            episodes=0,
            metrics={
                "collect/mean_step_reward": float(np.mean(trajectory["rewards"])),
                "collect/duration": float(time.time() - start),
            },
            trajectory=trajectory,
        )

    def update_world_model(self, batch: Batch, *, phase: PhaseConfig) -> Dict[str, float]:
        if self.rssm is None or self.reward_predictor is None:
            raise RuntimeError("PlaNet world-model components are not initialized.")
        if self.world_model_optimizer is None:
            raise RuntimeError("PlaNet world-model optimizer is not initialized.")

        tensor_batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.device)
            for key, value in batch.items()
        }
        observations = tensor_batch["observations"].to(torch.float32)
        actions = tensor_batch["actions"].to(torch.float32)
        rewards = tensor_batch["rewards"].to(torch.float32)
        dones = tensor_batch["dones"].to(torch.bool)

        sequence = self.rssm.observe_sequence(observations, actions=actions, dones=dones)
        posterior = sequence.posterior
        prior = sequence.prior
        if prior is None:
            raise RuntimeError("RSSM did not return prior states.")

        latent = posterior.to_tensor()
        reward_pred = self.reward_predictor(latent)

        reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards)
        kl_values = kl_divergence(sequence.posterior_dist, sequence.prior_dist)
        if self.free_nats > 0.0:
            kl_values = torch.clamp(kl_values - self.free_nats, min=0.0)
        kl_loss = kl_values.mean()
        total_loss = self.reward_loss_scale * reward_loss + self.kl_scale * kl_loss

        continue_loss = None
        if self.continue_predictor is not None:
            continue_logits = self.continue_predictor(latent)
            continue_target = (~dones).to(torch.float32)
            continue_loss = F.binary_cross_entropy_with_logits(
                continue_logits.squeeze(-1),
                continue_target,
            )
            total_loss = total_loss + self.continue_loss_scale * continue_loss

        observation_loss = None
        if self.observation_predictor is not None:
            observation_pred = self.observation_predictor(latent)
            observation_loss = F.mse_loss(observation_pred, observations)
            total_loss = total_loss + self.observation_loss_scale * observation_loss

        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.world_model_optimizer.param_groups[0]["params"], float(self.max_grad_norm))
        self.world_model_optimizer.step()
        self.world_model_updates += 1

        metrics = {
            "world_model/total_loss": float(total_loss.item()),
            "world_model/reward_loss": float(reward_loss.item()),
            "world_model/kl_loss": float(kl_loss.item()),
            "world_model/reward_loss_scale": float(self.reward_loss_scale),
            "world_model/kl_scale": float(self.kl_scale),
        }
        if continue_loss is not None:
            metrics["world_model/continue_loss"] = float(continue_loss.item())
            metrics["world_model/continue_loss_scale"] = float(self.continue_loss_scale)
        if observation_loss is not None:
            metrics["world_model/observation_loss"] = float(observation_loss.item())
            metrics["world_model/observation_loss_scale"] = float(self.observation_loss_scale)
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
        del observations, controller_role
        if self.rssm is None or self.reward_predictor is None:
            raise RuntimeError("PlaNet imagination requested before components are initialized.")
        if isinstance(latent, LatentState):
            state = latent.to(self.device)
        elif isinstance(latent, torch.Tensor):
            state = self._rssm_state_from_tensor(latent.to(self.device))
        else:
            raise TypeError("PlaNetWorkflow.imagine expects an RSSMState or flattened latent tensor.")
        if action_sequence is None:
            raise ValueError("PlaNetWorkflow.imagine requires an action_sequence for planning.")

        actions = torch.as_tensor(action_sequence, dtype=torch.float32, device=self.device)
        horizon = int(horizon or actions.shape[1])
        states, rewards, continues = [], [], []

        for t in range(horizon):
            state = self.rssm.imagine_step(state, actions[:, t], deterministic=deterministic)
            latent_tensor = state.to_tensor()
            states.append(latent_tensor)
            rewards.append(self.reward_predictor(latent_tensor))
            if self.continue_predictor is not None:
                continues.append(torch.sigmoid(self.continue_predictor(latent_tensor)))

        bootstrap = self.reward_predictor(state.to_tensor())
        rollout = {
            "states": torch.stack(states, dim=1),
            "actions": actions[:, :horizon],
            "rewards": torch.stack(rewards, dim=1),
            "bootstrap": bootstrap,
        }
        if continues:
            rollout["continues"] = torch.stack(continues, dim=1)
        return rollout

    def get_action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.action_low is None or self.action_high is None:
            low = torch.zeros(self.action_dim, device=self.device)
            high = torch.ones(self.action_dim, device=self.device)
            return low, high
        return self.action_low, self.action_high

    def _reset_single_env_if_done(self, next_obs: np.ndarray, done: np.ndarray) -> np.ndarray:
        if self.environment is None:
            return next_obs
        if int(getattr(self.environment, "num_envs", self.num_envs)) != 1:
            return next_obs
        if bool(getattr(self.environment, "is_vectorized", False)):
            return next_obs
        done_flat = np.asarray(done, dtype=bool).reshape(-1)
        if done_flat.size == 0 or not bool(done_flat[0]):
            return next_obs
        reset_fn = getattr(self.environment, "reset", None)
        if not callable(reset_fn):
            return next_obs

        reset_obs = reset_fn()
        if isinstance(reset_obs, tuple):
            reset_obs = reset_obs[0]
        reset_obs_arr = np.asarray(reset_obs, dtype=np.float32)
        if reset_obs_arr.ndim + 1 == next_obs.ndim:
            reset_obs_arr = np.expand_dims(reset_obs_arr, axis=0)
        return reset_obs_arr

    def evaluate(
        self,
        num_eval_batches: int = 5,
        max_steps_per_episode: int = 500,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        if self.rssm is None:
            raise RuntimeError("PlaNetWorkflow.evaluate requires an initialized RSSM.")
        env = self.eval_environment or self.environment
        if env is None:
            raise RuntimeError("PlaNetWorkflow.evaluate requires an eval or train environment.")
        planner = self.controllers.get("planner")
        if planner is None:
            raise RuntimeError("PlaNetWorkflow.evaluate requires a 'planner' controller.")

        returns: list[float] = []
        lengths: list[int] = []
        eval_episode_batches = int(num_eval_batches)
        eval_num_envs = int(getattr(env, "num_envs", 1) or 1)
        for episode_batch in range(eval_episode_batches):
            reset = env.reset(seed=episode_batch)
            obs = reset[0] if isinstance(reset, tuple) else reset
            obs = np.asarray(obs, dtype=np.float32)
            if obs.ndim == 1:
                obs = obs[None, :]
            prev_action = torch.zeros(obs.shape[0], self.action_dim, device=self.device)
            done = np.zeros(obs.shape[0], dtype=bool)
            episode_return = np.zeros(obs.shape[0], dtype=np.float32)
            episode_length = np.zeros(obs.shape[0], dtype=np.int32)

            for _ in range(int(max_steps_per_episode)):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                reset_mask = torch.as_tensor(done, dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    latent_step = self.rssm.observe(
                        obs_tensor,
                        prev_action=prev_action,
                        reset_mask=reset_mask,
                        detach_posteriors=True,
                    )
                    planned_action = planner.act(
                        latent_step.posterior.to_tensor(),
                        workflow=self,
                        deterministic=deterministic,
                    )

                env_action = planned_action.detach().cpu().numpy()
                next_obs, reward, done, info = env.step(env_action)
                reward_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
                done_arr = np.asarray(done, dtype=bool).reshape(-1)
                episode_return += reward_arr
                episode_length += 1
                prev_action = planned_action.detach()
                obs = np.asarray(next_obs, dtype=np.float32)
                if obs.ndim == 1:
                    obs = obs[None, :]
                done = done_arr
                if np.all(done_arr):
                    break

            returns.extend(float(value) for value in episode_return)
            lengths.extend(int(value) for value in episode_length)

        eval_total_episodes = len(returns)
        return {
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "return_std": float(np.std(returns)) if returns else 0.0,
            "return_max": float(np.max(returns)) if returns else 0.0,
            "return_min": float(np.min(returns)) if returns else 0.0,
            "length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "episodes": float(eval_total_episodes),
            "eval_episode_batches": float(eval_episode_batches),
            "eval_num_envs": float(eval_num_envs),
            "eval_total_episodes": float(eval_total_episodes),
        }

    def _rssm_state_from_tensor(self, latent: torch.Tensor) -> RSSMState:
        if self.rssm is None:
            raise RuntimeError("RSSM is not initialized.")
        deterministic_dim = int(getattr(self.rssm, "deterministic_dim"))
        stochastic_dim = int(getattr(self.rssm, "stochastic_dim"))
        expected_dim = deterministic_dim + stochastic_dim
        if latent.shape[-1] != expected_dim:
            raise ValueError(f"Expected latent dim {expected_dim}, got {latent.shape[-1]}.")
        deterministic = latent[..., :deterministic_dim]
        stochastic = latent[..., deterministic_dim:]
        min_std = float(getattr(self.rssm, "min_std", 0.1))
        return RSSMState(
            deterministic=deterministic,
            stochastic=stochastic,
            mean=stochastic,
            std=torch.ones_like(stochastic) * min_std,
        )

    def get_state(self) -> Dict[str, Any]:
        return {"world_model_updates": self.world_model_updates}

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.world_model_updates = int(state.get("world_model_updates", 0))
