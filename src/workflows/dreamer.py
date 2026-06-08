"""Minimal Dreamer V1 workflow skeleton.

Dreamer keeps PlaNet's RSSM world-model training objective and learns behavior
from imagined latent rollouts instead of online CEM planning.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..components.representation_learners import LatentState, RSSMState
from .planet import PlaNetWorkflow
from .utils.base import Batch, CollectResult, PhaseConfig
from .utils.context import WorkflowContext


def lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    *,
    discount: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Compute TD(lambda) returns for imagined rewards and value estimates."""
    if rewards.shape != values.shape:
        raise ValueError(f"rewards and values must have the same shape, got {rewards.shape} and {values.shape}.")
    if rewards.shape != continues.shape:
        raise ValueError(
            f"rewards and continues must have the same shape, got {rewards.shape} and {continues.shape}."
        )
    if bootstrap.shape != rewards[:, -1].shape:
        raise ValueError(f"bootstrap shape {bootstrap.shape} does not match final value shape {rewards[:, -1].shape}.")

    returns = []
    next_return = bootstrap
    horizon = rewards.shape[1]
    for t in reversed(range(horizon)):
        next_value = bootstrap if t == horizon - 1 else values[:, t + 1]
        next_return = rewards[:, t] + float(discount) * continues[:, t] * (
            (1.0 - float(lambda_)) * next_value + float(lambda_) * next_return
        )
        returns.append(next_return)
    returns.reverse()
    return torch.stack(returns, dim=1)


class DreamerV1Workflow(PlaNetWorkflow):
    """Small Dreamer V1 recipe: RSSM world model plus imagined actor-critic."""

    def __init__(self) -> None:
        super().__init__()
        self.actor_optimizer: Optional[torch.optim.Optimizer] = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self.lambda_ = 0.95
        self.imagination_horizon = 15
        self.controller_updates = 0

    def initialize(self, context: WorkflowContext) -> None:
        super().initialize(context)
        optimizers = context.optimizers or {}
        self.actor_optimizer = optimizers.get("actor")
        self.critic_optimizer = optimizers.get("critic")
        if self.actor_optimizer is None:
            raise RuntimeError("DreamerV1Workflow requires an actor optimizer.")
        if self.critic_optimizer is None:
            raise RuntimeError("DreamerV1Workflow requires a critic optimizer.")
        if "actor" not in self.controllers:
            raise RuntimeError("DreamerV1Workflow requires an 'actor' controller.")
        if "critic" not in self.controllers:
            raise RuntimeError("DreamerV1Workflow requires a 'critic' controller.")

        algorithm = getattr(self.config, "algorithm", None)
        self.lambda_ = float(getattr(algorithm, "lambda_", getattr(algorithm, "lambda", self.lambda_)))
        self.imagination_horizon = int(getattr(algorithm, "imagination_horizon", self.imagination_horizon))

    def update_controller(self, batch: Batch, *, phase: PhaseConfig) -> Dict[str, float]:
        if self.rssm is None:
            raise RuntimeError("Dreamer controller update requires an RSSM.")
        actor = self.controllers.get("actor")
        critic = self.controllers.get("critic")
        if actor is None or critic is None:
            raise RuntimeError("Dreamer controller update requires 'actor' and 'critic' controllers.")
        if self.actor_optimizer is None or self.critic_optimizer is None:
            raise RuntimeError("Dreamer controller optimizers are not initialized.")

        horizon = int(phase.get("horizon", phase.get("imagination_horizon", self.imagination_horizon)))
        start_state = self._posterior_start_state(batch)
        world_modules = [self.rssm, self.reward_predictor, self.continue_predictor, self.observation_predictor]

        with self._temporarily_frozen(world_modules):
            with torch.no_grad():
                critic_rollout = self.imagine_rollout(
                    start_state.detach(),
                    horizon=horizon,
                    deterministic_actor=False,
                    deterministic_model=False,
                    detach_actor=True,
                )
                critic_targets = lambda_returns(
                    critic_rollout["rewards"],
                    critic_rollout["values"],
                    critic_rollout["continues"],
                    critic_rollout["bootstrap"],
                    discount=self.gamma,
                    lambda_=self.lambda_,
                ).detach()

            self.critic_optimizer.zero_grad()
            critic_values = critic(critic_rollout["states"].detach())
            critic_loss = F.mse_loss(critic_values, critic_targets)
            critic_loss.backward()
            self.critic_optimizer.step()

            with self._temporarily_frozen([critic]):
                self.actor_optimizer.zero_grad()
                actor_rollout = self.imagine_rollout(
                    start_state.detach(),
                    horizon=horizon,
                    deterministic_actor=False,
                    deterministic_model=False,
                    detach_actor=False,
                )
                actor_returns = lambda_returns(
                    actor_rollout["rewards"],
                    actor_rollout["values"],
                    actor_rollout["continues"],
                    actor_rollout["bootstrap"],
                    discount=self.gamma,
                    lambda_=self.lambda_,
                )
                actor_loss = -actor_returns.mean()
                actor_loss.backward()
                self.actor_optimizer.step()

        self.controller_updates += 1
        return {
            "controller/actor_loss": float(actor_loss.item()),
            "controller/critic_loss": float(critic_loss.item()),
            "controller/lambda_return_mean": float(actor_returns.detach().mean().item()),
            "controller/imagination_horizon": float(horizon),
        }

    def collect_step(self, step: int, *, phase: PhaseConfig) -> Optional[CollectResult]:
        del step
        if self.environment is None or self.current_obs is None:
            raise RuntimeError("DreamerV1Workflow not initialized before collect_step.")

        controller_role = str(phase.get("controller", "actor"))
        controller = self.controllers.get(controller_role)
        if controller is None:
            raise RuntimeError(f"DreamerV1Workflow requires controller '{controller_role}' for collection.")

        collect_len = int(phase.get("collect_length", self.collect_length) or self.collect_length)
        start = time.time()
        observations, actions, rewards, dones = [], [], [], []

        for _ in range(collect_len):
            obs = np.asarray(self.current_obs, dtype=np.float32)
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                if controller_role == "seed_actor":
                    action_tensor = controller.act(obs_tensor)
                else:
                    if self.rssm is None:
                        raise RuntimeError("Dreamer actor collection requires an RSSM.")
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
                    action_tensor = self._actor_action(
                        controller,
                        self.rollout_state.to_tensor(),
                        deterministic=False,
                    )
                    exploration_noise = float(phase.get("exploration_noise", self.exploration_noise) or 0.0)
                    if exploration_noise > 0.0:
                        action_tensor = action_tensor + torch.randn_like(action_tensor) * exploration_noise
                        low, high = self.get_action_bounds()
                        action_tensor = torch.clamp(action_tensor, min=low.to(self.device), max=high.to(self.device))

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

    def imagine_rollout(
        self,
        start: LatentState | torch.Tensor,
        *,
        horizon: Optional[int] = None,
        deterministic_actor: bool = False,
        deterministic_model: bool = False,
        detach_actor: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.rssm is None or self.reward_predictor is None:
            raise RuntimeError("Dreamer imagination requested before world-model components are initialized.")
        actor = self.controllers.get("actor")
        critic = self.controllers.get("critic")
        if actor is None or critic is None:
            raise RuntimeError("Dreamer imagination requires 'actor' and 'critic' controllers.")

        state = self._state_from_latent(start)
        horizon = int(horizon or self.imagination_horizon)
        states, actions, rewards, continues, values = [], [], [], [], []

        for _ in range(horizon):
            latent_tensor = state.to_tensor()
            action = self._actor_action(
                actor,
                latent_tensor.detach() if detach_actor else latent_tensor,
                deterministic=deterministic_actor,
            )
            state = self.rssm.imagine_step(state, action, deterministic=deterministic_model)
            next_latent = state.to_tensor()
            states.append(next_latent)
            actions.append(action)
            rewards.append(self.reward_predictor(next_latent))
            values.append(critic(next_latent))
            if self.continue_predictor is not None:
                continues.append(torch.sigmoid(self.continue_predictor(next_latent)))
            else:
                continues.append(torch.ones_like(rewards[-1]))

        if horizon > 0:
            bootstrap_latent = state.to_tensor()
            bootstrap_action = self._actor_action(
                actor,
                bootstrap_latent.detach() if detach_actor else bootstrap_latent,
                deterministic=deterministic_actor,
            )
            state = self.rssm.imagine_step(state, bootstrap_action, deterministic=deterministic_model)

        bootstrap = critic(state.to_tensor())
        return {
            "states": torch.stack(states, dim=1),
            "actions": torch.stack(actions, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continues": torch.stack(continues, dim=1),
            "values": torch.stack(values, dim=1),
            "bootstrap": bootstrap,
        }

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
        del observations, controller_role, action_sequence
        if latent is None:
            raise TypeError("DreamerV1Workflow.imagine expects an RSSMState or flattened latent tensor.")
        return self.imagine_rollout(
            latent,
            horizon=horizon,
            deterministic_actor=deterministic,
            deterministic_model=deterministic,
        )

    def evaluate(
        self,
        num_eval_batches: int = 5,
        max_steps_per_episode: int = 500,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        if self.rssm is None:
            raise RuntimeError("DreamerV1Workflow.evaluate requires an initialized RSSM.")
        actor = self.controllers.get("actor")
        if actor is None:
            raise RuntimeError("DreamerV1Workflow.evaluate requires an 'actor' controller.")
        env = self.eval_environment or self.environment
        if env is None:
            raise RuntimeError("DreamerV1Workflow.evaluate requires an eval or train environment.")

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
                    action_tensor = self._actor_action(
                        actor,
                        latent_step.posterior.to_tensor(),
                        deterministic=deterministic,
                    )

                env_action = action_tensor.detach().cpu().numpy()
                next_obs, reward, done, info = env.step(env_action)
                del info
                reward_arr = np.asarray(reward, dtype=np.float32).reshape(-1)
                done_arr = np.asarray(done, dtype=bool).reshape(-1)
                episode_return += reward_arr
                episode_length += 1
                prev_action = action_tensor.detach()
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

    def _posterior_start_state(self, batch: Batch) -> RSSMState:
        if self.rssm is None:
            raise RuntimeError("RSSM is not initialized.")
        tensor_batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.device)
            for key, value in batch.items()
        }
        observations = tensor_batch["observations"].to(torch.float32)
        actions = tensor_batch["actions"].to(torch.float32)
        dones = tensor_batch["dones"].to(torch.bool)
        with torch.no_grad():
            sequence = self.rssm.observe_sequence(observations, actions=actions, dones=dones)
        return sequence.last_posterior.detach()

    def _state_from_latent(self, latent: LatentState | torch.Tensor) -> RSSMState:
        if isinstance(latent, RSSMState):
            return latent.to(self.device)
        if hasattr(latent, "to_tensor") and hasattr(latent, "to"):
            return latent.to(self.device)  # type: ignore[return-value]
        if isinstance(latent, torch.Tensor):
            return self._rssm_state_from_tensor(latent.to(self.device))
        raise TypeError("Dreamer imagination expects an RSSMState or flattened latent tensor.")

    @staticmethod
    def _actor_action(actor: Any, latent_tensor: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        action = actor.act(latent_tensor, deterministic=deterministic, return_log_prob=False)
        if isinstance(action, tuple):
            action = action[0]
        return action

    @staticmethod
    @contextmanager
    def _temporarily_frozen(modules: Iterable[Optional[torch.nn.Module]]):
        params = []
        for module in modules:
            if module is None or not hasattr(module, "parameters"):
                continue
            for param in module.parameters():
                params.append((param, param.requires_grad))
                param.requires_grad_(False)
        try:
            yield
        finally:
            for param, requires_grad in params:
                param.requires_grad_(requires_grad)

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["controller_updates"] = self.controller_updates
        return state

    def set_state(self, state: Mapping[str, Any]) -> None:
        super().set_state(state)
        self.controller_updates = int(state.get("controller_updates", 0))


__all__ = ["DreamerV1Workflow", "lambda_returns"]
