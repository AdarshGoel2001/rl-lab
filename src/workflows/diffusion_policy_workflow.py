"""Diffusion Policy Workflow.

Your job: Implement the training loss computation.
The scaffold handles environment interaction, buffer management, and orchestrator interface.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F

from .utils.base import WorldModelWorkflow, CollectResult


@dataclass
class DiffusionConfig:
    """Config extracted from hydra config."""
    lr: float = 1e-4
    horizon: int = 16
    num_diffusion_steps: int = 100
    batch_size: int = 256
    grad_clip: Optional[float] = 1.0
    actions_per_plan: int = 3  # How many actions to execute before re-planning


class DiffusionPolicyWorkflow(WorldModelWorkflow):
    """
    Workflow for training diffusion policy via behavioral cloning.

    No world model - just learns to imitate actions from buffer.
    """

    def initialize(self, context) -> None:
        """
        Called once at start. Set up controller, optimizer, buffer references.
        """
        self.context = context
        self.device = context.device
        self.config = self._extract_config(context.config)

        # Get controller from context
        self.controller = context.controller_manager.get("actor")

        # Get optimizer from context, or create fallback
        if context.optimizers and "actor" in context.optimizers:
            self.optimizer = context.optimizers["actor"]
        else:
            self.optimizer = torch.optim.AdamW(
                self.controller.parameters(),
                lr=self.config.lr,
            )

        # Get buffer reference
        self.buffer = context.buffers.get("replay") or context.buffers.get("train")

        # Get environment
        self.env = context.train_environment

        # Track current observation for collect_step
        self._current_obs = None
        self._episode_reward = 0.0
        self._episode_length = 0

    def _extract_config(self, full_config) -> DiffusionConfig:
        """Pull out relevant config values."""
        algo = getattr(full_config, "algorithm", {}) or {}
        if hasattr(algo, "__dict__"):
            algo = algo.__dict__

        return DiffusionConfig(
            lr=algo.get("lr", 1e-4),
            horizon=algo.get("horizon", 16),
            num_diffusion_steps=algo.get("num_diffusion_steps", 100),
            batch_size=algo.get("batch_size", 256),
            grad_clip=algo.get("grad_clip", 1.0),
            actions_per_plan=algo.get("actions_per_plan", 3),
        )

    # =========================================================================
    # TODO 5: Implement training loss
    # =========================================================================
    def update_controller(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        phase: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Train diffusion policy on a batch.

        The diffusion training objective:
            1. Sample random timesteps t ~ Uniform(0, T)
            2. Sample noise eps ~ N(0, I)
            3. Create noisy actions: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
            4. Predict noise: eps_pred = network(x_t, t, obs)
            5. Loss = MSE(eps_pred, eps)

        Args:
            batch: Contains 'observations' (B, obs_dim) and 'actions' (B, horizon, action_dim)
                   Note: you may need to reshape actions if buffer stores (B, action_dim)

        Returns:
            Dict with 'loss' and any other metrics you want to track
        """
        # Extract from batch
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)

        B = observations.shape[0]

        # Ensure actions have horizon dimension
        # Buffer might store (B, action_dim), need (B, horizon, action_dim)
        if actions.dim() == 2:
            # For now, repeat single action across horizon
            # TODO: Better approach is to store action sequences in buffer
            actions = actions.unsqueeze(1).expand(B, self.config.horizon, -1)

        # Use controller's compute_loss (handles diffusion internally)
        loss = self.controller.compute_loss(observations, actions)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # Return metrics
        return {
            "controller/loss": loss.item(),
        }

    # =========================================================================
    # Environment interaction - implemented for you
    # =========================================================================
    def collect_step(
        self,
        step: int,
        *,
        phase: Optional[Dict[str, Any]] = None,
    ) -> Optional[CollectResult]:
        """
        Interact with environment using learned policy.
        """
        # Initialize if needed
        if self._current_obs is None:
            self._current_obs = self.env.reset()
            self._episode_reward = 0.0
            self._episode_length = 0

        # Get action from controller
        obs_tensor = torch.as_tensor(self._current_obs, dtype=torch.float32, device=self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            action = self.controller.act(obs_tensor)

        # Step environment
        action_np = action.cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.squeeze(0)

        next_obs, reward, done, info = self.env.step(action_np)

        # Handle vectorized env output
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = float(reward[0]) if len(reward) > 0 else float(reward)
            done = bool(done[0]) if hasattr(done, '__len__') else bool(done)

        # Track episode stats
        self._episode_reward += reward
        self._episode_length += 1

        # Store transition in buffer
        self.buffer.add(
            observations=self._current_obs,
            actions=action_np,
            rewards=reward,
            dones=done,
        )

        # Handle episode end
        extras = {}
        if done:
            extras["episode_reward"] = self._episode_reward
            extras["episode_length"] = self._episode_length
            self._current_obs = self.env.reset()
            self._episode_reward = 0.0
            self._episode_length = 0
        else:
            self._current_obs = next_obs

        return CollectResult(steps=1, extras=extras)

    def update_world_model(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        phase: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Not used - diffusion policy has no world model."""
        return {}

    def imagine(self, **kwargs) -> Dict[str, Any]:
        """Not used - diffusion policy doesn't imagine."""
        return {}

    def evaluate(
        self,
        num_episodes: int = 10,
        max_steps_per_episode: int = 1000,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Run evaluation episodes and return statistics.

        Uses receding horizon: gets action sequence from controller,
        executes actions_per_plan steps, then re-plans.

        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Max steps before truncating an episode
            deterministic: Whether to use deterministic actions (ignored for diffusion)

        Returns:
            Dict with eval metrics: mean/std/min/max return, mean length
        """
        self.controller.eval()

        actions_per_plan = self.config.actions_per_plan
        episode_returns = []
        episode_lengths = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            episode_return = 0.0
            episode_length = 0
            done = False

            while episode_length < max_steps_per_episode and not done:
                # Get action sequence from controller
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)

                with torch.no_grad():
                    action_seq = self.controller.act(obs_tensor)  # (1, horizon, action_dim)

                # Extract actions: (horizon, action_dim)
                actions = action_seq.cpu().numpy().squeeze(0)

                # Execute first actions_per_plan actions
                for i in range(min(actions_per_plan, len(actions))):
                    if done or episode_length >= max_steps_per_episode:
                        break

                    action = actions[i]

                    # Step environment
                    result = self.env.step(action)
                    if len(result) == 5:
                        obs, reward, terminated, truncated, info = result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = result

                    # Handle vectorized env output
                    if isinstance(reward, (list, tuple, np.ndarray)):
                        reward = float(reward[0]) if hasattr(reward, '__len__') and len(reward) > 0 else float(reward)
                    if hasattr(done, '__len__'):
                        done = bool(done[0]) if len(done) > 0 else bool(done)

                    episode_return += reward
                    episode_length += 1

            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

        self.controller.train()

        # Compute statistics
        returns_arr = np.array(episode_returns)
        lengths_arr = np.array(episode_lengths)

        metrics = {
            "return_mean": float(returns_arr.mean()),
            "return_std": float(returns_arr.std()),
            "return_min": float(returns_arr.min()),
            "return_max": float(returns_arr.max()),
            "episode_length_mean": float(lengths_arr.mean()),
            "num_episodes": float(num_episodes),
        }

        return metrics


# Need this import for type hints
import numpy as np
