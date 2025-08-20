"""Training runners for collecting experience."""

from __future__ import annotations

from typing import Any

import torch

from .buffers import RolloutBuffer


class OnPolicyRunner:
    """Collect trajectories for on-policy algorithms like PPO."""

    def __init__(
        self,
        env: Any,
        actor: Any,
        critic: Any,
        buffer: RolloutBuffer,
        n_steps: int,
        device: torch.device | str = "cpu",
    ) -> None:
        self.env = env
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.n_steps = n_steps
        self.device = device

    def run(self) -> None:
        obs, _ = self.env.reset()
        for _ in range(self.n_steps):
            obs_tensor = torch.as_tensor(obs, device=self.device).float()
            with torch.no_grad():
                action, log_prob, _ = self.actor.act(obs_tensor)
                value = self.critic.value(obs_tensor)
            next_obs, reward, terminated, truncated, _ = self.env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated
            self.buffer.add(
                obs_tensor,
                torch.as_tensor(action, device=self.device),
                torch.tensor(reward, device=self.device),
                torch.tensor(done, dtype=torch.float32, device=self.device),
                value,
                log_prob,
            )
            obs = next_obs
            if done:
                obs, _ = self.env.reset()


class OffPolicyRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        pass

    def run(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError


class WorldModelRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        pass

    def run(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError

