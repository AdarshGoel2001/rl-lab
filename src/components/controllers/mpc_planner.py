"""Model predictive controller using the cross-entropy method."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor


class MPCPlanner(nn.Module):
    """Cross-entropy method planner that queries workflow imagination."""

    def __init__(
        self,
        *,
        representation_dim: int,
        action_dim: int,
        horizon: int = 12,
        num_samples: int = 512,
        top_k: int = 64,
        iterations: int = 6,
        std_init: float = 1.0,
        std_min: float = 0.05,
        use_continuation: bool = False,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.representation_dim = representation_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.top_k = min(top_k, num_samples)
        self.iterations = iterations
        self.std_init = std_init
        self.std_min = std_min
        self.use_continuation = use_continuation

        self.device = torch.device(device) if device is not None else torch.device("cpu")

        action_mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        action_std = torch.full((self.horizon, self.action_dim), std_init, device=self.device)

        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

    def act(
        self,
        latent_state: Tensor,
        *,
        workflow: Any,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        if workflow is None:
            raise ValueError("MPCPlanner requires a workflow instance for imagination.")

        if latent_state.dim() == 1:
            latent_state = latent_state.unsqueeze(0)
        latent_state = latent_state.to(self.device)

        low, high = workflow.get_action_bounds()
        low = low.to(self.device)
        high = high.to(self.device)

        actions: list[Tensor] = []
        for env_latent in latent_state:
            env_latent = env_latent.unsqueeze(0)
            plan = self._plan_single(env_latent, workflow, low, high)
            actions.append(plan[0])

        return torch.stack(actions, dim=0)

    def _plan_single(
        self,
        latent: Tensor,
        workflow: Any,
        low: Tensor,
        high: Tensor,
    ) -> Tensor:
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.full((self.horizon, self.action_dim), self.std_init, device=self.device)

        low = low.view(1, 1, -1)
        high = high.view(1, 1, -1)

        for _ in range(self.iterations):
            base_mean = mean.unsqueeze(0).repeat(self.num_samples, 1, 1)
            base_std = std.clamp(min=self.std_min).unsqueeze(0).repeat(self.num_samples, 1, 1)
            noise = torch.randn_like(base_mean)
            samples = base_mean + base_std * noise
            samples = torch.clamp(samples, min=low, max=high)

            values = []
            for candidate in samples:
                rollout = workflow.imagine(
                    latent=latent,
                    horizon=self.horizon,
                    action_sequence=candidate.unsqueeze(0),
                )
                discount = workflow.gamma if hasattr(workflow, "gamma") else 0.99
                value = self._score_rollout(
                    {
                        "rewards": rollout["rewards"].squeeze(0),
                        **(
                            {"continues": rollout["continues"].squeeze(0)}
                            if self.use_continuation and "continues" in rollout
                            else {}
                        ),
                        "bootstrap": rollout["bootstrap"].squeeze(0),
                    },
                    gamma=discount,
                )
                values.append(value)

            values_tensor = torch.stack(values)
            _, top_indices = torch.topk(values_tensor, k=self.top_k, dim=0)
            elite_actions = samples[top_indices]

            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0)

        return mean

    def _score_rollout(self, rollout: dict[str, Tensor], *, gamma: float) -> Tensor:
        rewards = rollout["rewards"].to(self.device).squeeze(-1)
        if not self.use_continuation:
            return rewards.sum()

        continues = rollout.get("continues")
        if continues is None:
            continues_tensor = torch.ones_like(rewards)
        else:
            continues_tensor = continues.to(self.device).squeeze(-1).clamp(0.0, 1.0)
        bootstrap = rollout.get("bootstrap")

        discount = torch.ones((), device=self.device, dtype=rewards.dtype)
        value = torch.zeros((), device=self.device, dtype=rewards.dtype)
        gamma_tensor = torch.as_tensor(gamma, device=self.device, dtype=rewards.dtype)

        for reward, cont in zip(rewards, continues_tensor):
            value = value + discount * reward
            discount = discount * gamma_tensor * cont

        if bootstrap is None:
            return value
        return value + discount * bootstrap.to(self.device).squeeze()
