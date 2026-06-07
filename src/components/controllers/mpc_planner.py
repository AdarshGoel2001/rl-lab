"""Model predictive controller using the cross-entropy method."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Optional

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

        self.register_buffer("action_mean", action_mean, persistent=False)
        self.register_buffer("action_std", action_std, persistent=False)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """Restore learned planner state while ignoring transient CEM buffers."""
        filtered = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if key not in {"action_mean", "action_std"}
        )
        return super().load_state_dict(filtered, strict=strict, assign=assign)

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

        plan = self._plan_batch(latent_state, workflow, low, high)
        return plan[:, 0, :]

    def _plan_batch(
        self,
        latent: Tensor,
        workflow: Any,
        low: Tensor,
        high: Tensor,
    ) -> Tensor:
        batch_size = int(latent.shape[0])
        mean = torch.zeros(batch_size, self.horizon, self.action_dim, device=self.device)
        std = torch.full((batch_size, self.horizon, self.action_dim), self.std_init, device=self.device)

        low = low.view(1, 1, 1, -1)
        high = high.view(1, 1, 1, -1)

        for _ in range(self.iterations):
            base_mean = mean.unsqueeze(1).repeat(1, self.num_samples, 1, 1)
            base_std = std.clamp(min=self.std_min).unsqueeze(1).repeat(1, self.num_samples, 1, 1)
            noise = torch.randn_like(base_mean)
            samples = base_mean + base_std * noise
            samples = torch.clamp(samples, min=low, max=high)

            flat_samples = samples.reshape(batch_size * self.num_samples, self.horizon, self.action_dim)
            latent_batch = latent.unsqueeze(1).repeat(1, self.num_samples, 1)
            latent_batch = latent_batch.reshape(batch_size * self.num_samples, latent.shape[-1])
            rollout = workflow.imagine(
                latent=latent_batch,
                horizon=self.horizon,
                action_sequence=flat_samples,
            )
            discount = workflow.gamma if hasattr(workflow, "gamma") else 0.99
            values_tensor = self._score_rollout(rollout, gamma=discount).reshape(batch_size, self.num_samples)
            _, top_indices = torch.topk(values_tensor, k=self.top_k, dim=1)
            elite_indices = top_indices[:, :, None, None].expand(-1, -1, self.horizon, self.action_dim)
            elite_actions = samples.gather(dim=1, index=elite_indices)

            mean = elite_actions.mean(dim=1)
            std = elite_actions.std(dim=1, unbiased=False)

        return mean

    def _score_rollout(self, rollout: dict[str, Tensor], *, gamma: float) -> Tensor:
        rewards = rollout["rewards"].to(self.device).squeeze(-1)
        if not self.use_continuation:
            if rewards.dim() <= 1:
                return rewards.sum()
            return rewards.sum(dim=-1)

        continues = rollout.get("continues")
        if continues is None:
            continues_tensor = torch.ones_like(rewards)
        else:
            continues_tensor = continues.to(self.device).squeeze(-1).clamp(0.0, 1.0)
        bootstrap = rollout.get("bootstrap")

        batch_shape = rewards.shape[:-1]
        discount = torch.ones(batch_shape, device=self.device, dtype=rewards.dtype)
        value = torch.zeros(batch_shape, device=self.device, dtype=rewards.dtype)
        gamma_tensor = torch.as_tensor(gamma, device=self.device, dtype=rewards.dtype)

        for t in range(rewards.shape[-1]):
            value = value + discount * rewards[..., t]
            discount = discount * gamma_tensor * continues_tensor[..., t]

        if bootstrap is None:
            return value
        return value + discount * bootstrap.to(self.device).squeeze(-1)
