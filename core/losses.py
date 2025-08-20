"""Loss function implementations for various algorithms."""

from __future__ import annotations

from typing import Tuple

import torch


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO surrogate, value and entropy losses."""

    ratio = (new_log_probs - old_log_probs).exp()
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(surrogate1, surrogate2).mean()

    value_loss = 0.5 * (returns - values).pow(2).mean()
    entropy_loss = -(-new_log_probs).mean()  # placeholder if entropy from dist not provided
    return policy_loss, value_loss, entropy_loss


def sac_losses(*args: torch.Tensor, **kwargs: torch.Tensor) -> Tuple:
    raise NotImplementedError


def world_model_losses(*args: torch.Tensor, **kwargs: torch.Tensor) -> Tuple:
    raise NotImplementedError


