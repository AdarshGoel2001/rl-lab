"""Policy network wrapper for PPO."""

from __future__ import annotations

from typing import Any, Tuple

import torch


class PPOActor(torch.nn.Module):
    """Encapsulates encoder and policy head for action selection."""

    def __init__(self, encoder: Any, policy_head: Any) -> None:
        super().__init__()
        self.encoder = encoder
        self.policy_head = policy_head

    def distribution(self, obs: torch.Tensor):
        feats = self.encoder(obs)
        return self.policy_head(feats)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[Any, Any, dict]:
        """Compute an action given an observation."""

        dist = self.distribution(obs)
        if deterministic:
            if hasattr(dist, "mean"):
                action = dist.mean
            else:
                action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, {"entropy": entropy}


