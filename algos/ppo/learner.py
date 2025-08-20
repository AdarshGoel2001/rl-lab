"""PPO learner implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn, optim

from core.losses import ppo_loss
from core.buffers import RolloutBuffer


@dataclass
class PPOLearner:
    """Learner component for Proximal Policy Optimization (PPO)."""

    actor: nn.Module
    critic: nn.Module
    optimizer: optim.Optimizer
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.config = self.config or {}
        self.clip_range = self.config.get("clip_range", 0.2)
        self.value_coef = self.config.get("value_coef", 0.5)
        self.entropy_coef = self.config.get("entropy_coef", 0.0)
        self.update_epochs = self.config.get("update_epochs", 1)
        self.minibatch_size = self.config.get("minibatch_size", 64)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run multiple PPO epochs over data from ``buffer``."""

        metrics: Dict[str, float] = {}
        for _ in range(self.update_epochs):
            for batch in buffer.get_minibatches(self.minibatch_size):
                dist = self.actor.distribution(batch["obs"])
                new_log_probs = dist.log_prob(batch["actions"])
                entropy = dist.entropy().mean()
                values = self.critic.value(batch["obs"])

                policy_loss, value_loss, _ = ppo_loss(
                    new_log_probs,
                    batch["log_probs"],
                    batch["advantages"],
                    values,
                    batch["returns"],
                    self.clip_range,
                )
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                metrics = {
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                }

        buffer.clear()
        return metrics


