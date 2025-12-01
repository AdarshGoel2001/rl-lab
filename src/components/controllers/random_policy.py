"""Random policy controller for exploration and data collection.

Used during the initial data gathering phase of World Models to collect
diverse trajectories without any learned policy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class RandomPolicyController(nn.Module):
    """Samples random actions uniformly from the action space.

    This controller is used for exploration during initial data collection.
    It doesn't learn anything - just provides random actions.

    Args:
        action_dim: Dimension of the action space
        action_low: Lower bounds for each action dimension
        action_high: Upper bounds for each action dimension
        device: Device to use for tensors
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()

        merged = dict(config or {})
        if kwargs:
            merged.update(kwargs)

        self.action_dim = int(merged.get("action_dim", 3))

        # Action space bounds
        action_low = merged.get("action_low", [-1.0, 0.0, 0.0])
        action_high = merged.get("action_high", [1.0, 1.0, 1.0])

        if isinstance(action_low, (list, tuple)):
            action_low = np.array(action_low, dtype=np.float32)
        if isinstance(action_high, (list, tuple)):
            action_high = np.array(action_high, dtype=np.float32)

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32)

        device = merged.get("device", "cpu")
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.action_low = self.action_low.to(self.device)
        self.action_high = self.action_high.to(self.device)

    def act(
        self,
        observation: torch.Tensor,
        hidden_state: Optional[Any] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """Sample random actions uniformly from action space bounds.

        Args:
            observation: Current observation (not used, but accepted for API compatibility)
            hidden_state: Hidden state (not used, but accepted for API compatibility)

        Returns:
            Random actions with shape (batch_size, action_dim)
        """
        # Infer batch size from observation
        if observation.dim() == 1:
            batch_size = 1
        else:
            batch_size = observation.shape[0]

        # Sample uniformly between action_low and action_high
        # uniform(0, 1) * (high - low) + low
        random_actions = torch.rand(
            batch_size, self.action_dim,
            device=self.device,
            dtype=torch.float32
        )

        actions = random_actions * (self.action_high - self.action_low) + self.action_low

        return actions

    def forward(self, *args, **kwargs):
        """Forward pass - alias for act() to match standard controller interface."""
        return self.act(*args, **kwargs)

    def to(self, device):
        """Move controller to specified device."""
        super().to(device)
        self.device = device
        self.action_low = self.action_low.to(device)
        self.action_high = self.action_high.to(device)
        return self


__all__ = ["RandomPolicyController"]
