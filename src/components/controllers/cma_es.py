"""CMA-ES inspired controller scaffold for the OG World Models agent.

Use this as homework: plug in the actual CMA-ES loop (using `cma` or a custom
implementation) so it can evolve a linear policy on top of [z_t, h_t].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch.distributions import Distribution

from .base import BaseController


class CMAESWorldModelController(BaseController):
    """Minimal placeholder that will eventually wrap a CMA-ES optimizer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        merged = dict(config or {})
        if kwargs:
            merged.update(kwargs)

        self.representation_dim = int(merged.get("representation_dim", 32))
        self.hidden_dim = int(merged.get("hidden_dim", 0))  # 0 == linear controller, per paper
        self.action_dim = int(merged.get("action_dim", 3))
        self.device = torch.device(merged.get("device", "cpu"))
        self.discrete_actions = bool(merged.get("discrete_actions", False))
        self.action_low = merged.get("action_low")
        self.action_high = merged.get("action_high")

        # TODO: Build the small policy network (linear or single hidden layer)
        # TODO: Integrate CMA-ES state (mean vector, covariance, sigma, etc.)
        self.policy: Optional[torch.nn.Module] = None
        self.optimizer_state: Optional[Any] = None

    def act(
        self,
        latent_state: torch.Tensor,
        dynamics_model: Optional[Any] = None,
        value_function: Optional[Any] = None,
        *,
        deterministic: bool = False,
        horizon: Optional[int] = None,
        **_: Any,
    ) -> Distribution:
        """
        Return an action distribution for the current latent.

        Homework checklist:
        - Concatenate z_t with MDN-RNN hidden state h_t before feeding the policy
        - Clamp/scale outputs to env action bounds
        - Expose a torch Distribution (Categorical or Normal) so the workflow's
          sampling utilities keep working
        """
        raise NotImplementedError("CMAESWorldModelController.act homework pending.")

    def learn(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        phase: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Run one CMA-ES update using dream rollouts collected in `batch`.

        Homework checklist:
        - Evaluate candidate weights inside dream trajectories
        - Feed returns to CMA-ES to update mean/covariance
        - Optionally sync the best candidate back into `self.policy`
        """
        raise NotImplementedError("CMAESWorldModelController.learn homework pending.")

    # ------------------------------------------------------------------
    # Serialization hooks (optional but useful for checkpoints)
    # ------------------------------------------------------------------
    def state_dict(self, *, mode: str = "checkpoint") -> Dict[str, Any]:
        """Snapshot policy weights and CMA-ES statistics."""
        # TODO: Serialize controller + optimizer state
        raise NotImplementedError("CMAESWorldModelController.state_dict homework pending.")

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore policy weights and CMA-ES statistics."""
        # TODO: Restore everything stored in state_dict()
        raise NotImplementedError("CMAESWorldModelController.load_state_dict homework pending.")


__all__ = ["CMAESWorldModelController"]
