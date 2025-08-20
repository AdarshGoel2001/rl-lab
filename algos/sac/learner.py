"""SAC learner: update step."""

from typing import Any, Dict, Optional


class SACLearner:
    """Learner component for Soft Actor-Critic (SAC)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    def update(self, batch: Any) -> Dict[str, float]:
        """Run a single SAC update step and return scalar metrics."""
        raise NotImplementedError


