"""Minimal viable world-model paradigm implementation."""

from typing import Dict, Any, Optional

from ...utils.registry import register_paradigm
from .paradigm import BaseWorldModelParadigm


@register_paradigm('world_model_mvp')
@register_paradigm('world_model')  # Default alias for backwards compatibility
class WorldModelMVPParadigm(BaseWorldModelParadigm):
    """Concrete world-model paradigm matching the MVP implementation."""

    default_config: Dict[str, Any] = {
        'world_model_lr': 1e-4,
        'actor_lr': 3e-5,
        'critic_lr': 3e-5,
        'imagination_horizon': 15,
        'gamma': 0.99,
        'lambda_return': 0.95,
        'entropy_coef': 0.01,
        'max_grad_norm': 1.0,
    }

    def __init__(self, *args, config: Optional[Dict[str, Any]] = None, **kwargs):
        combined_config: Dict[str, Any] = dict(self.default_config)
        if config:
            combined_config.update(config)
        super().__init__(*args, config=combined_config, **kwargs)
