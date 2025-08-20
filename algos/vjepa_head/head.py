"""V-JEPA action-conditioned predictor head.

This module defines a head trained on top of a frozen video encoder to predict
future representations conditioned on actions.
"""

from typing import Any


class VJEPAHead:
    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def predict(self, features: Any, actions: Any) -> Any:
        raise NotImplementedError


