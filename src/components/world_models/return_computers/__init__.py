"""Return computation strategies for world model paradigms."""

from .base import BaseReturnComputer
from .none import NoReturnComputer
from .discounted import DiscountedReturnComputer
from .n_step import NStepReturnComputer
from .td_lambda import TDLambdaReturnComputer

__all__ = [
    "BaseReturnComputer",
    "NoReturnComputer",
    "DiscountedReturnComputer",
    "NStepReturnComputer",
    "TDLambdaReturnComputer",
]
