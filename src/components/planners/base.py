"""
Base Planner Interface

This module defines the abstract base class for planners.
Planners use world models to plan sequences of actions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.distributions import Distribution

# Import component types that planners will use
from ..dynamics.base import BaseDynamicsModel
from ..value_functions.base import BaseValueFunction


class BasePlanner(nn.Module, ABC):
    """
    Abstract base class for all planners.

    Planners use world models (dynamics + value functions) to plan
    sequences of actions that maximize expected return.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize planner with configuration.

        Args:
            config: Dictionary containing hyperparameters and settings
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))
        self.horizon = config.get('horizon', 10)
        self.state_dim = config.get('state_dim')
        self.action_dim = config.get('action_dim')
        self._build_planner()

    @abstractmethod
    def _build_planner(self):
        """
        Build the planner architecture.
        Called during __init__.
        """
        pass

    @abstractmethod
    def plan(self,
            current_state: torch.Tensor,
            dynamics_model: BaseDynamicsModel,
            value_function: BaseValueFunction,
            horizon: Optional[int] = None) -> Distribution:
        """
        Plan actions given current state and world models.

        Args:
            current_state: Current state representation, shape (batch_size, state_dim)
            dynamics_model: Dynamics model for predictions
            value_function: Value function for evaluation
            horizon: Planning horizon (uses self.horizon if None)

        Returns:
            Distribution over planned actions
        """
        pass

    def plan_sequence(self,
                     current_state: torch.Tensor,
                     dynamics_model: BaseDynamicsModel,
                     value_function: BaseValueFunction,
                     horizon: Optional[int] = None) -> torch.Tensor:
        """
        Plan a deterministic sequence of actions.

        Args:
            current_state: Current state representation
            dynamics_model: Dynamics model for predictions
            value_function: Value function for evaluation
            horizon: Planning horizon

        Returns:
            Action sequence tensor, shape (batch_size, horizon, action_dim)
        """
        # Default implementation: sample from plan distribution
        action_dist = self.plan(current_state, dynamics_model, value_function, horizon)
        actions = action_dist.sample()

        # If single action returned, expand to sequence
        if actions.dim() == 2:  # (batch_size, action_dim)
            horizon = horizon or self.horizon
            actions = actions.unsqueeze(1).expand(-1, horizon, -1)

        return actions

    @property
    def requires_differentiable_dynamics(self) -> bool:
        """
        Whether planner needs gradients through dynamics model.

        Override in subclasses that use gradient-based planning.

        Returns:
            False by default
        """
        return False

    @property
    def supports_stochastic_dynamics(self) -> bool:
        """
        Whether planner can handle stochastic dynamics models.

        Override in subclasses as appropriate.

        Returns:
            True by default
        """
        return True

    def get_planner_info(self) -> Dict[str, Any]:
        """
        Get information about this planner.

        Returns:
            Dictionary with planner information
        """
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'horizon': self.horizon,
            'requires_differentiable_dynamics': self.requires_differentiable_dynamics,
            'supports_stochastic_dynamics': self.supports_stochastic_dynamics,
            'planner_type': self.__class__.__name__,
            'device': str(self.device)
        }