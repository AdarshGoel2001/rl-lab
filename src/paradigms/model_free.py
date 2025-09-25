"""
Model-Free Paradigm

Generic model-free RL paradigm that handles standard forward pass.
Specific algorithms (PPO, SAC, etc.) inherit from this and implement
their own loss computation and update methods.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, Union, Tuple
import torch
from torch.distributions import Distribution

from .base import BaseParadigm
from ..components.encoders.base import BaseEncoder
from ..components.representation_learners.base import BaseRepresentationLearner
from ..components.policy_heads.base import BasePolicyHead
from ..components.value_functions.base import BaseValueFunction


class ModelFreeParadigm(BaseParadigm):
    """
    Generic model-free RL paradigm.

    Handles standard forward pass and provides interface that algorithms
    like PPO, SAC, A2C can inherit from. Contains all generic model-free
    functionality while leaving algorithm-specific loss/update to subclasses.
    """

    def __init__(self,
                 encoder: BaseEncoder,
                 representation_learner: BaseRepresentationLearner,
                 policy_head: BasePolicyHead,
                 value_function: BaseValueFunction,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize model-free paradigm.

        Args:
            encoder: Encoder for observation processing
            representation_learner: Representation learning component
            policy_head: Policy head for action generation
            value_function: Value function for state evaluation
            config: Optional configuration dictionary
        """
        super().__init__(encoder, representation_learner, policy_head, config)
        self.value_function = value_function

        # Move value function to device
        self.value_function.to(self.device)

        # Track training step for compatibility
        self.step = 0

    def forward(self,
                observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
                context: Optional[Dict[str, Any]] = None) -> Distribution:
        """
        Generic forward pass: observations -> action distribution.

        This is the standard model-free forward pass used by all algorithms.

        Args:
            observations: Raw observations from environment
            context: Optional context information

        Returns:
            Action distribution
        """
        # Encode observations
        features = self.encoder(observations)

        # Learn representations (usually identity for model-free)
        representations = self.representation_learner.encode(features)

        # Generate action distribution
        action_dist = self.policy_head(representations, context)

        return action_dist

    def act(self,
            observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
            context: Optional[Dict[str, Any]] = None,
            deterministic: bool = False) -> torch.Tensor:
        """
        Generic action sampling.

        Args:
            observations: Raw observations
            context: Optional context
            deterministic: If True, select best action

        Returns:
            Action tensor
        """
        action_dist = self.forward(observations, context)

        if deterministic:
            if hasattr(action_dist, 'mode'):
                return action_dist.mode
            else:
                return action_dist.mean
        else:
            return action_dist.sample()

    def get_value(self,
                  observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Get value estimates for observations.

        Args:
            observations: Raw observations

        Returns:
            Value estimates
        """
        features = self.encoder(observations)
        representations = self.representation_learner.encode(features)
        return self.value_function(representations)

    def get_action_and_value(self,
                            observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
                            context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get actions and values in one forward pass (for efficiency).

        This is the key method that trainer expects - same interface as original algorithms.

        Args:
            observations: Raw observations
            context: Optional context

        Returns:
            Tuple of (actions, log_probs, values)
        """
        # Encode observations
        features = self.encoder(observations)
        representations = self.representation_learner.encode(features)

        # Get action distribution and sample
        action_dist = self.policy_head(representations, context)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)

        # Handle multi-dimensional actions - sum log_probs
        if not self.policy_head.discrete_actions and log_probs.dim() > 1:
            log_probs = log_probs.sum(dim=-1)

        # Get values
        values = self.value_function(representations)
        values = values.squeeze(-1)  # Remove last dimension for compatibility

        return actions, log_probs, values

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.

        Used during training to get log_probs, values, and entropy for given actions.

        Args:
            observations: Batch of observations
            actions: Batch of actions to evaluate

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        features = self.encoder(observations)
        representations = self.representation_learner.encode(features)

        # Get policy distribution
        action_dist = self.policy_head(representations)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()

        # Handle multi-dimensional actions
        if not self.policy_head.discrete_actions:
            if log_probs.dim() > 1:
                log_probs = log_probs.sum(dim=-1)
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

        # Get values
        values = self.value_function(representations)
        values = values.squeeze(-1)

        return log_probs, values, entropy

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute algorithm-specific losses.

        This is implemented by specific algorithms like PPO, SAC, etc.
        Each algorithm has its own loss computation logic.

        Args:
            batch: Experience batch

        Returns:
            Dictionary of loss components
        """
        pass

    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Algorithm-specific update method.

        This is implemented by specific algorithms like PPO, SAC, etc.
        Each algorithm has its own update logic (multi-epoch, single-step, etc.).

        Args:
            batch: Experience batch

        Returns:
            Dictionary of training metrics
        """
        pass

    # No need to override to(), train(), eval() - BaseParadigm handles all components automatically

    def _save_additional_components(self) -> Dict[str, Any]:
        """Save value function state."""
        return {
            'value_function': self.value_function.state_dict()
        }

    def _load_additional_components(self, checkpoint: Dict[str, Any]):
        """Load value function state."""
        if 'value_function' in checkpoint:
            self.value_function.load_state_dict(checkpoint['value_function'])

    def get_paradigm_info(self) -> Dict[str, Any]:
        """Get information about this paradigm and its components."""
        info = super().get_paradigm_info()
        info['value_info'] = self.value_function.get_value_info()
        return info