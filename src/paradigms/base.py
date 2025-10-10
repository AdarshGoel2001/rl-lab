"""
Base Paradigm Interface

This module defines the abstract base class for all paradigms.
Paradigms are high-level agent architectures that compose components
to create complete RL agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from torch.distributions import Distribution

from ..components.encoders.base import BaseEncoder
from ..components.world_models.representation_learners.base import BaseRepresentationLearner
from ..components.policy_heads.base import BasePolicyHead


class BaseParadigm(ABC):
    """
    Abstract base class for all paradigms.

    Paradigms compose different components to create complete RL agents.
    They handle the full forward pass and loss computation for training.

    Core components that all paradigms must have:
    - encoder: Transforms observations to features
    - representation_learner: Learns structured representations
    - policy_head: Converts representations to actions
    """

    def __init__(self,
                 encoder: BaseEncoder,
                 representation_learner: BaseRepresentationLearner,
                 policy_head: BasePolicyHead,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize paradigm with core components.

        Args:
            encoder: Encoder for observation processing
            representation_learner: Representation learning component
            policy_head: Policy head for action generation
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.device = torch.device(self.config.get('device', 'cpu'))

        # Core components (all paradigms must have these)
        self.encoder = encoder
        self.representation_learner = representation_learner
        self.policy_head = policy_head

        # Standard optimizer registry - subclasses should populate this
        # Example: self.optimizers = {'actor': adam_opt, 'critic': adam_opt2}
        self.optimizers = {}

        # Training step counter
        self.step = 0

        # Move all components to device
        self.to(self.device)

    @abstractmethod
    def forward(self,
                observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
                context: Optional[Dict[str, Any]] = None) -> Distribution:
        """
        Complete forward pass through the agent.

        Args:
            observations: Raw observations from environment
            context: Optional context information (goals, etc.)

        Returns:
            Action distribution
        """
        pass

    @abstractmethod
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute all learning objectives for this paradigm.

        Args:
            batch: Dictionary containing experience batch

        Returns:
            Dictionary of loss components
        """
        pass

    def act(self,
            observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
            context: Optional[Dict[str, Any]] = None,
            deterministic: bool = False) -> torch.Tensor:
        """
        Select action given observations.

        Args:
            observations: Raw observations from environment
            context: Optional context information
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

    def get_representation(self,
                          observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Get learned representation for given observations.

        Useful for analysis and debugging.

        Args:
            observations: Raw observations

        Returns:
            Representation tensor
        """
        features = self.encoder(observations)
        return self.representation_learner.encode(features)

    def save_checkpoint(self) -> Dict[str, Any]:
        """
        Save paradigm state for resuming training.

        Returns:
            Dictionary containing all state needed for resuming
        """
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'representation_learner': self.representation_learner.state_dict(),
            'policy_head': self.policy_head.state_dict(),
            'config': self.config,
            'paradigm_type': self.__class__.__name__,
            'step': self.step,
        }

        # Save all optimizers centrally
        if self.optimizers:
            checkpoint['optimizers'] = {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            }

        # Add additional component states (subclasses can override this method)
        checkpoint.update(self._save_additional_components())

        return checkpoint

    def load_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Load paradigm state from checkpoint.

        Args:
            checkpoint: Dictionary returned by save_checkpoint()
        """
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.representation_learner.load_state_dict(checkpoint['representation_learner'])
        self.policy_head.load_state_dict(checkpoint['policy_head'])

        # Restore step counter
        if 'step' in checkpoint:
            self.step = checkpoint['step']

        # Restore all optimizers centrally
        if 'optimizers' in checkpoint and self.optimizers:
            for name, opt_state in checkpoint['optimizers'].items():
                if name in self.optimizers:
                    self.optimizers[name].load_state_dict(opt_state)

        # Load additional component states (subclasses can override this method)
        self._load_additional_components(checkpoint)

    def _save_additional_components(self) -> Dict[str, Any]:
        """
        Save additional components specific to this paradigm.

        Override in subclasses that have additional components.

        Returns:
            Dictionary of additional component states
        """
        return {}

    def _load_additional_components(self, checkpoint: Dict[str, Any]):
        """
        Load additional components specific to this paradigm.

        Override in subclasses that have additional components.

        Args:
            checkpoint: Full checkpoint dictionary
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Alias for save_checkpoint() to support CheckpointManager interface.

        Returns:
            Dictionary containing all state needed for resuming
        """
        return self.save_checkpoint()

    def set_state(self, state: Dict[str, Any]):
        """
        Alias for load_checkpoint() to support CheckpointManager interface.

        Args:
            state: State dictionary from get_state()
        """
        self.load_checkpoint(state)

    @property
    def paradigm_type(self) -> str:
        """Get paradigm type string."""
        return self.__class__.__name__

    def get_paradigm_info(self) -> Dict[str, Any]:
        """
        Get information about this paradigm and its components.

        Returns:
            Dictionary with paradigm information
        """
        return {
            'paradigm_type': self.paradigm_type,
            'encoder_info': self.encoder.get_feature_info(),
            'representation_info': self.representation_learner.get_representation_info(),
            'policy_info': self.policy_head.get_policy_info(),
            'device': str(self.device)
        }

    def to(self, device: Union[str, torch.device]):
        """Move all components to specified device. Handles any optional components automatically."""
        self.device = torch.device(device)

        # All possible component types - handles modularity automatically
        component_attrs = ['encoder', 'representation_learner', 'policy_head', 'value_function',
                          'dynamics_model', 'reward_model', 'world_model']

        for attr_name in component_attrs:
            if hasattr(self, attr_name):
                component = getattr(self, attr_name)
                if component is not None:
                    component.to(self.device)

    def train(self):
        """Set all components to training mode. Handles any optional components automatically."""
        component_attrs = ['encoder', 'representation_learner', 'policy_head', 'value_function',
                          'dynamics_model', 'reward_model', 'world_model']

        for attr_name in component_attrs:
            if hasattr(self, attr_name):
                component = getattr(self, attr_name)
                if component is not None:
                    component.train()

    def eval(self):
        """Set all components to evaluation mode. Handles any optional components automatically."""
        component_attrs = ['encoder', 'representation_learner', 'policy_head', 'value_function',
                          'dynamics_model', 'reward_model', 'world_model']

        for attr_name in component_attrs:
            if hasattr(self, attr_name):
                component = getattr(self, attr_name)
                if component is not None:
                    component.eval()
