"""
World Model Paradigm

This module implements the world model RL paradigm (e.g., Dreamer, MuZero).
It learns explicit world models and uses them for planning and imagination.
"""

from typing import Dict, Any, Optional, Union
import torch
import torch.nn.functional as F
from torch.distributions import Distribution

from .base import BaseParadigm
from ..components.encoders.base import BaseEncoder
from ..components.representation_learners.base import BaseRepresentationLearner
from ..components.dynamics.base import BaseDynamicsModel
from ..components.policy_heads.base import BasePolicyHead
from ..components.value_functions.base import BaseValueFunction
from ..components.planners.base import BasePlanner
from ..utils.registry import register_paradigm


@register_paradigm("world_model")
class WorldModelParadigm(BaseParadigm):
    """
    World model RL paradigm implementation.

    This paradigm learns explicit world models (dynamics) and uses them
    for planning and imagination rollouts. Suitable for algorithms like
    Dreamer, MuZero, etc.

    Required components:
    - encoder: Observation processing
    - representation_learner: Feature learning and reconstruction
    - dynamics_model: State transition prediction
    - policy_head: Action generation
    - value_function: Value estimation
    Optional:
    - planner: For explicit planning (otherwise uses direct policy)
    """

    def __init__(self,
                 encoder: BaseEncoder,
                 representation_learner: BaseRepresentationLearner,
                 dynamics_model: BaseDynamicsModel,
                 policy_head: BasePolicyHead,
                 value_function: BaseValueFunction,
                 planner: Optional[BasePlanner] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize world model paradigm.

        Args:
            encoder: Encoder for observation processing
            representation_learner: Representation learning component
            dynamics_model: Dynamics model for state transitions
            policy_head: Policy head for action generation
            value_function: Value function for state evaluation
            planner: Optional planner for explicit planning
            config: Optional configuration dictionary
        """
        super().__init__(encoder, representation_learner, policy_head, config)

        self.dynamics_model = dynamics_model
        self.value_function = value_function
        self.planner = planner

        # Move additional components to device
        self.dynamics_model.to(self.device)
        self.value_function.to(self.device)
        if self.planner is not None:
            self.planner.to(self.device)

    def forward(self,
                observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
                context: Optional[Dict[str, Any]] = None) -> Distribution:
        """
        Forward pass: observations -> action distribution.

        Uses planner if available, otherwise uses direct policy.

        Args:
            observations: Raw observations from environment
            context: Optional context information

        Returns:
            Action distribution
        """
        # Encode observations to state representations
        features = self.encoder(observations)
        states = self.representation_learner.encode(features)

        if self.planner is not None:
            # Use planner for action selection
            action_dist = self.planner.plan(
                states, self.dynamics_model, self.value_function
            )
        else:
            # Direct policy prediction
            action_dist = self.policy_head(states, context)

        return action_dist

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
        states = self.representation_learner.encode(features)
        return self.value_function(states)

    def rollout_imagination(self,
                           initial_states: torch.Tensor,
                           length: int) -> Dict[str, torch.Tensor]:
        """
        Rollout trajectories in imagination using the world model.

        Args:
            initial_states: Initial state representations
            length: Length of imagination rollout

        Returns:
            Dictionary containing imagined trajectory
        """
        batch_size = initial_states.shape[0]
        device = initial_states.device

        # Storage for imagination rollout
        imagined_states = [initial_states]
        imagined_actions = []
        imagined_values = []
        imagined_rewards = []

        current_state = initial_states

        with torch.no_grad():
            for step in range(length):
                # Sample action from policy
                action_dist = self.policy_head(current_state)
                action = action_dist.sample()

                # Predict next state using dynamics
                next_state_dist = self.dynamics_model(current_state, action)
                next_state = next_state_dist.sample()

                # Predict value
                value = self.value_function(current_state)

                # Store
                imagined_actions.append(action)
                imagined_states.append(next_state)
                imagined_values.append(value)

                # Update state
                current_state = next_state

                # Note: In a full implementation, you'd also predict rewards
                # For now, we'll use placeholder rewards
                imagined_rewards.append(torch.zeros(batch_size, device=device))

        return {
            'states': torch.stack(imagined_states[:-1], dim=1),  # (batch, length, state_dim)
            'actions': torch.stack(imagined_actions, dim=1),      # (batch, length, action_dim)
            'values': torch.stack(imagined_values, dim=1),        # (batch, length)
            'rewards': torch.stack(imagined_rewards, dim=1),      # (batch, length)
            'next_states': torch.stack(imagined_states[1:], dim=1) # (batch, length, state_dim)
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute world model learning objectives.

        Args:
            batch: Experience batch containing:
                - observations: Current observations
                - actions: Taken actions
                - rewards: Received rewards
                - next_observations: Next observations
                - dones: Episode termination flags

        Returns:
            Dictionary of loss components
        """
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']

        losses = {}

        # Encode observations
        features = self.encoder(observations)
        states = self.representation_learner.encode(features)

        next_features = self.encoder(next_observations)
        next_states = self.representation_learner.encode(next_features)

        # Representation learning loss (reconstruction)
        if hasattr(self.representation_learner, 'decode'):
            reconstructed = self.representation_learner.decode(states)
            reconstruction_loss = F.mse_loss(reconstructed, features)
            losses['reconstruction_loss'] = reconstruction_loss

        # Additional representation losses
        repr_losses = self.representation_learner.representation_loss(features)
        losses.update(repr_losses)

        # Dynamics loss
        dynamics_losses = self.dynamics_model.dynamics_loss(states, actions, next_states)
        losses.update(dynamics_losses)

        # Value function loss (if we have returns/targets)
        if 'returns' in batch:
            returns = batch['returns']
            value_losses = self.value_function.value_loss(states, returns)
            losses.update(value_losses)

        # Policy loss via imagination rollouts
        imagination_length = self.config.get('imagination_length', 15)
        imagined = self.rollout_imagination(states.detach(), imagination_length)

        # Compute policy loss on imagined trajectories
        # This is a simplified version - real implementations like Dreamer are more complex
        imagined_states = imagined['states']  # (batch, length, state_dim)
        imagined_actions = imagined['actions']  # (batch, length, action_dim)

        # Flatten for loss computation
        flat_states = imagined_states.reshape(-1, imagined_states.shape[-1])
        flat_actions = imagined_actions.reshape(-1, imagined_actions.shape[-1])

        # Policy loss
        policy_dist = self.policy_head(flat_states)
        policy_log_probs = policy_dist.log_prob(flat_actions)

        # Use value estimates as targets (simplified)
        imagined_values = imagined['values']
        flat_values = imagined_values.reshape(-1)

        policy_loss = -(policy_log_probs * flat_values.detach()).mean()
        losses['policy_loss'] = policy_loss

        # Entropy regularization
        entropy_coef = self.config.get('entropy_coef', 0.01)
        if entropy_coef > 0:
            entropy = policy_dist.entropy().mean()
            losses['entropy_loss'] = -entropy_coef * entropy
            losses['entropy'] = entropy

        return losses

    def _save_additional_components(self) -> Dict[str, Any]:
        """Save additional world model components."""
        additional = {
            'dynamics_model': self.dynamics_model.state_dict(),
            'value_function': self.value_function.state_dict()
        }
        if self.planner is not None:
            additional['planner'] = self.planner.state_dict()
        return additional

    def _load_additional_components(self, checkpoint: Dict[str, Any]):
        """Load additional world model components."""
        if 'dynamics_model' in checkpoint:
            self.dynamics_model.load_state_dict(checkpoint['dynamics_model'])
        if 'value_function' in checkpoint:
            self.value_function.load_state_dict(checkpoint['value_function'])
        if 'planner' in checkpoint and self.planner is not None:
            self.planner.load_state_dict(checkpoint['planner'])

    def get_paradigm_info(self) -> Dict[str, Any]:
        """Get information about this paradigm and its components."""
        info = super().get_paradigm_info()
        info['dynamics_info'] = self.dynamics_model.get_dynamics_info()
        info['value_info'] = self.value_function.get_value_info()
        if self.planner is not None:
            info['planner_info'] = self.planner.get_planner_info()
        return info