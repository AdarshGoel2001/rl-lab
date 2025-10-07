"""Model-free paradigm trainer built on the shared trainer infrastructure."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from src.paradigms.factory import ComponentFactory
from src.paradigms.trainer_base import BaseParadigmTrainer
from src.utils.registry import get_algorithm

logger = logging.getLogger(__name__)


class Trainer(BaseParadigmTrainer):
    """Trainer that wires model-free components into the shared trainer loop."""

    def _prepare_component_config(self, raw_config: Dict[str, Any], device: Any) -> Dict[str, Any]:
        config_copy = dict(raw_config.get('config', {}))
        if 'device' not in config_copy:
            config_copy['device'] = device
        return config_copy

    def _create_paradigm(self, obs_space: Any, action_space: Any, device: Any):
        component_config = dict(self.config.components or {})
        if not component_config:
            raise ValueError("Components config is required for model-free paradigm")

        # Encoder
        encoder_entry = component_config.get('encoder')
        if not encoder_entry or 'type' not in encoder_entry:
            raise ValueError("Components config missing required 'encoder.type'")
        encoder_cfg = self._prepare_component_config(encoder_entry, device)
        encoder_cfg.setdefault('input_dim', obs_space.shape)
        encoder = ComponentFactory.create_component('encoder', encoder_entry['type'], encoder_cfg)
        if hasattr(encoder, 'to'):
            encoder.to(device)
        encoder_output_dim = getattr(encoder, 'output_dim', None)
        if encoder_output_dim is None:
            raise ValueError("Encoder must expose 'output_dim' for downstream components")
        logger.info(f"Created encoder component: {encoder_entry['type']}")

        # Representation learner
        repr_entry = component_config.get('representation_learner')
        if repr_entry:
            if 'type' not in repr_entry:
                raise ValueError("Representation learner entry requires 'type'")
            repr_cfg = self._prepare_component_config(repr_entry, device)
            repr_cfg.setdefault('representation_dim', encoder_output_dim)
            representation_learner = ComponentFactory.create_component(
                'representation_learner', repr_entry['type'], repr_cfg
            )
        else:
            logger.info("No representation learner provided - defaulting to identity")
            from src.components.representation_learners.identity import IdentityRepresentationLearner

            representation_learner = IdentityRepresentationLearner({
                'device': device,
                'representation_dim': encoder_output_dim
            })
        if hasattr(representation_learner, 'to'):
            representation_learner.to(device)

        # Policy head
        policy_entry = component_config.get('policy_head')
        if not policy_entry or 'type' not in policy_entry:
            raise ValueError("Components config missing required 'policy_head.type'")
        policy_cfg = self._prepare_component_config(policy_entry, device)
        policy_cfg.setdefault('representation_dim', encoder_output_dim)
        policy_cfg.setdefault(
            'action_dim',
            action_space.n if action_space.discrete else int(np.prod(action_space.shape))
        )
        policy_cfg.setdefault('discrete_actions', action_space.discrete)
        policy_head = ComponentFactory.create_component('policy_head', policy_entry['type'], policy_cfg)
        if hasattr(policy_head, 'to'):
            policy_head.to(device)
        logger.info(f"Created policy head component: {policy_entry['type']}")

        # Value function
        value_entry = component_config.get('value_function')
        if not value_entry or 'type' not in value_entry:
            raise ValueError("Components config missing required 'value_function.type'")
        value_cfg = self._prepare_component_config(value_entry, device)
        value_cfg.setdefault('representation_dim', encoder_output_dim)
        value_function = ComponentFactory.create_component('value_function', value_entry['type'], value_cfg)
        if hasattr(value_function, 'to'):
            value_function.to(device)
        logger.info(f"Created value function component: {value_entry['type']}")

        components = {
            'encoder': encoder,
            'representation_learner': representation_learner,
            'policy_head': policy_head,
            'value_function': value_function,
        }

        algorithm_config = self.config.algorithm.__dict__.copy()
        algorithm_config['observation_space'] = obs_space
        algorithm_config['action_space'] = action_space
        algorithm_config['device'] = device
        algorithm_config['components'] = components

        algorithm_class = get_algorithm(self.config.algorithm.name)
        algorithm = algorithm_class(algorithm_config)
        logger.info(f"Instantiated algorithm/paradigm: {self.config.algorithm.name}")
        return algorithm


def create_trainer_from_config(config_path: str,
                               experiment_dir: Optional[str] = None,
                               config_overrides: Optional[Dict[str, Any]] = None) -> Trainer:
    """Factory helper to build the model-free trainer from configuration."""
    from src.utils.config import load_config

    config = load_config(config_path, config_overrides)
    trainer = Trainer(config, experiment_dir)
    trainer.save_experiment_config()
    return trainer
