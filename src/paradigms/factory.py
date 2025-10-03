"""
Component Factory

This module provides factory methods for creating paradigms and their components
from configuration dictionaries. It enables the plug-and-play architecture
where entire agents can be configured via YAML files.
"""

from typing import Dict, Any, Optional, Union
import logging

from ..utils.registry import (
    get_encoder, get_representation_learner, get_dynamics_model,
    get_policy_head, get_value_function, get_planner, get_paradigm,
    RegistryMixin
)
from .base import BaseParadigm

logger = logging.getLogger(__name__)


class ComponentFactory(RegistryMixin):
    """
    Factory for creating paradigms and components from configuration.

    This factory enables the full modular architecture by providing
    methods to create any component or complete paradigm from config
    dictionaries.
    """

    @staticmethod
    def create_component(component_type: str,
                        component_name: str,
                        config: Dict[str, Any]) -> Any:
        """
        Create any component by type and name.

        Args:
            component_type: Type of component ('encoder', 'policy_head', etc.)
            component_name: Registered name of the component
            config: Component configuration

        Returns:
            Component instance
        """
        registry_map = {
            'encoder': get_encoder,
            'representation_learner': get_representation_learner,
            'dynamics_model': get_dynamics_model,
            'policy_head': get_policy_head,
            'value_function': get_value_function,
            'planner': get_planner,
            'paradigm': get_paradigm
        }

        if component_type not in registry_map:
            raise ValueError(f"Unknown component type: {component_type}")

        get_func = registry_map[component_type]
        component_class = get_func(component_name)

        logger.info(f"Creating {component_type}: {component_name}")
        return component_class(config)

    @staticmethod
    def create_paradigm(config: Dict[str, Any]) -> BaseParadigm:
        """
        Create a complete paradigm from configuration.

        Args:
            config: Paradigm configuration with component specifications

        Returns:
            Configured paradigm instance

        Example config:
        {
            "paradigm": "model_free",
            "encoder": {"type": "simple_mlp", "config": {...}},
            "representation_learner": {"type": "identity", "config": {...}},
            "policy_head": {"type": "gaussian_mlp", "config": {...}},
            "value_function": {"type": "mlp_critic", "config": {...}}
        }
        """
        paradigm_type = config.get('paradigm')
        if not paradigm_type:
            raise ValueError("Config must specify 'paradigm' field")

        logger.info(f"Creating paradigm: {paradigm_type}")

        # Create required components
        components = {}

        # All paradigms need these core components
        required_components = ['encoder', 'representation_learner', 'policy_head']

        for comp_name in required_components:
            comp_config = config.get(comp_name)
            if not comp_config:
                raise ValueError(f"Missing required component: {comp_name}")

            comp_type = comp_config.get('type')
            comp_params = comp_config.get('config', {})

            if not comp_type:
                raise ValueError(f"Component {comp_name} missing 'type' field")

            # Map component names to registry types
            registry_type_map = {
                'encoder': 'encoder',
                'representation_learner': 'representation_learner',
                'policy_head': 'policy_head',
                'value_function': 'value_function',
                'dynamics_model': 'dynamics_model',
                'planner': 'planner'
            }

            registry_type = registry_type_map[comp_name]
            components[comp_name] = ComponentFactory.create_component(
                registry_type, comp_type, comp_params
            )

        # Paradigm-specific components
        if paradigm_type == 'model_free':
            # Model-free needs value function
            value_config = config.get('value_function')
            if not value_config:
                raise ValueError("Model-free paradigm requires 'value_function'")

            components['value_function'] = ComponentFactory.create_component(
                'value_function', value_config['type'], value_config.get('config', {})
            )

            # Import here to avoid circular imports
            from .model_free.paradigm import ModelFreeParadigm
            return ModelFreeParadigm(
                encoder=components['encoder'],
                representation_learner=components['representation_learner'],
                policy_head=components['policy_head'],
                value_function=components['value_function'],
                config=config.get('paradigm_config', {})
            )

        elif paradigm_type == 'world_model':
            # World model needs dynamics model and value function
            dynamics_config = config.get('dynamics_model')
            if not dynamics_config:
                raise ValueError("World model paradigm requires 'dynamics_model'")

            value_config = config.get('value_function')
            if not value_config:
                raise ValueError("World model paradigm requires 'value_function'")

            components['dynamics_model'] = ComponentFactory.create_component(
                'dynamics_model', dynamics_config['type'], dynamics_config.get('config', {})
            )
            components['value_function'] = ComponentFactory.create_component(
                'value_function', value_config['type'], value_config.get('config', {})
            )

            # Planner is optional
            planner = None
            planner_config = config.get('planner')
            if planner_config:
                planner = ComponentFactory.create_component(
                    'planner', planner_config['type'], planner_config.get('config', {})
                )

            # Import here to avoid circular imports
            from .world_model import WorldModelParadigm
            return WorldModelParadigm(
                encoder=components['encoder'],
                representation_learner=components['representation_learner'],
                dynamics_model=components['dynamics_model'],
                policy_head=components['policy_head'],
                value_function=components['value_function'],
                planner=planner,
                config=config.get('paradigm_config', {})
            )

        else:
            # Try to create custom paradigm directly
            try:
                paradigm_class = get_paradigm(paradigm_type)
                # This assumes custom paradigms follow the base constructor
                return paradigm_class(
                    encoder=components['encoder'],
                    representation_learner=components['representation_learner'],
                    policy_head=components['policy_head'],
                    config=config.get('paradigm_config', {})
                )
            except Exception as e:
                raise ValueError(f"Failed to create paradigm '{paradigm_type}': {e}")

    @staticmethod
    def create_encoder(config: Dict[str, Any]) -> Any:
        """Create encoder from config."""
        encoder_type = config.get('type')
        if not encoder_type:
            raise ValueError("Encoder config must specify 'type'")
        return ComponentFactory.create_component('encoder', encoder_type, config.get('config', {}))

    @staticmethod
    def create_representation_learner(config: Dict[str, Any]) -> Any:
        """Create representation learner from config."""
        learner_type = config.get('type')
        if not learner_type:
            raise ValueError("Representation learner config must specify 'type'")
        return ComponentFactory.create_component('representation_learner', learner_type, config.get('config', {}))

    @staticmethod
    def create_policy_head(config: Dict[str, Any]) -> Any:
        """Create policy head from config."""
        head_type = config.get('type')
        if not head_type:
            raise ValueError("Policy head config must specify 'type'")
        return ComponentFactory.create_component('policy_head', head_type, config.get('config', {}))

    @staticmethod
    def validate_paradigm_config(config: Dict[str, Any]) -> bool:
        """
        Validate that a paradigm configuration is complete and valid.

        Args:
            config: Paradigm configuration to validate

        Returns:
            True if valid, raises ValueError if not
        """
        paradigm_type = config.get('paradigm')
        if not paradigm_type:
            raise ValueError("Config must specify 'paradigm' field")

        # Check required components exist
        required = ['encoder', 'representation_learner', 'policy_head']
        for comp in required:
            if comp not in config:
                raise ValueError(f"Missing required component: {comp}")

            comp_config = config[comp]
            if 'type' not in comp_config:
                raise ValueError(f"Component {comp} missing 'type' field")

        # Paradigm-specific validation
        if paradigm_type == 'model_free':
            if 'value_function' not in config:
                raise ValueError("Model-free paradigm requires 'value_function'")

        elif paradigm_type == 'world_model':
            if 'dynamics_model' not in config:
                raise ValueError("World model paradigm requires 'dynamics_model'")
            if 'value_function' not in config:
                raise ValueError("World model paradigm requires 'value_function'")

        logger.info(f"Paradigm config validation passed for: {paradigm_type}")
        return True

    @staticmethod
    def get_paradigm_template(paradigm_type: str) -> Dict[str, Any]:
        """
        Get a template configuration for a paradigm type.

        Args:
            paradigm_type: Type of paradigm ('model_free', 'world_model')

        Returns:
            Template configuration dictionary
        """
        if paradigm_type == 'model_free':
            return {
                "paradigm": "model_free",
                "encoder": {
                    "type": "simple_mlp",
                    "config": {
                        "hidden_dims": [256, 256],
                        "activation": "relu"
                    }
                },
                "representation_learner": {
                    "type": "identity",
                    "config": {}
                },
                "policy_head": {
                    "type": "gaussian_mlp",
                    "config": {
                        "hidden_dims": [256, 256]
                    }
                },
                "value_function": {
                    "type": "mlp_critic",
                    "config": {
                        "hidden_dims": [256, 256]
                    }
                }
            }

        elif paradigm_type == 'world_model':
            return {
                "paradigm": "world_model",
                "encoder": {
                    "type": "simple_mlp",
                    "config": {
                        "hidden_dims": [256, 256]
                    }
                },
                "representation_learner": {
                    "type": "identity",
                    "config": {}
                },
                "dynamics_model": {
                    "type": "deterministic_mlp",
                    "config": {
                        "hidden_dims": [256, 256]
                    }
                },
                "policy_head": {
                    "type": "gaussian_mlp",
                    "config": {
                        "hidden_dims": [256, 256]
                    }
                },
                "value_function": {
                    "type": "mlp_critic",
                    "config": {
                        "hidden_dims": [256, 256]
                    }
                }
            }

        else:
            raise ValueError(f"No template available for paradigm: {paradigm_type}")
