"""
Component Registry System

This module provides automatic component discovery and registration.
Simply decorate your classes with @register_* decorators and they'll 
be automatically available for experiments.

Example:
    @register_algorithm("ppo")
    class PPO(BaseAlgorithm):
        pass
        
    # Later in config: algorithm.name = "ppo"
"""

from typing import Dict, Any, Type, Callable
import logging

logger = logging.getLogger(__name__)

# Global registries for different component types
ALGORITHM_REGISTRY: Dict[str, Type] = {}
NETWORK_REGISTRY: Dict[str, Type] = {}
ENVIRONMENT_REGISTRY: Dict[str, Type] = {}
BUFFER_REGISTRY: Dict[str, Type] = {}

# New paradigm component registries
ENCODER_REGISTRY: Dict[str, Type] = {}
REPRESENTATION_REGISTRY: Dict[str, Type] = {}
DYNAMICS_REGISTRY: Dict[str, Type] = {}
POLICY_HEAD_REGISTRY: Dict[str, Type] = {}
VALUE_FUNCTION_REGISTRY: Dict[str, Type] = {}
PLANNER_REGISTRY: Dict[str, Type] = {}
PARADIGM_REGISTRY: Dict[str, Type] = {}


def register_algorithm(name: str) -> Callable:
    """Register an algorithm class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in ALGORITHM_REGISTRY:
            logger.warning(f"Algorithm '{name}' already registered, overwriting...")
        ALGORITHM_REGISTRY[name] = cls
        logger.info(f"Registered algorithm: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_network(name: str) -> Callable:
    """Register a network class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in NETWORK_REGISTRY:
            logger.warning(f"Network '{name}' already registered, overwriting...")
        NETWORK_REGISTRY[name] = cls
        logger.info(f"Registered network: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_environment(name: str) -> Callable:
    """Register an environment wrapper class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in ENVIRONMENT_REGISTRY:
            logger.warning(f"Environment '{name}' already registered, overwriting...")
        ENVIRONMENT_REGISTRY[name] = cls
        logger.info(f"Registered environment: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_buffer(name: str) -> Callable:
    """Register a buffer class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in BUFFER_REGISTRY:
            logger.warning(f"Buffer '{name}' already registered, overwriting...")
        BUFFER_REGISTRY[name] = cls
        logger.info(f"Registered buffer: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_encoder(name: str) -> Callable:
    """Register an encoder class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in ENCODER_REGISTRY:
            logger.warning(f"Encoder '{name}' already registered, overwriting...")
        ENCODER_REGISTRY[name] = cls
        logger.info(f"Registered encoder: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_representation_learner(name: str) -> Callable:
    """Register a representation learner class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in REPRESENTATION_REGISTRY:
            logger.warning(f"Representation learner '{name}' already registered, overwriting...")
        REPRESENTATION_REGISTRY[name] = cls
        logger.info(f"Registered representation learner: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_dynamics_model(name: str) -> Callable:
    """Register a dynamics model class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in DYNAMICS_REGISTRY:
            logger.warning(f"Dynamics model '{name}' already registered, overwriting...")
        DYNAMICS_REGISTRY[name] = cls
        logger.info(f"Registered dynamics model: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_policy_head(name: str) -> Callable:
    """Register a policy head class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in POLICY_HEAD_REGISTRY:
            logger.warning(f"Policy head '{name}' already registered, overwriting...")
        POLICY_HEAD_REGISTRY[name] = cls
        logger.info(f"Registered policy head: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_value_function(name: str) -> Callable:
    """Register a value function class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in VALUE_FUNCTION_REGISTRY:
            logger.warning(f"Value function '{name}' already registered, overwriting...")
        VALUE_FUNCTION_REGISTRY[name] = cls
        logger.info(f"Registered value function: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_planner(name: str) -> Callable:
    """Register a planner class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in PLANNER_REGISTRY:
            logger.warning(f"Planner '{name}' already registered, overwriting...")
        PLANNER_REGISTRY[name] = cls
        logger.info(f"Registered planner: {name} -> {cls.__name__}")
        return cls
    return decorator


def register_paradigm(name: str) -> Callable:
    """Register a paradigm class for automatic discovery"""
    def decorator(cls: Type) -> Type:
        if name in PARADIGM_REGISTRY:
            logger.warning(f"Paradigm '{name}' already registered, overwriting...")
        PARADIGM_REGISTRY[name] = cls
        logger.info(f"Registered paradigm: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_algorithm(name: str) -> Type:
    """Get algorithm class by name"""
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Algorithm '{name}' not found in registry. "
                        f"Available: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name]


def get_network(name: str) -> Type:
    """Get network class by name"""
    if name not in NETWORK_REGISTRY:
        raise ValueError(f"Network '{name}' not found in registry. "
                        f"Available: {list(NETWORK_REGISTRY.keys())}")
    return NETWORK_REGISTRY[name]


def get_environment(name: str) -> Type:
    """Get environment class by name"""
    if name not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Environment '{name}' not found in registry. "
                        f"Available: {list(ENVIRONMENT_REGISTRY.keys())}")
    return ENVIRONMENT_REGISTRY[name]


def get_buffer(name: str) -> Type:
    """Get buffer class by name"""
    if name not in BUFFER_REGISTRY:
        raise ValueError(f"Buffer '{name}' not found in registry. "
                        f"Available: {list(BUFFER_REGISTRY.keys())}")
    return BUFFER_REGISTRY[name]


def get_encoder(name: str) -> Type:
    """Get encoder class by name"""
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Encoder '{name}' not found in registry. "
                        f"Available: {list(ENCODER_REGISTRY.keys())}")
    return ENCODER_REGISTRY[name]


def get_representation_learner(name: str) -> Type:
    """Get representation learner class by name"""
    if name not in REPRESENTATION_REGISTRY:
        raise ValueError(f"Representation learner '{name}' not found in registry. "
                        f"Available: {list(REPRESENTATION_REGISTRY.keys())}")
    return REPRESENTATION_REGISTRY[name]


def get_dynamics_model(name: str) -> Type:
    """Get dynamics model class by name"""
    if name not in DYNAMICS_REGISTRY:
        raise ValueError(f"Dynamics model '{name}' not found in registry. "
                        f"Available: {list(DYNAMICS_REGISTRY.keys())}")
    return DYNAMICS_REGISTRY[name]


def get_policy_head(name: str) -> Type:
    """Get policy head class by name"""
    if name not in POLICY_HEAD_REGISTRY:
        raise ValueError(f"Policy head '{name}' not found in registry. "
                        f"Available: {list(POLICY_HEAD_REGISTRY.keys())}")
    return POLICY_HEAD_REGISTRY[name]


def get_value_function(name: str) -> Type:
    """Get value function class by name"""
    if name not in VALUE_FUNCTION_REGISTRY:
        raise ValueError(f"Value function '{name}' not found in registry. "
                        f"Available: {list(VALUE_FUNCTION_REGISTRY.keys())}")
    return VALUE_FUNCTION_REGISTRY[name]


def get_planner(name: str) -> Type:
    """Get planner class by name"""
    if name not in PLANNER_REGISTRY:
        raise ValueError(f"Planner '{name}' not found in registry. "
                        f"Available: {list(PLANNER_REGISTRY.keys())}")
    return PLANNER_REGISTRY[name]


def get_paradigm(name: str) -> Type:
    """Get paradigm class by name"""
    if name not in PARADIGM_REGISTRY:
        raise ValueError(f"Paradigm '{name}' not found in registry. "
                        f"Available: {list(PARADIGM_REGISTRY.keys())}")
    return PARADIGM_REGISTRY[name]


def list_registered_components() -> Dict[str, list]:
    """List all registered components"""
    return {
        'algorithms': list(ALGORITHM_REGISTRY.keys()),
        'networks': list(NETWORK_REGISTRY.keys()),
        'environments': list(ENVIRONMENT_REGISTRY.keys()),
        'buffers': list(BUFFER_REGISTRY.keys()),
        'encoders': list(ENCODER_REGISTRY.keys()),
        'representation_learners': list(REPRESENTATION_REGISTRY.keys()),
        'dynamics_models': list(DYNAMICS_REGISTRY.keys()),
        'policy_heads': list(POLICY_HEAD_REGISTRY.keys()),
        'value_functions': list(VALUE_FUNCTION_REGISTRY.keys()),
        'planners': list(PLANNER_REGISTRY.keys()),
        'paradigms': list(PARADIGM_REGISTRY.keys())
    }


def auto_import_modules():
    """
    Automatically import all modules to trigger registration decorators.
    This should be called at startup to populate registries.
    """
    import importlib
    import pkgutil
    from pathlib import Path
    
    # Get the src directory
    src_path = Path(__file__).parent.parent
    
    # Import all modules in core directories and new component directories
    module_dirs = [
        'algorithms', 'networks', 'environments', 'buffers',
        'components.encoders', 'components.representation_learners',
        'components.dynamics', 'components.policy_heads',
        'components.value_functions', 'components.planners',
        'paradigms'
    ]

    for module_dir in module_dirs:
        # Handle nested module paths (e.g., 'components.encoders')
        module_parts = module_dir.split('.')
        module_path = src_path
        for part in module_parts:
            module_path = module_path / part

        if module_path.exists():
            for finder, name, ispkg in pkgutil.iter_modules([str(module_path)]):
                try:
                    module_name = f"src.{module_dir}.{name}"
                    importlib.import_module(module_name)
                    logger.debug(f"Auto-imported {module_name}")
                except Exception as e:
                    logger.warning(f"Failed to auto-import {module_name}: {e}")


class RegistryMixin:
    """
    Mixin class that provides registry access to components.
    Useful for classes that need to create other components dynamically.
    """

    def create_algorithm(self, name: str, config: Any):
        """Create algorithm instance from registry"""
        cls = get_algorithm(name)
        return cls(config)

    def create_network(self, name: str, config: Any):
        """Create network instance from registry"""
        cls = get_network(name)
        return cls(config)

    def create_environment(self, name: str, config: Any):
        """Create environment instance from registry"""
        cls = get_environment(name)
        return cls(config)

    def create_buffer(self, name: str, config: Any):
        """Create buffer instance from registry"""
        cls = get_buffer(name)
        return cls(config)

    def create_encoder(self, name: str, config: Any):
        """Create encoder instance from registry"""
        cls = get_encoder(name)
        return cls(config)

    def create_representation_learner(self, name: str, config: Any):
        """Create representation learner instance from registry"""
        cls = get_representation_learner(name)
        return cls(config)

    def create_dynamics_model(self, name: str, config: Any):
        """Create dynamics model instance from registry"""
        cls = get_dynamics_model(name)
        return cls(config)

    def create_policy_head(self, name: str, config: Any):
        """Create policy head instance from registry"""
        cls = get_policy_head(name)
        return cls(config)

    def create_value_function(self, name: str, config: Any):
        """Create value function instance from registry"""
        cls = get_value_function(name)
        return cls(config)

    def create_planner(self, name: str, config: Any):
        """Create planner instance from registry"""
        cls = get_planner(name)
        return cls(config)

    def create_paradigm(self, name: str, config: Any):
        """Create paradigm instance from registry"""
        cls = get_paradigm(name)
        return cls(config)