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


def list_registered_components() -> Dict[str, list]:
    """List all registered components"""
    return {
        'algorithms': list(ALGORITHM_REGISTRY.keys()),
        'networks': list(NETWORK_REGISTRY.keys()),
        'environments': list(ENVIRONMENT_REGISTRY.keys()),
        'buffers': list(BUFFER_REGISTRY.keys())
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
    
    # Import all modules in algorithms, networks, environments, buffers
    for module_dir in ['algorithms', 'networks', 'environments', 'buffers']:
        module_path = src_path / module_dir
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