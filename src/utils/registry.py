"""Lightweight component registry utilities (legacy support)."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, Type

logger = logging.getLogger(__name__)


ALGORITHM_REGISTRY: Dict[str, Type] = {}
NETWORK_REGISTRY: Dict[str, Type] = {}
ENVIRONMENT_REGISTRY: Dict[str, Type] = {}
BUFFER_REGISTRY: Dict[str, Type] = {}

ENCODER_REGISTRY: Dict[str, Type] = {}
REPRESENTATION_REGISTRY: Dict[str, Type] = {}
DYNAMICS_REGISTRY: Dict[str, Type] = {}
POLICY_HEAD_REGISTRY: Dict[str, Type] = {}
VALUE_FUNCTION_REGISTRY: Dict[str, Type] = {}
PLANNER_REGISTRY: Dict[str, Type] = {}
CONTROLLER_REGISTRY: Dict[str, Type] = {}
REWARD_PREDICTOR_REGISTRY: Dict[str, Type] = {}
OBSERVATION_DECODER_REGISTRY: Dict[str, Type] = {}
RETURN_COMPUTER_REGISTRY: Dict[str, Type] = {}


def _register(registry: Dict[str, Type], name: str, cls: Type, kind: str) -> Type:
    if name in registry:
        logger.warning("%s '%s' already registered, overwriting...", kind, name)
    registry[name] = cls
    logger.info("Registered %s: %s -> %s", kind, name, cls.__name__)
    return cls


def register_algorithm(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(ALGORITHM_REGISTRY, name, cls, "algorithm")

    return decorator


def register_network(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(NETWORK_REGISTRY, name, cls, "network")

    return decorator


def register_environment(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(ENVIRONMENT_REGISTRY, name, cls, "environment")

    return decorator


def register_buffer(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(BUFFER_REGISTRY, name, cls, "buffer")

    return decorator


def register_encoder(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(ENCODER_REGISTRY, name, cls, "encoder")

    return decorator


def register_representation_learner(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(REPRESENTATION_REGISTRY, name, cls, "representation learner")

    return decorator


def register_dynamics_model(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(DYNAMICS_REGISTRY, name, cls, "dynamics model")

    return decorator


def register_policy_head(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(POLICY_HEAD_REGISTRY, name, cls, "policy head")

    return decorator


def register_value_function(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(VALUE_FUNCTION_REGISTRY, name, cls, "value function")

    return decorator


def register_planner(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(PLANNER_REGISTRY, name, cls, "planner")

    return decorator


def register_controller(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(CONTROLLER_REGISTRY, name, cls, "controller")

    return decorator


def register_reward_predictor(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(REWARD_PREDICTOR_REGISTRY, name, cls, "reward predictor")

    return decorator


def register_observation_decoder(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(OBSERVATION_DECODER_REGISTRY, name, cls, "observation decoder")

    return decorator


def register_return_computer(name: str) -> Callable[[Type], Type]:
    def decorator(cls: Type) -> Type:
        return _register(RETURN_COMPUTER_REGISTRY, name, cls, "return computer")

    return decorator


def _get(registry: Dict[str, Type], name: str, kind: str) -> Type:
    if name not in registry:
        raise ValueError(f"{kind.capitalize()} '{name}' not found in registry. Available: {list(registry.keys())}")
    return registry[name]


def get_algorithm(name: str) -> Type:
    return _get(ALGORITHM_REGISTRY, name, "algorithm")


def get_network(name: str) -> Type:
    return _get(NETWORK_REGISTRY, name, "network")


def get_environment(name: str) -> Type:
    return _get(ENVIRONMENT_REGISTRY, name, "environment")


def get_buffer(name: str) -> Type:
    return _get(BUFFER_REGISTRY, name, "buffer")


def get_encoder(name: str) -> Type:
    return _get(ENCODER_REGISTRY, name, "encoder")


def get_representation_learner(name: str) -> Type:
    return _get(REPRESENTATION_REGISTRY, name, "representation learner")


def get_dynamics_model(name: str) -> Type:
    return _get(DYNAMICS_REGISTRY, name, "dynamics model")


def get_policy_head(name: str) -> Type:
    return _get(POLICY_HEAD_REGISTRY, name, "policy head")


def get_value_function(name: str) -> Type:
    return _get(VALUE_FUNCTION_REGISTRY, name, "value function")


def get_planner(name: str) -> Type:
    return _get(PLANNER_REGISTRY, name, "planner")


def get_controller(name: str) -> Type:
    return _get(CONTROLLER_REGISTRY, name, "controller")


def get_reward_predictor(name: str) -> Type:
    return _get(REWARD_PREDICTOR_REGISTRY, name, "reward predictor")


def get_observation_decoder(name: str) -> Type:
    return _get(OBSERVATION_DECODER_REGISTRY, name, "observation decoder")


def get_return_computer(name: str) -> Type:
    return _get(RETURN_COMPUTER_REGISTRY, name, "return computer")


def list_registered_components() -> Dict[str, list[str]]:
    return {
        "algorithms": list(ALGORITHM_REGISTRY.keys()),
        "networks": list(NETWORK_REGISTRY.keys()),
        "environments": list(ENVIRONMENT_REGISTRY.keys()),
        "buffers": list(BUFFER_REGISTRY.keys()),
        "encoders": list(ENCODER_REGISTRY.keys()),
        "representation_learners": list(REPRESENTATION_REGISTRY.keys()),
        "dynamics_models": list(DYNAMICS_REGISTRY.keys()),
        "policy_heads": list(POLICY_HEAD_REGISTRY.keys()),
        "value_functions": list(VALUE_FUNCTION_REGISTRY.keys()),
        "planners": list(PLANNER_REGISTRY.keys()),
        "controllers": list(CONTROLLER_REGISTRY.keys()),
        "reward_predictors": list(REWARD_PREDICTOR_REGISTRY.keys()),
        "observation_decoders": list(OBSERVATION_DECODER_REGISTRY.keys()),
        "return_computers": list(RETURN_COMPUTER_REGISTRY.keys()),
    }


def auto_import_modules() -> None:
    """Backwards-compatible auto importer (no-op for Hydra workflows)."""

    src_path = Path(__file__).parent.parent
    module_dirs = [
        "algorithms",
        "networks",
        "environments",
        "buffers",
        "components.encoders",
        "components.world_models.representation_learners",
        "components.policy_heads",
        "components.value_functions",
        "components.world_models.controllers",
        "components.world_models.return_computers",
    ]

    for module_dir in module_dirs:
        module_path = src_path / module_dir.replace(".", "/")
        if not module_path.exists():
            continue
        for _, name, _ in pkgutil.iter_modules([str(module_path)]):
            module_name = f"src.{module_dir}.{name}"
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - legacy path
                logger.debug("Auto-import failed for %s: %s", module_name, exc)


class RegistryMixin:
    """Mixin class retained for backward compatibility."""

    def create_algorithm(self, name: str, config: Any):
        cls = get_algorithm(name)
        return cls(config)

    def create_network(self, name: str, config: Any):
        cls = get_network(name)
        return cls(config)

    def create_environment(self, name: str, config: Any):
        cls = get_environment(name)
        return cls(config)
