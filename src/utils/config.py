from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""


class ConfigNode:
    """Lightweight namespace with recursive conversion helpers."""

    __slots__ = ("__dict__",)

    def __init__(self, **entries: Any) -> None:
        for key, value in entries.items():
            object.__setattr__(self, key, self._wrap(value))

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, ConfigNode):
            return value
        if isinstance(value, dict):
            return ConfigNode(**value)
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {key: _unwrap(value) for key, value in self.__dict__.items()}

    def update(self, other: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        data = dict(other or {})
        data.update(kwargs)
        for key, value in data.items():
            object.__setattr__(self, key, self._wrap(value))

    def copy(self) -> "ConfigNode":
        return ConfigNode(**self.to_dict())


@dataclass
class Config:
    experiment: ConfigNode
    algorithm: ConfigNode
    environment: ConfigNode
    network: Union[ConfigNode, Dict[str, ConfigNode]]
    buffer: ConfigNode
    training: ConfigNode
    logging: ConfigNode
    components: Dict[str, Any]
    paradigm_config: Dict[str, Any]
    controllers: Dict[str, Any] = field(default_factory=dict)
    data_sources: Dict[str, Any] = field(default_factory=dict)
    evaluation: Optional[ConfigNode] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        data = copy.deepcopy(config_dict)

        experiment = ConfigNode(**data.get("experiment", {}))
        algorithm = ConfigNode(**data.get("algorithm", {}))
        environment = ConfigNode(**data.get("environment", {}))
        network = _build_network(data.get("network", {}))
        buffer = ConfigNode(**data.get("buffer", {}))
        training = ConfigNode(**data.get("training", {}))
        logging_cfg = ConfigNode(**data.get("logging", {}))
        components = copy.deepcopy(data.get("components", {}))
        paradigm_cfg = copy.deepcopy(data.get("paradigm_config", {}))
        controllers = copy.deepcopy(data.get("controllers", {})) or {}
        data_sources = copy.deepcopy(data.get("data_sources", {})) or {}
        evaluation_cfg = data.get("evaluation")
        evaluation = ConfigNode(**evaluation_cfg) if evaluation_cfg else None

        return cls(
            experiment=experiment,
            algorithm=algorithm,
            environment=environment,
            network=network,
            buffer=buffer,
            training=training,
            logging=logging_cfg,
            components=components,
            paradigm_config=paradigm_cfg,
            controllers=controllers,
            data_sources=data_sources,
            evaluation=evaluation,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment.to_dict(),
            "algorithm": self.algorithm.to_dict(),
            "environment": self.environment.to_dict(),
            "network": _network_to_dict(self.network),
            "buffer": self.buffer.to_dict(),
            "training": self.training.to_dict(),
            "logging": self.logging.to_dict(),
            "components": copy.deepcopy(self.components),
            "paradigm_config": copy.deepcopy(self.paradigm_config),
            "controllers": copy.deepcopy(self.controllers),
            "data_sources": copy.deepcopy(self.data_sources),
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }

    def get_hash(self) -> str:
        serialised = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(serialised.encode()).hexdigest()[:8]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.backends.mps.is_available():
            logger.info("Auto-detected device: MPS (Apple Silicon GPU)")
            return "mps"
        if torch.cuda.is_available():
            logger.info("Auto-detected device: CUDA")
            return "cuda"
        logger.info("Auto-detected device: CPU")
        return "cpu"
    except (ImportError, AttributeError):
        logger.warning("PyTorch not available for device detection, falling back to CPU")
        return "cpu"


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Error parsing YAML file {config_path}: {exc}") from exc

    if config_dict is None:
        raise ConfigError(f"Configuration file is empty: {config_path}")

    config_dict = expand_environment_variables(config_dict)

    base_ref = config_dict.pop("base_config", None)
    if base_ref:
        base_config = load_yaml_config(config_path.parent / base_ref)
        config_dict = merge_configs(base_config, config_dict)

    return config_dict


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_config_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    if not overrides:
        return config

    nested = {}
    for dotted_key, value in overrides.items():
        current = nested
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    merged = merge_configs(config.to_dict(), nested)
    return Config.from_dict(merged)


def load_config(
    config_path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None
) -> Config:
    data = load_yaml_config(config_path)
    if overrides:
        data = merge_configs(data, overrides)
    try:
        config = Config.from_dict(data)
    except Exception as exc:
        raise ConfigError(f"Configuration validation failed: {exc}") from exc

    logger.info("Configuration loaded successfully (hash: %s)", config.get_hash())
    return config


def save_config(config: Config, save_path: Union[str, Path]) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, default_flow_style=False, indent=2)

    logger.info("Configuration saved to %s", save_path)


def validate_config_compatibility(config: Config) -> List[str]:
    warnings: List[str] = []

    algo_name = getattr(config.algorithm, "name", None)
    buffer_type = getattr(config.buffer, "type", None)

    if algo_name in {"ppo", "a2c"} and buffer_type != "trajectory":
        warnings.append(
            f"Algorithm {algo_name} typically uses trajectory buffer, but {buffer_type} configured"
        )
    if algo_name in {"dqn", "sac"} and buffer_type == "trajectory":
        warnings.append(
            f"Algorithm {algo_name} typically uses replay buffer, but trajectory buffer configured"
        )

    device = getattr(config.experiment, "device", "cpu")
    if device in {"cuda", "mps"}:
        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                warnings.append("CUDA device requested but not available")
            if device == "mps" and not torch.backends.mps.is_available():
                warnings.append("MPS device requested but not available")
        except ImportError:
            warnings.append("PyTorch not available for device validation")

    if getattr(config.logging, "wandb_enabled", False) and not getattr(
        config.logging, "wandb_project", None
    ):
        warnings.append("WandB enabled but no project specified")

    return warnings


class ConfigManager:
    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[Config] = None
        if self.config_path:
            self.load_config()

    def load_config(
        self,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if config_path:
            self.config_path = Path(config_path)
        if not self.config_path:
            raise ConfigError("No configuration path specified")

        self.config = load_config(self.config_path, overrides)
        for warning in validate_config_compatibility(self.config):
            logger.warning("Config validation warning: %s", warning)

    def save_config(self, save_path: Union[str, Path]) -> None:
        if not self.config:
            raise ConfigError("No configuration loaded")
        save_config(self.config, save_path)

    def get_config_hash(self) -> str:
        if not self.config:
            raise ConfigError("No configuration loaded")
        return self.config.get_hash()

    def update_config(self, updates: Dict[str, Any]) -> None:
        if not self.config:
            raise ConfigError("No configuration loaded")
        merged = merge_configs(self.config.to_dict(), updates)
        self.config = Config.from_dict(merged)


def expand_environment_variables(config: Any) -> Any:
    if isinstance(config, str):
        return _expand_env_string(config)
    if isinstance(config, dict):
        return {key: expand_environment_variables(value) for key, value in config.items()}
    if isinstance(config, list):
        return [expand_environment_variables(value) for value in config]
    return config


def _expand_env_string(value: str) -> str:
    import re

    pattern = re.compile(r"\$\{([^:}]+)(?::([^}]*))?\}")

    def replacer(match: re.Match[str]) -> str:
        var, default = match.group(1), match.group(2) or ""
        return os.environ.get(var, default)

    return pattern.sub(replacer, value)


def _unwrap(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return copy.deepcopy(value)


def _build_network(raw: Dict[str, Any]) -> Union[ConfigNode, Dict[str, ConfigNode]]:
    if not raw:
        return ConfigNode()

    multi_network_keys = {"actor", "critic", "encoder", "decoder"}
    if any(key in raw for key in multi_network_keys):
        return {name: ConfigNode(**cfg) for name, cfg in raw.items()}
    return ConfigNode(**raw)


def _network_to_dict(network: Union[ConfigNode, Dict[str, ConfigNode]]) -> Dict[str, Any]:
    if isinstance(network, dict):
        return {name: node.to_dict() if isinstance(node, ConfigNode) else copy.deepcopy(node) for name, node in network.items()}
    return network.to_dict()
