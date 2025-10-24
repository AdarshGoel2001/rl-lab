"""
Component Factory

This module provides factory helpers for building orchestrator components from
configuration dictionaries. It underpins the plug-and-play architecture where
world-model workflows assemble encoders, dynamics models, controllers, and
related modules directly from YAML specs.
"""

import copy
import logging
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, Mapping

from ..utils.registry import (
    get_encoder,
    get_representation_learner,
    get_dynamics_model,
    get_policy_head,
    get_value_function,
    get_planner,
    get_controller,
    get_reward_predictor,
    get_observation_decoder,
    RegistryMixin,
)

logger = logging.getLogger(__name__)


@dataclass
class WorldModelComponentBundle:
    """Structured container for instantiated world-model modules."""

    encoder: Any
    representation_learner: Any
    dynamics_model: Any
    reward_predictor: Optional[Any] = None
    observation_decoder: Optional[Any] = None
    planner: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    specs: Dict[str, Any] = field(default_factory=dict)

    def to(self, device: Union[str, "torch.device"]) -> "WorldModelComponentBundle":
        components = [
            self.encoder,
            self.representation_learner,
            self.dynamics_model,
            self.reward_predictor,
            self.observation_decoder,
            self.planner,
        ]
        for module in components:
            if module is None:
                continue
            mover = getattr(module, "to", None)
            if callable(mover):
                mover(device)
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder,
            "representation_learner": self.representation_learner,
            "dynamics_model": self.dynamics_model,
            "reward_predictor": self.reward_predictor,
            "observation_decoder": self.observation_decoder,
            "planner": self.planner,
            "config": self.config,
            "specs": self.specs,
        }


class ComponentFactory(RegistryMixin):
    """
    Factory for instantiating orchestration-ready components from configuration.

    It wires encoders, RSSM models, controllers, planners, and related modules
    directly from registry-backed YAML specs so workflows can stay declarative.
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
            'controller': get_controller,
            'reward_predictor': get_reward_predictor,
            'observation_decoder': get_observation_decoder,
        }

        if component_type not in registry_map:
            raise ValueError(f"Unknown component type: {component_type}")

        get_func = registry_map[component_type]
        component_class = get_func(component_name)

        logger.info(f"Creating {component_type}: {component_name}")
        return component_class(config)

    @staticmethod
    def create_world_model_components(
        config: Dict[str, Any],
        *,
        device: Optional[str] = None,
        paradigm_config: Optional[Dict[str, Any]] = None,
    ) -> WorldModelComponentBundle:
        """
        Instantiate all modules required for a world-model workflow.
        """

        if not config:
            raise ValueError("World model component configuration is empty.")

        spec = copy.deepcopy(config)
        if hasattr(spec, "to_dict"):
            spec = spec.to_dict()

        # Remove paradigm-specific keys if present.
        spec.pop("paradigm", None)
        internal_paradigm_cfg = spec.pop("paradigm_config", None)
        model_config = dict(paradigm_config or internal_paradigm_cfg or {})

        def _normalise(entry: Any, *, name: str, required: bool = True) -> Optional[Dict[str, Any]]:
            if entry is None:
                if required:
                    raise ValueError(f"World model configuration missing '{name}' entry.")
                return None
            if hasattr(entry, "to_dict"):
                return entry.to_dict()
            if isinstance(entry, dict):
                return dict(entry)
            if hasattr(entry, "__dict__"):
                return dict(entry.__dict__)
            raise TypeError(f"Unsupported configuration type for '{name}': {type(entry)}")

        def _extract(
            name: str,
            *,
            required: bool = True,
        ) -> Optional[tuple[str, Dict[str, Any]]]:
            data = _normalise(spec.get(name), name=name, required=required)
            if data is None:
                return None
            entry_type = data.get("type")
            if not entry_type:
                raise ValueError(f"Component '{name}' is missing a 'type' field.")
            entry_config = dict(data.get("config", {}))
            return entry_type, entry_config

        encoder_type, encoder_cfg = _extract("encoder")  # type: ignore[misc]
        representation_type, representation_cfg = _extract("representation_learner")  # type: ignore[misc]
        dynamics_type, dynamics_cfg = _extract("dynamics_model")  # type: ignore[misc]
        policy_entry = _extract("policy_head", required=False)
        value_entry = _extract("value_function", required=False)

        encoder = ComponentFactory.create_component("encoder", encoder_type, encoder_cfg)
        representation_learner = ComponentFactory.create_component(
            "representation_learner", representation_type, representation_cfg
        )

        dynamics_params = dict(dynamics_cfg)
        specs: Dict[str, Any] = {}
        rep = representation_learner
        if hasattr(rep, "representation_dim"):
            state_dim = rep.representation_dim
            dynamics_params.setdefault("state_dim", state_dim)
            specs["representation_dim"] = state_dim
        if hasattr(rep, "deterministic_state_dim"):
            dynamics_params.setdefault("deterministic_dim", rep.deterministic_state_dim)
            specs["deterministic_state_dim"] = rep.deterministic_state_dim
        if hasattr(rep, "stochastic_state_dim"):
            dynamics_params.setdefault("stochastic_dim", rep.stochastic_state_dim)
            specs["stochastic_state_dim"] = rep.stochastic_state_dim
        if "action_dim" in dynamics_params and "action_dim" not in specs:
            try:
                specs["action_dim"] = int(np.prod(dynamics_params["action_dim"]))
            except Exception:
                specs["action_dim"] = dynamics_params["action_dim"]
        if "discrete_actions" in dynamics_params and "discrete_actions" not in specs:
            specs["discrete_actions"] = bool(dynamics_params["discrete_actions"])

        policy_config: Dict[str, Any] = {}
        temp_policy_head = None
        if policy_entry is not None:
            policy_type, policy_cfg = policy_entry
            policy_config = dict(policy_cfg)
            specs["policy_head"] = {
                "type": policy_type,
                "config": policy_cfg,
            }
            if "representation_dim" not in policy_config and "representation_dim" in specs:
                policy_config["representation_dim"] = specs["representation_dim"]
            # Legacy behaviour: infer action dim for dynamics
            temp_policy_head = ComponentFactory.create_component("policy_head", policy_type, policy_config)
            action_dim = getattr(temp_policy_head, "action_dim", None)
            if action_dim is not None:
                dynamics_params.setdefault("action_dim", action_dim)
                specs["action_dim"] = action_dim
            specs["discrete_actions"] = getattr(temp_policy_head, "discrete_actions", False)

        if temp_policy_head is not None:
            del temp_policy_head

        dynamics_model = ComponentFactory.create_component("dynamics_model", dynamics_type, dynamics_params)
        if value_entry is not None:
            value_type, value_cfg = value_entry
            specs["value_function"] = {
                "type": value_type,
                "config": value_cfg,
            }

        reward_predictor = None
        reward_entry = _extract("reward_predictor", required=False)
        if reward_entry is not None:
            reward_type, reward_params = reward_entry
            if hasattr(rep, "representation_dim"):
                reward_params.setdefault("representation_dim", rep.representation_dim)
            reward_predictor = ComponentFactory.create_component(
                "reward_predictor", reward_type, reward_params
            )

        observation_decoder = None
        decoder_entry = _extract("observation_decoder", required=False)
        if decoder_entry is not None:
            decoder_type, decoder_params = decoder_entry
            if hasattr(rep, "representation_dim"):
                decoder_params.setdefault("representation_dim", rep.representation_dim)
            observation_decoder = ComponentFactory.create_component(
                "observation_decoder", decoder_type, decoder_params
            )

        planner = None
        planner_entry = _extract("planner", required=False)
        if planner_entry is not None:
            planner_type, planner_params = planner_entry
            planner = ComponentFactory.create_component("planner", planner_type, planner_params)

        bundle = WorldModelComponentBundle(
            encoder=encoder,
            representation_learner=representation_learner,
            dynamics_model=dynamics_model,
            reward_predictor=reward_predictor,
            observation_decoder=observation_decoder,
            planner=planner,
            config=model_config,
            specs=specs,
        )

        if device is not None:
            bundle.to(device)

        return bundle

    @staticmethod
    def create_world_model_optimizers(
        components: WorldModelComponentBundle,
        *,
        algorithm_config: Optional[Any] = None,
    ) -> Dict[str, torch.optim.Optimizer]:
        """
        Build optimizers for world-model components using algorithm configuration.
        """

        def _config_value(name: str, default: Any) -> Any:
            if algorithm_config is None:
                return default
            if isinstance(algorithm_config, Mapping):
                return algorithm_config.get(name, default)
            return getattr(algorithm_config, name, default)

        modules = (
            components.encoder,
            components.representation_learner,
            components.reward_predictor,
            components.observation_decoder,
        )

        params: list[torch.nn.Parameter] = []
        for module in modules:
            if module is None:
                continue
            params.extend(list(module.parameters()))

        if not params:
            raise RuntimeError("No parameters available to build world-model optimizer.")

        lr = float(_config_value("world_model_lr", 2e-4))
        betas = _config_value("world_model_betas", (0.9, 0.999))
        weight_decay = float(_config_value("world_model_weight_decay", 0.0))

        world_model_optimizer = torch.optim.Adam(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        return {"world_model": world_model_optimizer}

    @staticmethod
    def create_controller(
        role: str,
        controller_config: Dict[str, Any],
        *,
        device: Optional[str] = None,
        components: Optional[WorldModelComponentBundle] = None,
    ) -> Any:
        """
        Instantiate a controller for a given role.

        Args:
            role: Logical role name (e.g., 'policy', 'planner').
            controller_config: Controller configuration with type/config fields.
            device: Device hint to move underlying module to.
            components: Optional world-model component bundle.

        Returns:
            Controller instance registered under the provided role.
        """
        if not controller_config:
            raise ValueError(f"Controller configuration for role '{role}' is empty.")

        config_copy = dict(controller_config)
        controller_type = config_copy.get('type')
        if not controller_type:
            raise ValueError(f"Controller '{role}' missing 'type' field.")

        params = dict(config_copy.get('config', {}))
        params.setdefault('role', role)
        if device is not None:
            params.setdefault('device', device)
        if components is not None:
            params.setdefault('components', components)
            if 'specs' not in params:
                params['specs'] = getattr(components, "specs", {})

        controller = ComponentFactory.create_component('controller', controller_type, params)
        if hasattr(controller, 'to') and device is not None:
            controller.to(device)
        return controller

    @staticmethod
    def create_controllers(
        controllers_config: Optional[Dict[str, Any]],
        *,
        device: Optional[str] = None,
        components: Optional[WorldModelComponentBundle] = None,
    ) -> Dict[str, Any]:
        """
        Instantiate all controllers declared in configuration.

        Args:
            controllers_config: Mapping of role -> controller specification.
            device: Device hint for controllers.
            components: Optional world-model component bundle.

        Returns:
            Dictionary mapping role to controller instances.
        """
        if not controllers_config:
            return {}

        controllers: Dict[str, Any] = {}
        for role, controller_config in controllers_config.items():
            if controller_config is None:
                continue

            if hasattr(controller_config, 'to_dict'):
                controller_config = controller_config.to_dict()
            elif not isinstance(controller_config, dict):
                raise TypeError(
                    f"Controller config for role '{role}' must be dict-like, "
                    f"got {type(controller_config)}."
                )

            controllers[role] = ComponentFactory.create_controller(
                role,
                controller_config,
                device=device,
                components=components,
            )
        return controllers

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
