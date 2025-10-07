"""World model paradigm trainer built atop the shared trainer infrastructure."""

import logging
from typing import Any, Dict, Optional

import numpy as np

from src.paradigms.factory import ComponentFactory
from src.paradigms.trainer_base import BaseParadigmTrainer
from src.utils.registry import get_paradigm

logger = logging.getLogger(__name__)


class Trainer(BaseParadigmTrainer):
    """Trainer that wires world-model components into the shared trainer loop."""

    def _prepare_component_config(self, raw_config: Dict[str, Any], device: Any) -> Dict[str, Any]:
        config_copy = dict(raw_config.get('config', {}))
        if 'device' not in config_copy:
            config_copy['device'] = device
        return config_copy

    def _create_paradigm(self, obs_space: Any, action_space: Any, device: Any):
        component_config = dict(self.config.components or {})
        if not component_config:
            raise ValueError("Components config is required for world model paradigm")

        action_dim = action_space.n if action_space.discrete else int(np.prod(action_space.shape))

        obs_shape = obs_space.shape
        single_obs_shape = obs_shape
        if isinstance(obs_shape, tuple) and getattr(self.environment, 'is_vectorized', False) and len(obs_shape) > 1:
            single_obs_shape = obs_shape[1:]
            if len(single_obs_shape) == 1:
                single_obs_shape = (single_obs_shape[0],)

        # Encoder
        encoder_entry = component_config.get('encoder')
        if not encoder_entry or 'type' not in encoder_entry:
            raise ValueError("Components config missing required 'encoder.type'")
        encoder_cfg = self._prepare_component_config(encoder_entry, device)
        encoder_cfg.setdefault('input_dim', single_obs_shape)
        encoder = ComponentFactory.create_component('encoder', encoder_entry['type'], encoder_cfg)
        if hasattr(encoder, 'to'):
            encoder.to(device)
        encoder_output_dim = getattr(encoder, 'output_dim', None)
        if encoder_output_dim is None:
            raise ValueError("Encoder must expose 'output_dim' for downstream components")
        logger.info(f"Created encoder component: {encoder_entry['type']}")

        # Representation learner (optional)
        repr_entry = component_config.get('representation_learner')
        if repr_entry:
            if 'type' not in repr_entry:
                raise ValueError("Representation learner entry requires 'type'")
            repr_cfg = self._prepare_component_config(repr_entry, device)
            repr_cfg.setdefault('representation_dim', encoder_output_dim)
            repr_cfg.setdefault('input_dim', encoder_output_dim)
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
        state_dim = getattr(representation_learner, 'representation_dim', encoder_output_dim)

        decoder = None
        decoder_entry = component_config.get('decoder')
        if decoder_entry:
            if 'type' not in decoder_entry:
                raise ValueError("Decoder entry requires 'type'")
            decoder_cfg = self._prepare_component_config(decoder_entry, device)
            decoder_cfg.setdefault('latent_dim', state_dim)
            decoder_cfg.setdefault('output_dim', single_obs_shape)
            decoder = ComponentFactory.create_component('decoder', decoder_entry['type'], decoder_cfg)
            if hasattr(decoder, 'to'):
                decoder.to(device)
            logger.info(f"Created decoder: {decoder_entry['type']}")

        # Dynamics model
        dynamics_entry = component_config.get('dynamics_model')
        if not dynamics_entry or 'type' not in dynamics_entry:
            raise ValueError("World model paradigm requires 'dynamics_model.type'")
        dynamics_cfg = self._prepare_component_config(dynamics_entry, device)
        dynamics_cfg.setdefault('state_dim', state_dim)
        dynamics_cfg.setdefault('action_dim', action_dim)
        dynamics_model = ComponentFactory.create_component('dynamics_model', dynamics_entry['type'], dynamics_cfg)
        if hasattr(dynamics_model, 'to'):
            dynamics_model.to(device)
        logger.info(f"Created dynamics model: {dynamics_entry['type']}")

        # Reward predictor
        reward_entry = component_config.get('reward_predictor')
        if not reward_entry or 'type' not in reward_entry:
            raise ValueError("World model paradigm requires 'reward_predictor.type'")
        reward_cfg = self._prepare_component_config(reward_entry, device)
        reward_cfg.setdefault('state_dim', state_dim)
        reward_predictor = ComponentFactory.create_component('reward_predictor', reward_entry['type'], reward_cfg)
        if hasattr(reward_predictor, 'to'):
            reward_predictor.to(device)
        logger.info(f"Created reward predictor: {reward_entry['type']}")

        # Continue predictor
        continue_entry = component_config.get('continue_predictor')
        if not continue_entry or 'type' not in continue_entry:
            raise ValueError("World model paradigm requires 'continue_predictor.type'")
        continue_cfg = self._prepare_component_config(continue_entry, device)
        continue_cfg.setdefault('state_dim', state_dim)
        continue_predictor = ComponentFactory.create_component('continue_predictor', continue_entry['type'], continue_cfg)
        if hasattr(continue_predictor, 'to'):
            continue_predictor.to(device)
        logger.info(f"Created continue predictor: {continue_entry['type']}")

        # Policy head
        policy_entry = component_config.get('policy_head')
        if not policy_entry or 'type' not in policy_entry:
            raise ValueError("Components config missing required 'policy_head.type'")
        policy_cfg = self._prepare_component_config(policy_entry, device)
        policy_cfg.setdefault('representation_dim', state_dim)
        policy_cfg.setdefault('action_dim', action_dim)
        policy_cfg.setdefault('discrete_actions', action_space.discrete)
        policy_head = ComponentFactory.create_component('policy_head', policy_entry['type'], policy_cfg)
        if hasattr(policy_head, 'to'):
            policy_head.to(device)
        logger.info(f"Created policy head: {policy_entry['type']}")

        # Value function
        value_entry = component_config.get('value_function')
        if not value_entry or 'type' not in value_entry:
            raise ValueError("Components config missing required 'value_function.type'")
        value_cfg = self._prepare_component_config(value_entry, device)
        value_cfg.setdefault('representation_dim', state_dim)
        value_function = ComponentFactory.create_component('value_function', value_entry['type'], value_cfg)
        if hasattr(value_function, 'to'):
            value_function.to(device)
        logger.info(f"Created value function: {value_entry['type']}")

        # Optional planner
        planner = None
        planner_entry = component_config.get('planner')
        if planner_entry:
            if 'type' not in planner_entry:
                raise ValueError("Planner entry requires 'type'")
            planner_cfg = self._prepare_component_config(planner_entry, device)
            planner_cfg.setdefault('state_dim', state_dim)
            planner_cfg.setdefault('action_dim', action_dim)
            planner = ComponentFactory.create_component('planner', planner_entry['type'], planner_cfg)
            if hasattr(planner, 'to'):
                planner.to(device)
            logger.info(f"Created planner: {planner_entry['type']}")

        # Config priority: algorithm section (preferred) < paradigm_config (legacy fallback)
        # This allows users to configure via 'algorithm' section while maintaining backwards compatibility
        paradigm_config = {}

        # First, load from 'algorithm' section (preferred, user-friendly)
        algorithm_obj = getattr(self.config, 'algorithm', None)
        if algorithm_obj:
            # Convert config object to dict - handle both dict and object types
            if hasattr(algorithm_obj, '__dict__'):
                algorithm_config = vars(algorithm_obj)
            elif isinstance(algorithm_obj, dict):
                algorithm_config = algorithm_obj
            else:
                algorithm_config = {}

            if algorithm_config:
                logger.info("Loading paradigm configuration from 'algorithm' section")
                paradigm_config.update(algorithm_config)
        else:
            algorithm_config = {}

        # Then override with 'paradigm_config' section if present (legacy support)
        paradigm_config_obj = getattr(self.config, 'paradigm_config', None)
        if paradigm_config_obj:
            # Convert config object to dict - handle both dict and object types
            if hasattr(paradigm_config_obj, '__dict__'):
                paradigm_config_section = vars(paradigm_config_obj)
            elif isinstance(paradigm_config_obj, dict):
                paradigm_config_section = paradigm_config_obj
            else:
                paradigm_config_section = {}

            if paradigm_config_section:
                if algorithm_config:
                    logger.warning(
                        "Both 'algorithm' and 'paradigm_config' sections found. "
                        "'paradigm_config' values will override 'algorithm' values. "
                        "Consider using only 'algorithm' section as 'paradigm_config' is deprecated."
                    )
                else:
                    logger.info("Loading paradigm configuration from 'paradigm_config' section (legacy)")
                paradigm_config.update(paradigm_config_section)

        if not paradigm_config:
            logger.warning("No algorithm or paradigm_config section found, using defaults only")

        implementation = paradigm_config.pop('implementation', None) or \
            getattr(self.config, 'implementation', None) or 'world_model_mvp'
        paradigm_config.setdefault('device', device)

        # Log critical hyperparameters for verification
        logger.info("=" * 60)
        logger.info("PARADIGM CONFIGURATION (values that will be used):")
        logger.info("=" * 60)
        critical_params = [
            'imagination_horizon', 'world_model_lr', 'actor_lr', 'critic_lr',
            'entropy_coef', 'gamma', 'lambda_return', 'critic_target_standardize',
            'critic_real_return_mix', 'actor_normalize_returns', 'world_model_warmup_steps'
        ]
        for param in critical_params:
            value = paradigm_config.get(param, 'NOT SET (will use default)')
            logger.info(f"  {param:30s} = {value}")
        logger.info("=" * 60)

        paradigm_class = get_paradigm(implementation)

        paradigm = paradigm_class(
            encoder=encoder,
            representation_learner=representation_learner,
            dynamics_model=dynamics_model,
            reward_predictor=reward_predictor,
            continue_predictor=continue_predictor,
            policy_head=policy_head,
            value_function=value_function,
            planner=planner,
            decoder=decoder,
            config=paradigm_config
        )
        logger.info("Instantiated world model paradigm")
        return paradigm


def create_trainer_from_config(config_path: str,
                               experiment_dir: Optional[str] = None,
                               config_overrides: Optional[Dict[str, Any]] = None) -> Trainer:
    """Factory helper to build the world-model trainer from configuration."""
    from src.utils.config import load_config

    config = load_config(config_path, config_overrides)
    trainer = Trainer(config, experiment_dir)
    trainer.save_experiment_config()
    return trainer
