"""Base building blocks for world-model paradigms."""

from typing import Dict, Any, Optional, Union, Tuple
import inspect
import torch
import torch.nn.functional as F
from torch.distributions import Distribution

from ..base import BaseParadigm
from ...components.encoders.base import BaseEncoder
from ...components.representation_learners.base import BaseRepresentationLearner
from ...components.dynamics.base import BaseDynamicsModel
from ...components.policy_heads.base import BasePolicyHead
from ...components.value_functions.base import BaseValueFunction
from ...components.planners.base import BasePlanner
from ...components.reward_predictors.base import BaseRewardPredictor
from ...components.continue_predictors.base import BaseContinuePredictor
from ...components.decoders.base import BaseDecoder


class BaseWorldModelParadigm(BaseParadigm):
    """Reusable core for world-model based paradigms.

    Subclasses can override hooks (e.g., imagination rollout, loss computation)
    while reusing collection helpers, optimization scaffolding, and checkpoint
    utilities defined here.
    """

    def __init__(self,
                 encoder: BaseEncoder,
                 representation_learner: BaseRepresentationLearner,
                 dynamics_model: BaseDynamicsModel,
                 reward_predictor: BaseRewardPredictor,
                 continue_predictor: BaseContinuePredictor,
                 policy_head: BasePolicyHead,
                 value_function: BaseValueFunction,
                 planner: Optional[BasePlanner] = None,
                 decoder: Optional[BaseDecoder] = None,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(encoder, representation_learner, policy_head, config)

        self.dynamics_model = dynamics_model
        self.reward_predictor = reward_predictor
        self.continue_predictor = continue_predictor
        self.value_function = value_function
        self.planner = planner
        self.decoder = decoder

        # Move additional components to device
        self.dynamics_model.to(self.device)
        self.reward_predictor.to(self.device)
        self.continue_predictor.to(self.device)
        self.value_function.to(self.device)
        if self.planner is not None:
            self.planner.to(self.device)
        if self.decoder is not None:
            self.decoder.to(self.device)

        # Optimizers configured per phase
        def _to_float(value, default):
            if value is None:
                return float(default)
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Expected numeric value, got {value!r}")

        def _to_int(value, default):
            if value is None:
                return int(default)
            if isinstance(value, bool):
                # avoid casting booleans to int unintentionally
                return int(default)
            if isinstance(value, int):
                return value
            try:
                return int(float(value))
            except (TypeError, ValueError):
                raise ValueError(f"Expected integer value, got {value!r}")

        def _to_bool(value, default):
            if value is None:
                return bool(default)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {'1', 'true', 'yes', 'y'}
            return bool(value)

        world_model_lr = _to_float(self.config.get('world_model_lr', 1e-4), 1e-4)
        actor_lr = _to_float(self.config.get('actor_lr', 3e-5), 3e-5)
        critic_lr = _to_float(self.config.get('critic_lr', 3e-5), 3e-5)
        self.config['world_model_lr'] = world_model_lr
        self.config['actor_lr'] = actor_lr
        self.config['critic_lr'] = critic_lr

        # ensure imagination horizon is int
        imagination_horizon = _to_int(self.config.get('imagination_horizon', 15), 15)
        self.config['imagination_horizon'] = imagination_horizon

        # numeric regularizers
        self.config['gamma'] = _to_float(self.config.get('gamma', 0.99), 0.99)
        self.config['lambda_return'] = _to_float(self.config.get('lambda_return', 0.95), 0.95)
        self.config['entropy_coef'] = _to_float(self.config.get('entropy_coef', 0.01), 0.01)
        self.config['max_grad_norm'] = _to_float(self.config.get('max_grad_norm', 1.0), 1.0)
        self.config['decoder_loss_weight'] = _to_float(self.config.get('decoder_loss_weight', 1.0), 1.0)
        if 'critic_target_clip' in self.config and self.config['critic_target_clip'] is not None:
            self.config['critic_target_clip'] = _to_float(self.config['critic_target_clip'], self.config['critic_target_clip'])
        if 'critic_real_return_mix' in self.config and self.config['critic_real_return_mix'] is not None:
            self.config['critic_real_return_mix'] = _to_float(self.config['critic_real_return_mix'], self.config['critic_real_return_mix'])

        if 'world_model_warmup_steps' in self.config and self.config['world_model_warmup_steps'] is not None:
            self.config['world_model_warmup_steps'] = _to_int(self.config['world_model_warmup_steps'], self.config['world_model_warmup_steps'])
        if 'world_model_updates_per_batch' in self.config and self.config['world_model_updates_per_batch'] is not None:
            self.config['world_model_updates_per_batch'] = _to_int(self.config['world_model_updates_per_batch'], self.config['world_model_updates_per_batch'])
        if 'actor_updates_per_batch' in self.config and self.config['actor_updates_per_batch'] is not None:
            self.config['actor_updates_per_batch'] = _to_int(self.config['actor_updates_per_batch'], self.config['actor_updates_per_batch'])
        if 'critic_updates_per_batch' in self.config and self.config['critic_updates_per_batch'] is not None:
            self.config['critic_updates_per_batch'] = _to_int(self.config['critic_updates_per_batch'], self.config['critic_updates_per_batch'])

        self.config['imagination_gumbel_tau'] = _to_float(self.config.get('imagination_gumbel_tau', 1.0), 1.0)
        self.config['imagination_gumbel_hard'] = _to_bool(self.config.get('imagination_gumbel_hard', True), True)

        world_model_params = (
            list(self.encoder.parameters()) +
            list(self.representation_learner.parameters()) +
            list(self.dynamics_model.parameters()) +
            list(self.reward_predictor.parameters()) +
            list(self.continue_predictor.parameters()) +
            (list(self.decoder.parameters()) if self.decoder is not None else [])
        )

        self.world_model_optimizer = torch.optim.Adam(world_model_params, lr=world_model_lr)
        self.actor_optimizer = torch.optim.Adam(self.policy_head.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=critic_lr)

        self.optimizers = {
            'world_model': self.world_model_optimizer,
            'actor': self.actor_optimizer,
            'critic': self.critic_optimizer,
        }

        # Representation loss behaviour
        rep_loss_cfg = self.config.get('representation_loss', {})
        if not isinstance(rep_loss_cfg, dict):
            raise ValueError("'representation_loss' config must be a dictionary if provided")
        self._decoder_representation_loss_mode = rep_loss_cfg.get('decoder_mode', 'accumulate')
        if self._decoder_representation_loss_mode not in {'accumulate', 'replace', 'none'}:
            raise ValueError(
                "Invalid 'decoder_mode' for representation_loss config. "
                "Expected one of {'accumulate', 'replace', 'none'}"
            )
        self._representation_loss_key = rep_loss_cfg.get('key', 'representation_loss')

        # Sequence processing configuration (enables plug-and-play components)
        sequence_cfg_raw = self.config.get('sequence_processing', {})
        if sequence_cfg_raw is None:
            sequence_cfg_raw = {}
        if not isinstance(sequence_cfg_raw, dict):
            raise ValueError("'sequence_processing' config must be a dictionary if provided")

        component_inputs = sequence_cfg_raw.get('component_inputs', {})
        if not isinstance(component_inputs, dict):
            raise ValueError("'component_inputs' inside sequence_processing must be a dictionary if provided")

        context_methods = sequence_cfg_raw.get('context_methods', {})
        if not isinstance(context_methods, dict):
            raise ValueError("'context_methods' inside sequence_processing must be a dictionary if provided")

        self.sequence_processing: Dict[str, Any] = {
            'mode': sequence_cfg_raw.get('mode', 'auto'),
            'default_component_input': sequence_cfg_raw.get('default_component_input', 'flatten'),
            'component_inputs': component_inputs,
            'provide_causal_mask': bool(sequence_cfg_raw.get('provide_causal_mask', False)),
            'causal_mask_includes_future': bool(sequence_cfg_raw.get('causal_mask_includes_future', False)),
            'provide_padding_mask': bool(sequence_cfg_raw.get('provide_padding_mask', False)),
            'padding_mask_source': sequence_cfg_raw.get('padding_mask_source', 'dones'),
            'context_method': sequence_cfg_raw.get('context_method', None),
            'context_methods': context_methods,
            'context_attribute': sequence_cfg_raw.get('context_attribute', 'sequence_context'),
            'attach_context_attribute': sequence_cfg_raw.get('attach_context_attribute', True),
            'dynamics_loss_mode': sequence_cfg_raw.get('dynamics_loss_mode', 'auto'),
            'reward_loss_mode': sequence_cfg_raw.get('reward_loss_mode', 'auto'),
            'continue_loss_mode': sequence_cfg_raw.get('continue_loss_mode', 'auto'),
            'imagination': sequence_cfg_raw.get('imagination', {}),
            'include_flat_in_context': bool(sequence_cfg_raw.get('include_flat_in_context', False)),
        }

        if self.sequence_processing['mode'] not in {'auto', 'always', 'never'}:
            raise ValueError("sequence_processing.mode must be one of {'auto', 'always', 'never'}")

        if self.sequence_processing['default_component_input'] not in {'flatten', 'sequence'}:
            raise ValueError("sequence_processing.default_component_input must be 'flatten' or 'sequence'")

        imagination_cfg = self.sequence_processing['imagination']
        if imagination_cfg and not isinstance(imagination_cfg, dict):
            raise ValueError("sequence_processing.imagination must be a dictionary if provided")
        imagination_cfg = imagination_cfg or {}
        imagination_cfg.setdefault('enabled', False)
        imagination_cfg.setdefault('carry_history', True)
        imagination_cfg.setdefault('context_key', 'sequence_context')
        self.sequence_processing['imagination'] = imagination_cfg

        # Warm-up and scheduling helpers
        self.env_steps_processed = 0
        self._critic_real_mix_override: Optional[float] = None
        self._actor_normalize_returns: Optional[bool] = None

        # Log final configuration after all defaults and conversions applied
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=" * 60)
        logger.info("FINAL PARADIGM HYPERPARAMETERS (after defaults applied):")
        logger.info("=" * 60)
        logger.info(f"  {'imagination_horizon':30s} = {self.config['imagination_horizon']} (int)")
        logger.info(f"  {'world_model_lr':30s} = {self.config['world_model_lr']:.2e}")
        logger.info(f"  {'actor_lr':30s} = {self.config['actor_lr']:.2e}")
        logger.info(f"  {'critic_lr':30s} = {self.config['critic_lr']:.2e}")
        logger.info(f"  {'entropy_coef':30s} = {self.config['entropy_coef']:.2e}")
        logger.info(f"  {'gamma':30s} = {self.config['gamma']}")
        logger.info(f"  {'lambda_return':30s} = {self.config['lambda_return']}")
        logger.info(f"  {'max_grad_norm':30s} = {self.config['max_grad_norm']}")
        logger.info(f"  {'critic_target_standardize':30s} = {self.config.get('critic_target_standardize', 'NOT SET')}")
        logger.info(f"  {'critic_real_return_mix':30s} = {self.config.get('critic_real_return_mix', 'NOT SET')}")
        logger.info(f"  {'critic_target_clip':30s} = {self.config.get('critic_target_clip', 'NOT SET')}")
        logger.info(f"  {'actor_normalize_returns':30s} = {self.config.get('actor_normalize_returns', 'NOT SET')}")
        logger.info(f"  {'world_model_warmup_steps':30s} = {self.config.get('world_model_warmup_steps', 'NOT SET')}")
        logger.info(f"  {'world_model_updates_per_batch':30s} = {self.config.get('world_model_updates_per_batch', 'NOT SET')}")
        logger.info(f"  {'actor_updates_per_batch':30s} = {self.config.get('actor_updates_per_batch', 'NOT SET')}")
        logger.info(f"  {'critic_updates_per_batch':30s} = {self.config.get('critic_updates_per_batch', 'NOT SET')}")
        logger.info("=" * 60)

        # Validate critical hyperparameters
        if self.config['imagination_horizon'] < 1:
            logger.error(f"imagination_horizon must be >= 1, got {self.config['imagination_horizon']}")
            raise ValueError(f"Invalid imagination_horizon: {self.config['imagination_horizon']}")

        if self.config['imagination_horizon'] == 1:
            logger.warning(
                "⚠️  imagination_horizon=1 detected! This severely limits world model planning. "
                "Consider using horizon >= 10 for meaningful multi-step imagination. "
                "With horizon=1, the world model provides minimal benefit over model-free methods."
            )
        elif self.config['imagination_horizon'] < 5:
            logger.warning(
                f"⚠️  imagination_horizon={self.config['imagination_horizon']} is quite low. "
                f"Consider using horizon >= 10 for better credit assignment."
            )

        if self.config['entropy_coef'] < 1e-5:
            logger.warning(
                f"⚠️  entropy_coef={self.config['entropy_coef']:.2e} is very low. "
                f"This may cause premature convergence to suboptimal deterministic policies."
            )

    # ------------------------------------------------------------------
    # Sequence-aware helpers
    # ------------------------------------------------------------------

    def _get_component_input_format(self, component_name: str, default: str = 'flatten') -> str:
        component_cfg = self.sequence_processing.get('component_inputs', {})
        return component_cfg.get(component_name, self.sequence_processing.get('default_component_input', default))

    def _select_component_input(self,
                                component_name: str,
                                flat_value: torch.Tensor,
                                sequence_value: Optional[torch.Tensor]) -> torch.Tensor:
        format_choice = self._get_component_input_format(component_name)
        if format_choice == 'sequence' and sequence_value is not None:
            return sequence_value
        return flat_value

    @staticmethod
    def _invoke_with_optional_kwargs(func, *args, **optional_kwargs):
        if not optional_kwargs:
            return func(*args)

        signature_target = func
        if hasattr(func, 'forward') and callable(getattr(func, 'forward')):
            signature_target = getattr(func, 'forward')

        signature = inspect.signature(signature_target)
        parameter_values = signature.parameters.values()
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameter_values)
        if accepts_kwargs:
            return func(*args, **optional_kwargs)

        accepted_names = {
            name for name, param in signature.parameters.items()
            if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        }
        filtered_kwargs = {key: value for key, value in optional_kwargs.items() if key in accepted_names}
        if filtered_kwargs:
            return func(*args, **filtered_kwargs)
        return func(*args)

    def _attach_sequence_context(self, context: Optional[Dict[str, torch.Tensor]]):
        if not context:
            return

        context_attr = self.sequence_processing.get('context_attribute')
        attach_attr = bool(self.sequence_processing.get('attach_context_attribute', True)) and context_attr
        context_method_default = self.sequence_processing.get('context_method')
        context_methods = self.sequence_processing.get('context_methods', {})

        components = {
            'encoder': self.encoder,
            'representation_learner': self.representation_learner,
            'dynamics_model': self.dynamics_model,
            'reward_predictor': self.reward_predictor,
            'continue_predictor': self.continue_predictor,
            'policy_head': self.policy_head,
            'value_function': self.value_function,
            'planner': self.planner,
            'decoder': self.decoder,
        }

        for name, component in components.items():
            if component is None:
                continue

            method_name = context_methods.get(name, context_method_default)
            if method_name and hasattr(component, method_name):
                getattr(component, method_name)(context)

            if attach_attr and hasattr(component, '__setattr__'):
                setattr(component, context_attr, context)

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    def forward(self,
                observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
                context: Optional[Dict[str, Any]] = None) -> Distribution:
        features = self.encoder(observations)
        states = self.representation_learner.encode(features)

        if self.planner is not None:
            return self.planner.plan(states, self.dynamics_model, self.value_function)
        return self.policy_head(states, context)

    def get_value(self,
                  observations: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        features = self.encoder(observations)
        states = self.representation_learner.encode(features)
        return self.value_function(states)

    def get_action_and_value(self,
                             observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
                             context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_tensor = self._to_device(observations)

        action_dist = self.forward(obs_tensor, context)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        if isinstance(log_prob, torch.Tensor) and log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)

        features = self.encoder(obs_tensor)
        states = self.representation_learner.encode(features)
        value = self.value_function(states)
        if isinstance(value, torch.Tensor) and value.dim() > 1:
            value = value.squeeze(-1)

        return action, log_prob, value

    def train(self):
        super().train()
        for component in (self.reward_predictor, self.continue_predictor, self.planner):
            if component is not None and hasattr(component, 'train'):
                component.train()

    def eval(self):
        super().eval()
        for component in (self.reward_predictor, self.continue_predictor, self.planner):
            if component is not None and hasattr(component, 'eval'):
                component.eval()

    # ------------------------------------------------------------------
    # Core learning utilities
    # ------------------------------------------------------------------

    def rollout_imagination(self,
                            initial_states: torch.Tensor,
                            length: int,
                            with_grad: bool = False,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        batch_size = initial_states.shape[0]
        _ = batch_size  # kept for readability in overrides

        imagined_states = [initial_states]
        imagined_actions = []
        imagined_action_log_probs = []
        imagined_values = []
        imagined_rewards = []
        imagined_continues = []

        current_state = initial_states

        imagination_cfg = self.sequence_processing.get('imagination', {})
        imagination_enabled = bool(imagination_cfg.get('enabled', False))
        carry_history = bool(imagination_cfg.get('carry_history', True))
        context_key = imagination_cfg.get('context_key', 'sequence_context')

        base_context = imagination_cfg.get('base_context', {}) if imagination_enabled else {}
        if base_context is None:
            base_context = {}

        rollout_context: Optional[Dict[str, Any]]
        if imagination_enabled:
            rollout_context = dict(base_context)
            if context:
                rollout_context.update(context)
        else:
            rollout_context = dict(context) if context else None

        rollout_history = [] if imagination_enabled and carry_history else None

        context_manager = torch.no_grad() if not with_grad else torch.enable_grad()
        action_dim = self.policy_head.action_dim

        with context_manager:
            for step in range(length):
                policy_context = context
                if rollout_context is not None:
                    if imagination_enabled:
                        rollout_context['step'] = step
                        if carry_history and rollout_history is not None:
                            rollout_history.append(current_state)
                            rollout_context['state_history'] = torch.stack(rollout_history, dim=1)
                    key = context_key or 'sequence_context'
                    policy_context = dict(policy_context) if policy_context else {}
                    policy_context[key] = rollout_context

                action_dist = self._invoke_with_optional_kwargs(
                    self.policy_head,
                    current_state,
                    context=policy_context
                )

                if self.policy_head.discrete_actions:
                    if with_grad:
                        logits = getattr(action_dist, 'logits', None)
                        if logits is None:
                            probs = action_dist.probs
                            logits = (probs + 1e-8).log()
                        tau = float(self.config.get('imagination_gumbel_tau', 1.0) or 1.0)
                        hard = bool(self.config.get('imagination_gumbel_hard', True))
                        action_onehot = F.gumbel_softmax(logits, tau=tau, hard=hard)
                        action_encoded = action_onehot
                        action_indices = action_onehot.argmax(dim=-1)
                        action_log_prob = action_dist.log_prob(action_indices)
                        action = action_indices
                    else:
                        action = action_dist.sample()
                        action_log_prob = action_dist.log_prob(action)
                        action_encoded = F.one_hot(action.long(), num_classes=action_dim).float()
                else:
                    if with_grad and getattr(action_dist, 'has_rsample', False):
                        action = action_dist.rsample()
                    else:
                        action = action_dist.sample()
                    action_log_prob = action_dist.log_prob(action)
                    action_encoded = action

                dynamics_kwargs: Dict[str, Any] = {}
                if rollout_context is not None:
                    dynamics_kwargs['context'] = rollout_context
                next_state_dist = self._invoke_with_optional_kwargs(
                    self.dynamics_model,
                    current_state,
                    action_encoded,
                    **dynamics_kwargs
                )
                next_state = next_state_dist.sample() if not with_grad else next_state_dist.mean

                reward_kwargs: Dict[str, Any] = {}
                continue_kwargs: Dict[str, Any] = {}
                if rollout_context is not None:
                    reward_kwargs['context'] = rollout_context
                    continue_kwargs['context'] = rollout_context

                reward_dist = self._invoke_with_optional_kwargs(
                    self.reward_predictor,
                    current_state,
                    **reward_kwargs
                )
                reward = reward_dist.mean

                continue_dist = self._invoke_with_optional_kwargs(
                    self.continue_predictor,
                    current_state,
                    **continue_kwargs
                )
                continue_prob = continue_dist.probs

                value = self.value_function(current_state)
                if value.dim() > 1:
                    value = value.squeeze(-1)

                imagined_actions.append(action)
                imagined_action_log_probs.append(action_log_prob)
                imagined_states.append(next_state)
                imagined_values.append(value)
                imagined_rewards.append(reward)
                imagined_continues.append(continue_prob)

                if rollout_context is not None:
                    rollout_context['last_action'] = action
                    rollout_context['last_state'] = next_state

                current_state = next_state

        return {
            'states': torch.stack(imagined_states[:-1], dim=1),
            'actions': torch.stack(imagined_actions, dim=1),
            'action_log_probs': torch.stack(imagined_action_log_probs, dim=1),
            'values': torch.stack(imagined_values, dim=1),
            'rewards': torch.stack(imagined_rewards, dim=1),
            'continues': torch.stack(imagined_continues, dim=1),
            'next_states': torch.stack(imagined_states[1:], dim=1),
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']

        losses: Dict[str, torch.Tensor] = {}

        sequence_length = 1
        if '_sequence_length' in batch:
            sequence_length = int(batch['_sequence_length'].item())

        if not isinstance(observations, torch.Tensor):
            raise TypeError("BaseWorldModelParadigm currently expects tensor observations in compute_loss")

        auto_sequence_mode = observations.dim() > 2 and sequence_length > 1
        mode_override = self.sequence_processing.get('mode', 'auto')
        if mode_override == 'always':
            sequence_mode = True
        elif mode_override == 'never':
            sequence_mode = False
        else:
            sequence_mode = auto_sequence_mode

        if sequence_mode:
            batch_size = observations.shape[0]
            obs_shape = observations.shape[2:]
            observations_flat = observations.reshape(batch_size * sequence_length, *obs_shape)
            next_observations_flat = next_observations.reshape(batch_size * sequence_length, *obs_shape)
        else:
            observations_flat = observations
            next_observations_flat = next_observations
            batch_size = observations.shape[0]

        features = self.encoder(observations_flat)
        features_seq = None
        if sequence_mode and isinstance(features, torch.Tensor):
            feature_dim = features.shape[-1]
            features_seq = features.reshape(batch_size, sequence_length, feature_dim)

        provide_context = sequence_mode or self.sequence_processing.get('include_flat_in_context', False)
        sequence_context: Optional[Dict[str, Any]] = None
        if provide_context:
            sequence_context = {
                'sequence_length': sequence_length,
                'batch_size': batch_size,
            }

        representation_input = self._select_component_input('representation_learner', features, features_seq)
        states_encoded = self._invoke_with_optional_kwargs(
            self.representation_learner.encode,
            representation_input,
            sequence_context=sequence_context
        )

        if sequence_mode and isinstance(states_encoded, torch.Tensor) and states_encoded.dim() == 3:
            states_seq = states_encoded
            state_dim = states_seq.shape[-1]
            states_flat = states_seq.reshape(batch_size * sequence_length, state_dim)
        else:
            states_flat = states_encoded
            if sequence_mode and isinstance(states_flat, torch.Tensor):
                state_dim = states_flat.shape[-1]
                states_seq = states_flat.reshape(batch_size, sequence_length, state_dim)
            else:
                states_seq = None
                state_dim = states_flat.shape[-1] if isinstance(states_flat, torch.Tensor) else None

        next_features = self.encoder(next_observations_flat)
        next_features_seq = None
        if sequence_mode and isinstance(next_features, torch.Tensor):
            next_features_seq = next_features.reshape(batch_size, sequence_length, next_features.shape[-1])

        next_representation_input = self._select_component_input('representation_learner', next_features, next_features_seq)
        next_states_encoded = self._invoke_with_optional_kwargs(
            self.representation_learner.encode,
            next_representation_input,
            sequence_context=sequence_context
        )

        if sequence_mode and isinstance(next_states_encoded, torch.Tensor) and next_states_encoded.dim() == 3:
            next_states_seq = next_states_encoded
            next_states_flat = next_states_seq.reshape(batch_size * sequence_length, next_states_seq.shape[-1])
        else:
            next_states_flat = next_states_encoded
            if sequence_mode and isinstance(next_states_flat, torch.Tensor):
                next_states_seq = next_states_flat.reshape(batch_size, sequence_length, next_states_flat.shape[-1])
            else:
                next_states_seq = None

        actions_tensor = actions
        if sequence_mode:
            if actions_tensor.dim() == 3 and actions_tensor.shape[-1] == 1:
                actions_flat = actions_tensor.reshape(-1)
            elif actions_tensor.dim() == 2:
                actions_flat = actions_tensor.reshape(-1)
            else:
                actions_flat = actions_tensor.reshape(actions_tensor.shape[0] * actions_tensor.shape[1], -1)
        else:
            actions_flat = actions_tensor

        action_dim = self.config.get('action_dim', self.policy_head.action_dim)
        if actions_flat.dim() == 2 and actions_flat.shape[-1] == 1:
            actions_flat = actions_flat.squeeze(-1)

        if actions_flat.dim() == 1:
            actions_encoded_flat = F.one_hot(actions_flat.long(), num_classes=action_dim).float()
        else:
            actions_encoded_flat = actions_flat

        if sequence_mode:
            actions_encoded_seq = actions_encoded_flat.reshape(actions.shape[0], sequence_length, -1)
        else:
            actions_encoded_seq = None

        if sequence_context is not None:
            if sequence_mode and actions_encoded_seq is not None:
                sequence_context['actions'] = actions_encoded_seq
            if self.sequence_processing.get('include_flat_in_context', False):
                sequence_context['actions_flat'] = actions_encoded_flat

        rewards_tensor = rewards
        dones_tensor = dones.float()
        rewards_seq = None
        dones_seq = None
        continues_seq = None
        if sequence_mode:
            rewards_flat = rewards_tensor.reshape(-1)
            dones_flat = dones_tensor.reshape(-1)
            rewards_seq = rewards_tensor.reshape(batch_size, sequence_length, -1)
            if rewards_seq.shape[-1] == 1:
                rewards_seq = rewards_seq.squeeze(-1)
            dones_seq = dones_tensor.reshape(batch_size, sequence_length)
            continues_seq = (1.0 - dones_seq)
        else:
            rewards_flat = rewards_tensor
            if rewards_flat.dim() > 1:
                rewards_flat = rewards_flat.squeeze(-1)
            dones_flat = dones_tensor
            if dones_flat.dim() > 1:
                dones_flat = dones_flat.squeeze(-1)

        if sequence_context is not None:
            if sequence_mode and rewards_seq is not None:
                sequence_context['rewards'] = rewards_seq
            elif self.sequence_processing.get('include_flat_in_context', False):
                sequence_context['rewards_flat'] = rewards_flat

            if sequence_mode and dones_seq is not None:
                sequence_context['dones'] = dones_seq
                sequence_context['continues'] = continues_seq
            else:
                sequence_context['dones'] = dones_flat
                sequence_context['continues'] = 1.0 - dones_flat

            if self.sequence_processing.get('include_flat_in_context', False):
                sequence_context['features_flat'] = features
                sequence_context['next_features_flat'] = next_features
                sequence_context['states_flat'] = states_flat
                sequence_context['next_states_flat'] = next_states_flat

            if sequence_mode:
                if features_seq is not None:
                    sequence_context['features'] = features_seq
                if next_features_seq is not None:
                    sequence_context['next_features'] = next_features_seq
                if states_seq is not None:
                    sequence_context['states'] = states_seq
                if next_states_seq is not None:
                    sequence_context['next_states'] = next_states_seq

                device = states_flat.device if isinstance(states_flat, torch.Tensor) else None
                if self.sequence_processing.get('provide_causal_mask', False):
                    include_future = self.sequence_processing.get('causal_mask_includes_future', False)
                    diagonal = 0 if include_future else 1
                    causal_mask = torch.triu(
                        torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
                        diagonal=diagonal
                    )
                    sequence_context['causal_mask'] = causal_mask.unsqueeze(0)

                if self.sequence_processing.get('provide_padding_mask', False):
                    padding_source = self.sequence_processing.get('padding_mask_source', 'dones')
                    if padding_source not in {'dones', 'continues'}:
                        raise ValueError("sequence_processing.padding_mask_source must be 'dones' or 'continues'")
                    if padding_source == 'continues' and continues_seq is not None:
                        padding_mask = continues_seq <= 0.0
                    else:
                        padding_mask = dones_seq.bool() if dones_seq is not None else (dones_flat > 0.0)
                    sequence_context['padding_mask'] = padding_mask

            elif self.sequence_processing.get('provide_padding_mask', False):
                padding_mask = dones_flat > 0.0
                sequence_context['padding_mask'] = padding_mask

            self._attach_sequence_context(sequence_context)

        if hasattr(self.representation_learner, 'representation_loss'):
            repr_input = self._select_component_input('representation_learner', features, features_seq)
            repr_losses = self._invoke_with_optional_kwargs(
                self.representation_learner.representation_loss,
                repr_input,
                sequence_context=sequence_context
            )
            if repr_losses:
                losses.update(repr_losses)
                if (self._representation_loss_key != 'representation_loss' and
                        'representation_loss' in repr_losses and
                        self._representation_loss_key not in losses):
                    losses[self._representation_loss_key] = repr_losses['representation_loss']

        states_for_dynamics = self._select_component_input('dynamics_model', states_flat, states_seq)
        actions_for_dynamics = self._select_component_input('dynamics_model', actions_encoded_flat, actions_encoded_seq)
        next_states_for_dynamics = self._select_component_input('dynamics_model', next_states_flat, next_states_seq)

        dynamics_losses = self._invoke_with_optional_kwargs(
            self.dynamics_model.dynamics_loss,
            states_for_dynamics,
            actions_for_dynamics,
            next_states_for_dynamics,
            sequence_context=sequence_context
        )
        losses.update(dynamics_losses)

        reward_states_input = self._select_component_input('reward_predictor', states_flat, states_seq)
        reward_input_format = self._get_component_input_format('reward_predictor')
        rewards_for_loss = rewards_flat
        if reward_input_format == 'sequence' and rewards_seq is not None:
            rewards_for_loss = rewards_seq

        reward_losses = self._invoke_with_optional_kwargs(
            self.reward_predictor.reward_loss,
            reward_states_input,
            rewards_for_loss,
            sequence_context=sequence_context
        )
        losses.update(reward_losses)

        continues_flat = (1.0 - dones_flat)
        continue_states_input = self._select_component_input('continue_predictor', states_flat, states_seq)
        continue_input_format = self._get_component_input_format('continue_predictor')
        continues_for_loss = continues_flat
        if continue_input_format == 'sequence' and continues_seq is not None:
            continues_for_loss = continues_seq

        continue_losses = self._invoke_with_optional_kwargs(
            self.continue_predictor.continue_loss,
            continue_states_input,
            continues_for_loss,
            sequence_context=sequence_context
        )
        losses.update(continue_losses)

        world_model_loss = (
            dynamics_losses.get('dynamics_loss', torch.tensor(0.0, device=states_flat.device)) +
            reward_losses.get('reward_loss', torch.tensor(0.0, device=states_flat.device)) +
            continue_losses.get('continue_loss', torch.tensor(0.0, device=states_flat.device))
        )

        decoder_weight = float(self.config.get('decoder_loss_weight', 1.0) or 0.0)
        decoder_recon_loss = torch.tensor(0.0, device=states_flat.device)
        if decoder_weight != 0.0 and self.decoder is not None:
            reconstructed_obs = self.decoder(states_flat)
            obs_targets = observations_flat.reshape(reconstructed_obs.shape)
            decoder_recon_loss = F.mse_loss(reconstructed_obs, obs_targets)
            world_model_loss = world_model_loss + decoder_weight * decoder_recon_loss

        losses['decoder_reconstruction_loss'] = decoder_recon_loss

        if self._decoder_representation_loss_mode != 'none':
            rep_key = self._representation_loss_key
            if rep_key not in losses:
                losses[rep_key] = decoder_recon_loss
            elif self._decoder_representation_loss_mode == 'replace':
                losses[rep_key] = decoder_recon_loss
            elif self._decoder_representation_loss_mode == 'accumulate':
                losses[rep_key] = losses[rep_key] + decoder_recon_loss

        multi_step_weight = float(self.config.get('multi_step_loss_weight', 0.0) or 0.0)
        multi_step_loss = torch.tensor(0.0, device=states_flat.device)
        if sequence_mode and sequence_length > 1 and multi_step_weight > 0.0 and states_seq is not None and actions_encoded_seq is not None:
            continues_multi = continues_seq
            if continues_multi is None:
                continues_multi = (1.0 - dones_tensor.reshape(observations.shape[0], sequence_length))

            predicted_state = states_seq[:, 0]
            total_weight = torch.tensor(0.0, device=states_flat.device)
            total_loss = torch.tensor(0.0, device=states_flat.device)
            mask = torch.ones(observations.shape[0], device=states_flat.device)

            for t in range(sequence_length - 1):
                dist = self._invoke_with_optional_kwargs(
                    self.dynamics_model,
                    predicted_state,
                    actions_encoded_seq[:, t],
                    sequence_context=sequence_context
                )
                predicted_mean = dist.mean
                target_state = states_seq[:, t + 1].detach()

                step_mse = F.mse_loss(predicted_mean, target_state, reduction='none').mean(dim=-1)
                step_weight = mask

                total_loss = total_loss + (step_mse * step_weight).sum()
                total_weight = total_weight + step_weight.sum()

                continues_t = continues_multi[:, t]
                predicted_state = torch.where(continues_t.unsqueeze(-1) > 0.0, predicted_mean, target_state)
                mask = mask * continues_t

                if mask.sum().item() <= 0:
                    break

            if total_weight.item() > 0:
                multi_step_loss = total_loss / total_weight
                world_model_loss = world_model_loss + multi_step_weight * multi_step_loss

        losses['multi_step_state_loss'] = multi_step_loss
        losses['world_model_loss'] = world_model_loss

        imagination_length = self.config.get('imagination_horizon', 15)
        imagination_context = None
        if sequence_context is not None and self.sequence_processing.get('imagination', {}).get('enabled', False):
            imagination_context = sequence_context
        imagined = self.rollout_imagination(
            states_flat.detach(),
            imagination_length,
            with_grad=True,
            context=imagination_context
        )

        imagined_rewards = imagined['rewards']
        imagined_values = imagined['values']
        imagined_continues = imagined['continues']

        gamma = self.config.get('gamma', 0.99)
        lambda_ = self.config.get('lambda_return', 0.95)
        returns = self._compute_lambda_returns(imagined_rewards, imagined_values, imagined_continues, gamma, lambda_)

        critic_target_clip = self.config.get('critic_target_clip')
        critic_target_standardize = self.config.get('critic_target_standardize', False)
        critic_real_mix = self._critic_real_mix_override
        if critic_real_mix is None:
            critic_real_mix = self.config.get('critic_real_return_mix', 0.0)
        critic_real_mix = float(critic_real_mix or 0.0)

        targets = returns
        if critic_target_clip is not None and critic_target_clip > 0:
            targets = targets.clamp(-critic_target_clip, critic_target_clip)

        with torch.no_grad():
            next_state_values = self.value_function(next_states_flat)
            if next_state_values.dim() > 1:
                next_state_values = next_state_values.squeeze(-1)

            real_rewards = rewards_flat
            real_targets_base = real_rewards + gamma * (1.0 - dones_flat) * next_state_values

            if critic_target_clip is not None and critic_target_clip > 0:
                real_targets_base = real_targets_base.clamp(-critic_target_clip, critic_target_clip)

        if targets.dim() > 1:
            real_targets_expanded = real_targets_base.unsqueeze(1).expand_as(targets)
        else:
            real_targets_expanded = real_targets_base

        if critic_real_mix and critic_real_mix > 0:
            targets = (1.0 - critic_real_mix) * targets + critic_real_mix * real_targets_expanded

        targets_pre_norm = targets

        if critic_target_standardize:
            target_mean = targets.mean()
            target_std = targets.std(unbiased=False).clamp(min=1e-5)
            targets = (targets - target_mean) / target_std

        actor_returns = returns
        actor_normalize_returns = self._actor_normalize_returns
        if actor_normalize_returns is None:
            actor_normalize_returns = bool(self.config.get('actor_normalize_returns', False))

        if actor_normalize_returns:
            actor_scale = actor_returns.std(unbiased=False).clamp(min=1e-5)
            actor_returns = actor_returns / actor_scale
            losses['actor_return_scale'] = actor_scale
        else:
            losses['actor_return_scale'] = torch.tensor(1.0, device=actor_returns.device)

        actor_loss = -actor_returns.mean()
        losses['actor_loss'] = actor_loss
        losses['policy_loss'] = actor_loss

        imagined_states_flat = imagined['states'].reshape(-1, imagined['states'].shape[-1])
        returns_flat = targets.reshape(-1)

        value_preds = self.value_function(imagined_states_flat)
        if value_preds.dim() > 1:
            value_preds = value_preds.squeeze(-1)

        value_targets = returns_flat.detach()
        critic_loss = torch.nn.functional.mse_loss(value_preds, value_targets)
        losses['critic_loss'] = critic_loss
        losses['value_loss'] = critic_loss

        # Detached statistics to monitor critic behavior without affecting gradients.
        value_preds_stats = value_preds.detach()
        td_errors = value_preds_stats - value_targets
        losses['critic_value_mean'] = value_preds_stats.mean()
        losses['critic_value_std'] = value_preds_stats.std(unbiased=False)
        losses['critic_target_mean'] = value_targets.mean()
        losses['critic_target_std'] = value_targets.std(unbiased=False)
        losses['critic_td_error_mean'] = td_errors.mean()
        losses['critic_td_error_abs_mean'] = td_errors.abs().mean()

        entropy_coef = self.config.get('entropy_coef', 0.01)
        if entropy_coef > 0:
            imagined_states_for_entropy = imagined['states'].reshape(-1, imagined['states'].shape[-1])
            policy_dist = self.policy_head(imagined_states_for_entropy)
            entropy = policy_dist.entropy().mean()
            losses['entropy'] = entropy
            losses['entropy_bonus'] = entropy_coef * entropy
            losses['actor_loss'] = actor_loss - entropy_coef * entropy

        losses['total_loss'] = world_model_loss + actor_loss + critic_loss
        losses['mean_imagined_reward'] = imagined_rewards.mean()
        losses['mean_imagined_value'] = imagined_values.mean()
        losses['mean_imagined_continue'] = imagined_continues.mean()
        losses['mean_imagined_return'] = returns.mean()
        losses['mean_return'] = targets_pre_norm.mean()
        losses['target_std'] = targets_pre_norm.std(unbiased=False)
        if critic_target_standardize:
            losses['standardized_mean_return'] = targets.mean()
            losses['standardized_target_std'] = targets.std(unbiased=False)
        if real_targets_expanded is not None:
            losses['mean_real_return'] = real_targets_expanded.mean()
        losses['critic_real_mix'] = torch.tensor(critic_real_mix, device=returns.device)
        losses['actor_normalize_returns'] = torch.tensor(1.0 if actor_normalize_returns else 0.0, device=returns.device)

        return losses

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = {key: self._to_device(value) for key, value in batch.items()}

        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        warmup_steps = int(self.config.get('world_model_warmup_steps', 0) or 0)
        wm_updates = int(self.config.get('world_model_updates_per_batch', 1) or 1)
        actor_updates = int(self.config.get('actor_updates_per_batch', 1) or 1)
        critic_updates = int(self.config.get('critic_updates_per_batch', 1) or 1)

        batch_size = 0
        if 'observations' in batch:
            obs = batch['observations']
            if isinstance(obs, torch.Tensor):
                batch_size = obs.shape[0]
            elif hasattr(obs, 'shape'):
                batch_size = int(obs.shape[0])
        self.env_steps_processed += batch_size

        actor_enabled = self.env_steps_processed >= warmup_steps
        effective_actor_updates = actor_updates if actor_enabled else 0
        effective_critic_updates = critic_updates if actor_enabled else 0

        total_iterations = max(1, wm_updates, effective_actor_updates, effective_critic_updates)

        critic_real_mix_base = self.config.get('critic_real_return_mix', 0.0)
        critic_real_mix = float(critic_real_mix_base or 0.0)
        if not actor_enabled:
            critic_real_mix = 1.0

        actor_normalize_returns = bool(self.config.get('actor_normalize_returns', False))
        self._critic_real_mix_override = critic_real_mix
        self._actor_normalize_returns = actor_normalize_returns

        world_model_params = (
            list(self.encoder.parameters()) +
            list(self.representation_learner.parameters()) +
            list(self.dynamics_model.parameters()) +
            list(self.reward_predictor.parameters()) +
            list(self.continue_predictor.parameters())
        )

        primary_losses: Optional[Dict[str, torch.Tensor]] = None
        wm_grad_norm = actor_grad_norm = critic_grad_norm = None

        for iteration in range(total_iterations):
            losses = self.compute_loss(batch)

            if primary_losses is None:
                primary_losses = losses

            world_model_loss = losses['world_model_loss']
            actor_loss = losses['actor_loss']
            critic_loss = losses['critic_loss']

            run_world_model = iteration < wm_updates
            run_actor = iteration < effective_actor_updates
            run_critic = iteration < effective_critic_updates

            saved_world_model_grads = None

            if run_world_model:
                self.world_model_optimizer.zero_grad()
                world_model_loss.backward(retain_graph=(run_actor or run_critic))
                saved_world_model_grads = [
                    None if param.grad is None else param.grad.detach().clone()
                    for param in world_model_params
                ]

            if run_actor:
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=run_critic)

            if run_critic:
                self.critic_optimizer.zero_grad()
                critic_loss.backward()

            if run_world_model and saved_world_model_grads is not None:
                for param, saved_grad in zip(world_model_params, saved_world_model_grads):
                    if saved_grad is None:
                        param.grad = None
                    else:
                        if param.grad is None:
                            param.grad = saved_grad
                        else:
                            param.grad.copy_(saved_grad)

            if max_grad_norm and max_grad_norm > 0:
                if run_world_model:
                    wm_grad_norm = torch.nn.utils.clip_grad_norm_(world_model_params, max_grad_norm)
                if run_actor:
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_head.parameters(), max_grad_norm)
                if run_critic:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.value_function.parameters(), max_grad_norm)

            if run_world_model:
                self.world_model_optimizer.step()
            if run_actor:
                self.actor_optimizer.step()
            if run_critic:
                self.critic_optimizer.step()

        self._critic_real_mix_override = None
        self._actor_normalize_returns = None

        self.step += 1

        losses_for_logging = primary_losses or {}
        metrics: Dict[str, float] = {}
        for key, value in losses_for_logging.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    metrics[key] = float(value.detach().cpu())
                else:
                    metrics[key] = float(value.detach().mean().cpu())
            else:
                metrics[key] = float(value)

        if max_grad_norm and max_grad_norm > 0:
            if wm_grad_norm is not None:
                metrics['world_model_grad_norm'] = float(_tensor_to_float(wm_grad_norm))
            if actor_grad_norm is not None:
                metrics['actor_grad_norm'] = float(_tensor_to_float(actor_grad_norm))
            if critic_grad_norm is not None:
                metrics['critic_grad_norm'] = float(_tensor_to_float(critic_grad_norm))

        metrics['warmup_phase'] = 0.0 if actor_enabled else 1.0
        metrics['critic_real_mix'] = critic_real_mix
        metrics['actor_normalize_returns'] = 1.0 if actor_normalize_returns else 0.0
        metrics['env_steps_processed'] = float(self.env_steps_processed)

        return metrics

    def _compute_lambda_returns(self,
                                rewards: torch.Tensor,
                                values: torch.Tensor,
                                continues: torch.Tensor,
                                gamma: float,
                                lambda_: float) -> torch.Tensor:
        batch_size, horizon = rewards.shape
        returns = torch.zeros_like(rewards)

        last_value = values[:, -1] if horizon > 0 else torch.zeros(batch_size, device=rewards.device)
        next_return = last_value
        for t in reversed(range(horizon)):
            discount = gamma * continues[:, t]
            reward = rewards[:, t]
            value = values[:, t]
            next_return = reward + discount * ((1 - lambda_) * value + lambda_ * next_return)
            returns[:, t] = next_return

        return returns

    def _save_additional_components(self) -> Dict[str, Any]:
        additional = {
            'dynamics_model': self.dynamics_model.state_dict(),
            'reward_predictor': self.reward_predictor.state_dict(),
            'continue_predictor': self.continue_predictor.state_dict(),
            'value_function': self.value_function.state_dict(),
        }
        if self.planner is not None:
            additional['planner'] = self.planner.state_dict()
        return additional

    def _load_additional_components(self, checkpoint: Dict[str, Any]):
        if 'dynamics_model' in checkpoint:
            self.dynamics_model.load_state_dict(checkpoint['dynamics_model'])
        if 'reward_predictor' in checkpoint:
            self.reward_predictor.load_state_dict(checkpoint['reward_predictor'])
        if 'continue_predictor' in checkpoint:
            self.continue_predictor.load_state_dict(checkpoint['continue_predictor'])
        if 'value_function' in checkpoint:
            self.value_function.load_state_dict(checkpoint['value_function'])
        if 'planner' in checkpoint and self.planner is not None:
            self.planner.load_state_dict(checkpoint['planner'])

    def _to_device(self, value: Union[torch.Tensor, Dict[str, torch.Tensor], Any]):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        return value


def _tensor_to_float(value: Union[torch.Tensor, float]) -> float:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)
