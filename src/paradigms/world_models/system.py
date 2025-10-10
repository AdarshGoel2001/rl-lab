"""Core orchestration utilities for modular world-model paradigms."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch.distributions import Distribution

from ...components.encoders.base import BaseEncoder
from ...components.world_models.representation_learners.base import BaseRepresentationLearner
from ...components.world_models.dynamics.base import BaseDynamicsModel
from ...components.policy_heads.base import BasePolicyHead
from ...components.value_functions.base import BaseValueFunction
from ...components.world_models.controllers.base import BaseController
from ...components.world_models.latents import LatentBatch
from ...components.world_models.predictors import BaseRewardPredictor
from ...components.world_models.decoders import BaseObservationDecoder


class WorldModelSystem:
    """High-level coordinator for modular world-model training and rollout."""

    def __init__(
        self,
        *,
        encoder: BaseEncoder,
        representation_learner: BaseRepresentationLearner,
        dynamics_model: BaseDynamicsModel,
        policy_head: BasePolicyHead,
        value_function: BaseValueFunction,
        planner: Optional[BaseController] = None,
        reward_predictor: Optional[BaseRewardPredictor] = None,
        observation_decoder: Optional[BaseObservationDecoder] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.encoder = encoder
        self.representation_learner = representation_learner
        self.dynamics_model = dynamics_model
        self.policy_head = policy_head
        self.value_function = value_function
        self.planner = planner
        self.reward_predictor = reward_predictor
        self.observation_decoder = observation_decoder
        self.config = config or {}

        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(self.encoder.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self._move_to_device(self.device)

    # ---------------------------------------------------------------------
    # Device helpers
    # ---------------------------------------------------------------------
    def _move_to_device(self, device: torch.device) -> None:
        self.encoder.to(device)
        self.representation_learner.to(device)
        self.dynamics_model.to(device)
        self.policy_head.to(device)
        self.value_function.to(device)
        if self.reward_predictor is not None:
            self.reward_predictor.to(device)
        if self.planner is not None:
            self.planner.to(device)
        if self.observation_decoder is not None:
            self.observation_decoder.to(device)

    def to(self, device: torch.device) -> None:
        """Move all managed modules to the target device."""
        self.device = device
        self._move_to_device(device)

    # ---------------------------------------------------------------------
    # Encoding utilities
    # ---------------------------------------------------------------------
    def encode(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        prev_state: Optional[Any] = None,
        prev_action: Optional[torch.Tensor] = None,
        sample: bool = True,
        reset_state: bool = False,
    ) -> LatentBatch:
        """Encode raw observations into latent state representations."""
        if reset_state and hasattr(self.representation_learner, "reset_state"):
            self.representation_learner.reset_state()

        features = self.encoder(observations)

        encode_kwargs: Dict[str, Any] = {"sample": sample}
        if prev_state is not None:
            encode_kwargs["prev_state"] = prev_state
        if prev_action is not None:
            encode_kwargs["prev_action"] = prev_action

        latent = self.representation_learner.encode(features, **encode_kwargs)

        extras: Dict[str, Any] = {}
        cached_state = None
        if hasattr(self.representation_learner, "get_cached_state"):
            cached_state = self.representation_learner.get_cached_state()
        if cached_state is not None:
            extras["rssm_state"] = cached_state
            extras["deterministic"] = cached_state.deterministic
            extras["stochastic"] = cached_state.stochastic
            if cached_state.posterior_mean is not None:
                extras["posterior_mean"] = cached_state.posterior_mean
                extras["posterior_std"] = cached_state.posterior_std
            if cached_state.prior_mean is not None:
                extras["prior_mean"] = cached_state.prior_mean
                extras["prior_std"] = cached_state.prior_std

        return LatentBatch(latent=latent, features=features, extras=extras)

    # ---------------------------------------------------------------------
    # Acting utilities
    # ---------------------------------------------------------------------
    def act(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Distribution:
        """Produce an action distribution for the provided observations."""
        latent_batch = self.encode(observations)
        latent = latent_batch.latent

        if self.planner is not None:
            if hasattr(self.planner, "act"):
                return self.planner.act(latent, self.dynamics_model, self.value_function)
            return self.planner.plan(latent, self.dynamics_model, self.value_function)

        return self.policy_head(latent, context)

    def value(
        self,
        observations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Estimate state value from raw observations."""
        latent_batch = self.encode(observations)
        return self.value_function(latent_batch.latent)

    # ---------------------------------------------------------------------
    # Imagination rollout
    # ---------------------------------------------------------------------
    def imagine(
        self,
        latent_batch: LatentBatch,
        *,
        horizon: Optional[int] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Roll out imagined trajectories starting from the given latent state."""
        horizon = horizon or self.config.get("imagination_length", 15)
        current_state = latent_batch.latent

        imagined_states = [current_state]
        imagined_actions = []
        imagined_rewards = []
        imagined_values = []
        imagined_next_values = []
        imagined_log_probs = []
        imagined_entropies = []

        for _ in range(horizon):
            action_dist = self.policy_head(current_state)
            if deterministic:
                action = getattr(action_dist, "mean", action_dist.sample())
            else:
                if getattr(action_dist, "has_rsample", False):
                    action = action_dist.rsample()
                else:
                    action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            if log_prob.dim() > 1:
                log_prob = log_prob.sum(dim=-1)

            entropy = action_dist.entropy()
            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            state_value = self.value_function(current_state)
            if state_value.dim() > 1:
                state_value = state_value.squeeze(-1)

            action_for_model = action
            if action_for_model.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                num_actions = getattr(self.policy_head, "action_dim", None)
                if num_actions is None:
                    raise ValueError("Discrete actions require policy head to define action_dim")
                action_for_model = F.one_hot(action_for_model.long(), num_classes=num_actions).float()

            next_state_dist = self.dynamics_model(current_state, action_for_model)
            if deterministic:
                next_state = getattr(next_state_dist, "mean", next_state_dist.sample())
            else:
                if getattr(next_state_dist, "has_rsample", False):
                    next_state = next_state_dist.rsample()
                else:
                    next_state = next_state_dist.sample()

            next_value = self.value_function(next_state)
            if next_value.dim() > 1:
                next_value = next_value.squeeze(-1)

            if self.reward_predictor is not None:
                reward_pred = self.reward_predictor(next_state)
                if reward_pred.dim() > 1:
                    reward_pred = reward_pred.squeeze(-1)
            else:
                reward_pred = torch.zeros(current_state.shape[0], device=current_state.device)

            imagined_states.append(next_state)
            imagined_actions.append(action)
            imagined_rewards.append(reward_pred)
            imagined_values.append(state_value)
            imagined_next_values.append(next_value)
            imagined_log_probs.append(log_prob)
            imagined_entropies.append(entropy)

            current_state = next_state

        states = torch.stack(imagined_states[:-1], dim=1)
        next_states = torch.stack(imagined_states[1:], dim=1)
        actions = torch.stack(imagined_actions, dim=1)
        rewards = torch.stack(imagined_rewards, dim=1)
        values = torch.stack(imagined_values, dim=1)
        next_values = torch.stack(imagined_next_values, dim=1)
        log_probs = torch.stack(imagined_log_probs, dim=1)
        entropies = torch.stack(imagined_entropies, dim=1)

        bootstrap_value = next_values[:, -1]

        return {
            "states": states,
            "next_states": next_states,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "next_values": next_values,
            "log_probs": log_probs,
            "entropies": entropies,
            "bootstrap": bootstrap_value,
        }

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        next_values: torch.Tensor,
        bootstrap: torch.Tensor,
        discounts: torch.Tensor,
        lambda_coef: float,
    ) -> torch.Tensor:
        """Compute Dreamer-style lambda-returns for imagined rollouts."""
        time_horizon = rewards.shape[1]
        next_return = bootstrap
        returns_list = []

        for t in reversed(range(time_horizon)):
            reward_t = rewards[:, t]
            discount_t = discounts[:, t]
            value_tp1 = next_values[:, t]
            blended_target = (1.0 - lambda_coef) * value_tp1 + lambda_coef * next_return
            next_return = reward_t + discount_t * blended_target
            returns_list.append(next_return)

        returns = torch.stack(list(reversed(returns_list)), dim=1)
        return returns

    # ---------------------------------------------------------------------
    # Loss computation
    # ---------------------------------------------------------------------
    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute supervised losses for world-model training."""
        losses: Dict[str, torch.Tensor] = {}

        tensor_batch: Dict[str, torch.Tensor] = {
            key: value.to(self.device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

        observations = tensor_batch["observations"]
        next_observations = tensor_batch["next_observations"]
        actions = tensor_batch.get("actions")
        rewards_tensor = tensor_batch.get("rewards")
        sequence_mask = tensor_batch.get("sequence_mask")

        observations = observations.to(self.device)
        next_observations = next_observations.to(self.device)
        if not torch.is_floating_point(observations):
            observations = observations.float()
        if not torch.is_floating_point(next_observations):
            next_observations = next_observations.float()

        is_sequence = observations.dim() >= 3
        if is_sequence:
            batch_size, time_steps = observations.shape[:2]
            if sequence_mask is None:
                sequence_mask = torch.ones(
                    batch_size,
                    time_steps,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                sequence_mask = sequence_mask.to(self.device).float()

            obs_flat = observations.reshape(batch_size * time_steps, *observations.shape[2:])
            next_obs_flat = next_observations.reshape(
                batch_size * time_steps, *next_observations.shape[2:]
            )
            if actions is not None:
                actions_flat = actions.reshape(batch_size * time_steps, *actions.shape[2:]) if actions.dim() > 2 else actions.reshape(-1)
            else:
                actions_flat = None
            if rewards_tensor is not None:
                rewards_flat = rewards_tensor.reshape(batch_size * time_steps, -1)
            else:
                rewards_flat = None
            mask_flat = sequence_mask.reshape(-1)
            valid_mask = mask_flat > 0.0
            if valid_mask.sum() == 0:
                raise ValueError("Sequence mask removed all transitions from batch")

            observations = obs_flat[valid_mask]
            next_observations = next_obs_flat[valid_mask]
            if actions_flat is not None:
                actions = actions_flat[valid_mask]
            if rewards_flat is not None:
                rewards_tensor = rewards_flat[valid_mask]
        else:
            if not torch.is_floating_point(observations):
                observations = observations.float()
            if not torch.is_floating_point(next_observations):
                next_observations = next_observations.float()

        if actions is not None:
            if actions.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                action_dim = getattr(self.policy_head, "action_dim", None)
                if action_dim is None:
                    raise ValueError("Discrete actions require policy head to define action_dim")
                actions = F.one_hot(actions.long(), num_classes=action_dim).float()
                if actions.dim() == 3 and actions.shape[1] == 1:
                    actions = actions.squeeze(1)
            elif not torch.is_floating_point(actions):
                actions = actions.float()

        if rewards_tensor is not None and not torch.is_floating_point(rewards_tensor):
            rewards_tensor = rewards_tensor.float()

        latent_batch = self.encode(observations, reset_state=True)
        posterior_state: Optional[Any] = None
        if latent_batch.extras:
            posterior_state = latent_batch.extras.get("rssm_state")

        if self.observation_decoder is not None:
            target_mode = self.config.get("reconstruction_target", "observations")
            if target_mode == "features" and latent_batch.features is not None:
                reconstruction_target = latent_batch.features
            else:
                reconstruction_target = observations

            reconstruction_pred = self.observation_decoder(latent_batch.latent)
            reconstruction_pred = reconstruction_pred.view(reconstruction_pred.shape[0], -1)
            reconstruction_target = reconstruction_target.view(reconstruction_target.shape[0], -1)

            reconstruction_loss = F.mse_loss(reconstruction_pred, reconstruction_target)
            recon_scale = self.config.get("reconstruction_loss_scale", 1.0)
            losses["reconstruction_loss"] = reconstruction_loss * recon_scale

        elif hasattr(self.representation_learner, "decode") and latent_batch.features is not None:
            reconstructed = self.representation_learner.decode(latent_batch.latent)
            reconstruction_loss = F.mse_loss(reconstructed, latent_batch.features)
            recon_scale = self.config.get("reconstruction_loss_scale", 1.0)
            losses["reconstruction_loss"] = reconstruction_loss * recon_scale

        if latent_batch.features is not None:
            # NOTE: we compute representation losses before encoding next observations so the
            # learner's cached statistics still match this latent. Later we can revisit this
            # ordering to see if sharing encoder passes can unlock extra efficiency.
            repr_losses = self.representation_learner.representation_loss(
                latent_batch.features,
                state=posterior_state,
            )
            kl_scale = self.config.get("kl_scale", 1.0)
            free_nats = self.config.get("kl_free_nats", 0.0)
            processed = {}
            for name, loss in repr_losses.items():
                if name.endswith("loss"):
                    target = loss
                    if name == "representation_loss" and free_nats > 0:
                        free_threshold = loss.new_tensor(free_nats)
                        target = torch.clamp(target - free_threshold, min=0.0)
                    processed[name] = target * kl_scale
                else:
                    processed[name] = loss
            losses.update(processed)

        next_latent_batch = self.encode(
            next_observations,
            prev_state=posterior_state,
            prev_action=actions,
            sample=True,
        )

        if actions is not None:
            dynamics_losses = self.dynamics_model.dynamics_loss(
                latent_batch.latent,
                actions,
                next_latent_batch.latent,
            )
            loss_scale = self.config.get("dynamics_loss_scale", 1.0)
            losses.update({
                name: loss * loss_scale if name.endswith("loss") else loss
                for name, loss in dynamics_losses.items()
            })

        if self.reward_predictor is not None and rewards_tensor is not None:
            if rewards_tensor.dim() > 1:
                rewards_flat = rewards_tensor.squeeze(-1)
            else:
                rewards_flat = rewards_tensor
            predicted_reward = self.reward_predictor(latent_batch.latent)
            if predicted_reward.dim() > 1:
                predicted_reward = predicted_reward.squeeze(-1)
            reward_loss = F.mse_loss(predicted_reward, rewards_flat)
            reward_scale = self.config.get("reward_loss_scale", 1.0)
            losses["reward_loss"] = reward_loss * reward_scale

        returns_tensor = tensor_batch.get("returns")
        if returns_tensor is not None:
            returns_tensor = returns_tensor.to(self.device)
            if is_sequence:
                returns_flat = returns_tensor.reshape(batch_size * time_steps, -1)
                returns_tensor = returns_flat[valid_mask]
            if returns_tensor.dim() > 1:
                returns_tensor = returns_tensor.squeeze(-1)
        
        real_value_loss: Optional[torch.Tensor] = None
        if returns_tensor is not None:
            value_losses = self.value_function.value_loss(latent_batch.latent, returns_tensor)
            losses.update(value_losses)
            real_value_loss = value_losses.get("value_loss")

        if is_sequence:
            # Sample random starting points from sequences for imagination
            # This utilizes temporal diversity instead of always starting from t=0
            batch_size, time_steps = tensor_batch["observations"].shape[:2]

            # For each sequence, sample a valid timestep to start imagination
            if sequence_mask is not None:
                start_indices = []
                for b in range(batch_size):
                    valid_timesteps = torch.where(sequence_mask[b] > 0.0)[0]
                    if len(valid_timesteps) > 0:
                        # Randomly sample from valid timesteps
                        idx = torch.randint(0, len(valid_timesteps), (1,)).item()
                        start_indices.append(valid_timesteps[idx].item())
                    else:
                        start_indices.append(0)
                start_indices = torch.tensor(start_indices, device=self.device)
            else:
                # No mask: sample uniformly from all timesteps
                start_indices = torch.randint(0, time_steps, (batch_size,), device=self.device)

            # Extract observations at sampled timesteps
            batch_indices = torch.arange(batch_size, device=self.device)
            initial_observations = tensor_batch["observations"][batch_indices, start_indices]
            if not torch.is_floating_point(initial_observations):
                initial_observations = initial_observations.float()
            initial_latent_batch = self.encode(initial_observations, reset_state=True)
        else:
            initial_latent_batch = latent_batch.detach()

        imagined = self.imagine(initial_latent_batch.detach())
        imagined_rewards = imagined["rewards"]
        imagined_values = imagined["values"]
        imagined_next_values = imagined["next_values"]
        imagined_log_probs = imagined["log_probs"]
        imagined_entropies = imagined["entropies"]

        batch_size, horizon = imagined_rewards.shape
        gamma = self.config.get("gamma", 0.99)
        lambda_coef = self.config.get("lambda_return", 0.95)

        discounts = torch.full_like(imagined_rewards, fill_value=gamma)
        lambda_returns = self._compute_lambda_returns(
            imagined_rewards,
            imagined_next_values,
            imagined["bootstrap"],
            discounts,
            lambda_coef,
        )

        advantages = lambda_returns - imagined_values
        actor_loss = -(imagined_log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(imagined_values, lambda_returns.detach())

        policy_scale = self.config.get("policy_loss_scale", 1.0)
        value_scale = self.config.get("value_loss_scale", 1.0)

        policy_objective = actor_loss * policy_scale
        critic_objective = critic_loss * value_scale

        losses["policy_loss"] = policy_objective

        if real_value_loss is not None:
            losses["value_loss"] = real_value_loss + critic_objective
        else:
            losses["value_loss"] = critic_objective

        entropy_coef = self.config.get("entropy_coef", 0.01)
        if entropy_coef > 0:
            entropy = imagined_entropies.mean()
            losses["entropy_loss"] = -entropy_coef * entropy
            losses["entropy"] = entropy

        total_loss = torch.tensor(0.0, device=self.device)
        for name, loss in losses.items():
            if name.endswith("loss"):
                total_loss = total_loss + loss
        losses["total_loss"] = total_loss

        return losses
