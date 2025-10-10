"""PPO paradigm driven by externally constructed components."""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .paradigm import ModelFreeParadigm
from ...utils.registry import register_algorithm


@register_algorithm("ppo")
class PPOParadigm(ModelFreeParadigm):
    """PPO implementation in the modular paradigm architecture."""

    def __init__(self, config: Dict[str, Any]):
        cfg = dict(config)
        components = cfg.pop('components', None)
        if not components:
            raise ValueError("PPOParadigm requires a 'components' dict with encoder, policy_head, value_function")

        missing = [name for name in ('encoder', 'policy_head', 'value_function') if name not in components]
        if missing:
            raise ValueError(f"PPOParadigm missing required components: {missing}")

        encoder = components['encoder']
        representation_learner = components.get('representation_learner')
        if representation_learner is None:
            from ...components.world_models.representation_learners.identity import IdentityRepresentationLearner
            representation_learner = IdentityRepresentationLearner({'device': cfg.get('device', 'cpu'),
                                                                     'representation_dim': getattr(encoder, 'output_dim', None)})

        policy_head = components['policy_head']
        value_function = components['value_function']

        observation_space = cfg.get('observation_space')
        action_space = cfg.get('action_space')
        if observation_space is None or action_space is None:
            raise ValueError("PPOParadigm requires observation_space and action_space in config")

        self.action_space = action_space

        super().__init__(encoder, representation_learner, policy_head, value_function, cfg)

        # Hyperparameters
        self.actor_lr = float(cfg.get('actor_lr', cfg.get('lr', 3e-4)))
        self.critic_lr = float(cfg.get('critic_lr', cfg.get('lr', 3e-4)))
        self.clip_ratio = float(cfg.get('clip_ratio', 0.2))
        self.value_coef = float(cfg.get('value_coef', 0.5))
        self.entropy_coef_initial = float(cfg.get('entropy_coef', 0.01))
        self.entropy_coef_final = float(cfg.get('entropy_coef_final', self.entropy_coef_initial))
        self.entropy_coef_schedule = cfg.get('entropy_coef_schedule', None)
        self.entropy_schedule_fraction = float(cfg.get('entropy_schedule_fraction', 1.0))
        self.entropy_coef = self.entropy_coef_initial
        self.max_grad_norm = float(cfg.get('max_grad_norm', 0.5))
        self.ppo_epochs = int(cfg.get('ppo_epochs', 4))
        self.minibatch_size = int(cfg.get('minibatch_size', 64))
        self.normalize_advantages = bool(cfg.get('normalize_advantages', True))
        self.clip_value_loss = bool(cfg.get('clip_value_loss', True))
        self.log_std_min = float(cfg.get('log_std_min', -20.0))
        self.log_std_max = float(cfg.get('log_std_max', 2.0))

        self.action_space_type = 'discrete' if getattr(action_space, 'discrete', False) else 'continuous'

        # Composite networks for trainer utilities
        self.actor_network = CompositeActor(self.encoder, self.representation_learner, self.policy_head)
        self.critic_network = CompositeCritic(self.encoder, self.representation_learner, self.value_function)
        self.networks = {
            'actor': self.actor_network,
            'critic': self.critic_network,
            'encoder': self.encoder,
            'policy_head': self.policy_head,
            'value_function': self.value_function,
        }

        # Shared optimizer with parameter groups
        param_groups = [
            {'params': self.encoder.parameters(), 'lr': self.actor_lr},
            {'params': self.policy_head.parameters(), 'lr': self.actor_lr},
            {'params': self.value_function.parameters(), 'lr': self.critic_lr},
        ]
        self.optimizer = torch.optim.Adam(param_groups)
        self.optimizers = {'combined': self.optimizer}

        self.step = 0

    # Convenience property for clipping
    @property
    def _trainable_parameters(self):
        for module in (self.encoder, self.policy_head, self.value_function):
            for param in module.parameters():
                yield param

    def compute_ppo_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = batch['observations']
        actions = batch['actions']
        if self.action_space_type == 'discrete':
            actions = actions.long()

        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        returns = batch['returns']

        if self.normalize_advantages:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / adv_std
            else:
                advantages = advantages - adv_mean

        log_probs, values, entropy = self.evaluate_actions(observations, actions)
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        if self.clip_value_loss and 'old_values' in batch:
            old_values = batch['old_values']
            values_clipped = old_values + torch.clamp(values - old_values, -self.clip_ratio, self.clip_ratio)
            value_loss_1 = F.mse_loss(values, returns)
            value_loss_2 = F.mse_loss(values_clipped, returns)
            value_loss = torch.max(value_loss_1, value_loss_2)
        else:
            value_loss = F.mse_loss(values, returns)

        entropy_loss = -entropy.mean()
        return policy_loss, value_loss, entropy_loss

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        policy_loss, value_loss, entropy_loss = self.compute_ppo_loss(batch)
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
        }

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = {k: v.to(self.device) for k, v in batch.items()}
        batch_size = batch['observations'].shape[0]

        diagnostics: Dict[str, float] = {}

        with torch.no_grad():
            advantages = batch.get('advantages')
            if advantages is not None:
                adv_mean = advantages.mean().item()
                adv_std = advantages.std(unbiased=False).item() if advantages.numel() > 1 else 0.0
                diagnostics['advantages_mean'] = adv_mean
                diagnostics['advantages_std'] = adv_std

            returns = batch.get('returns')
            if returns is not None:
                returns_flat = returns.view(-1)
                diagnostics['critic_target_mean'] = returns_flat.mean().item()
                diagnostics['critic_target_std'] = (
                    returns_flat.std(unbiased=False).item() if returns_flat.numel() > 1 else 0.0
                )

            observations = batch['observations']
            if not torch.is_floating_point(observations):
                observations = observations.float()

            values_pred = self.critic_network(observations)
            values_pred = values_pred.view(-1)
            diagnostics['critic_pred_mean'] = values_pred.mean().item()
            diagnostics['critic_pred_std'] = (
                values_pred.std(unbiased=False).item() if values_pred.numel() > 1 else 0.0
            )

            if returns is not None:
                value_delta = values_pred - returns_flat
                diagnostics['critic_delta_mean'] = value_delta.mean().item()
                diagnostics['critic_delta_abs_mean'] = value_delta.abs().mean().item()

            dist = self.actor_network(observations)
            if hasattr(dist, 'logits'):
                logits = dist.logits
                probs = dist.probs
                if logits.dim() >= 2:
                    mean_logits = logits.mean(dim=0)
                    for idx in range(mean_logits.shape[-1]):
                        diagnostics[f'action_logit_mean_{idx}'] = mean_logits[idx].item()
                if probs.dim() >= 2:
                    mean_probs = probs.mean(dim=0)
                    for idx in range(mean_probs.shape[-1]):
                        diagnostics[f'action_prob_mean_{idx}'] = mean_probs[idx].item()
            elif hasattr(dist, 'loc') and hasattr(dist, 'scale'):
                diagnostics['action_loc_mean'] = dist.loc.mean().item()
                diagnostics['action_scale_mean'] = dist.scale.mean().item()
                if dist.loc.numel() > 1:
                    diagnostics['action_loc_std'] = dist.loc.std(unbiased=False).item()
                if dist.scale.numel() > 1:
                    diagnostics['action_scale_std'] = dist.scale.std(unbiased=False).item()

        indices = np.arange(batch_size)

        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'grad_norm': 0.0,
        }

        updates = 0
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]
                minibatch = {k: v[mb_indices] for k, v in batch.items()}

                losses = self.compute_loss(minibatch)
                total_loss = losses['total_loss']

                self.optimizer.zero_grad()
                total_loss.backward()

                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(list(self._trainable_parameters), self.max_grad_norm)
                    grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                else:
                    grad_norm_value = 0.0

                self.optimizer.step()

                metrics['policy_loss'] += losses['policy_loss'].item()
                metrics['value_loss'] += losses['value_loss'].item()
                metrics['entropy_loss'] += losses['entropy_loss'].item()
                metrics['total_loss'] += total_loss.item()
                metrics['grad_norm'] += grad_norm_value
                updates += 1

        if updates > 0:
            for key in metrics:
                metrics[key] /= updates

        self.step += 1
        metrics['entropy_coef'] = float(self.entropy_coef)
        metrics.update(diagnostics)
        return metrics

    def update_schedules(self, progress: float) -> None:
        """Update any annealed hyperparameters based on training progress."""
        if self.entropy_coef_schedule == 'linear':
            if self.entropy_schedule_fraction <= 0:
                schedule_progress = 1.0
            else:
                schedule_progress = progress / self.entropy_schedule_fraction
            clipped = max(0.0, min(1.0, schedule_progress))
            self.entropy_coef = (
                self.entropy_coef_initial
                + (self.entropy_coef_final - self.entropy_coef_initial) * clipped
            )


class CompositeActor(nn.Module):
    def __init__(self, encoder, representation_learner, policy_head):
        super().__init__()
        self.encoder = encoder
        self.representation_learner = representation_learner
        self.policy_head = policy_head

    def forward(self, observations):
        features = self.encoder(observations)
        representations = self.representation_learner.encode(features)
        return self.policy_head(representations)


class CompositeCritic(nn.Module):
    def __init__(self, encoder, representation_learner, value_function):
        super().__init__()
        self.encoder = encoder
        self.representation_learner = representation_learner
        self.value_function = value_function

    def forward(self, observations):
        features = self.encoder(observations)
        representations = self.representation_learner.encode(features)
        return self.value_function(representations)
