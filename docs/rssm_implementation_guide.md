# RSSM Implementation Guide

## Overview

The Recurrent State-Space Model (RSSM) is the core representation learner used in PlaNet, Dreamer v1, v2, and v3. This guide shows exactly how to implement it in your architecture.

---

## RSSM Architecture

### Key Concept: Two Paths

```
Deterministic Path (GRU):   h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
Stochastic Path (Prior):    s_t ~ p(s_t | h_t)
Stochastic Path (Posterior): s_t ~ q(s_t | h_t, o_t)
```

**During Training:** Use posterior (has observation)
**During Imagination:** Use prior (no observation)

### Intuition

- **Deterministic (h):** Carries history, like RNN memory
- **Stochastic (s):** Captures uncertainty and multi-modality
- **Combined (h, s):** Full state representation

---

## Implementation: Gaussian RSSM (PlaNet, Dreamer v1)

### File: `src/components/representation_learners/rssm_gaussian.py`

```python
"""
Gaussian RSSM Representation Learner

Used in:
- PlaNet (Hafner et al., 2019)
- Dreamer v1 (Hafner et al., 2020)
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

from .base import BaseRepresentationLearner
from ...utils.registry import register_representation_learner


def build_mlp(input_dim, output_dim, hidden_dims, activation='relu'):
    """Helper to build MLP"""
    layers = []
    dims = [input_dim] + hidden_dims + [output_dim]

    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:  # No activation on output
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())

    return nn.Sequential(*layers)


@register_representation_learner('rssm_gaussian')
class RSSMGaussianRepresentationLearner(BaseRepresentationLearner):
    """
    Recurrent State-Space Model with Gaussian stochastic states.

    Architecture:
        Deterministic: h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])
        Prior:         s_t ~ N(μ_prior(h_t), σ_prior(h_t))
        Posterior:     s_t ~ N(μ_post([h_t, o_t]), σ_post([h_t, o_t]))

    Config:
        stochastic_size: Dimension of stochastic state (default: 30)
        deterministic_size: Dimension of deterministic state (default: 200)
        hidden_size: Hidden layer size for prior/posterior nets (default: 200)
        min_std: Minimum standard deviation (default: 0.1)
        kl_strategy: 'regular', 'free_bits', or 'balanced' (default: 'regular')
        free_bits: Free bits threshold if using free_bits strategy (default: 1.0)
        kl_balance_alpha: Alpha for balanced KL (default: 0.8)
    """

    def __init__(self, config: Dict[str, Any]):
        self.stochastic_size = config.get('stochastic_size', 30)
        self.deterministic_size = config.get('deterministic_size', 200)
        self.hidden_size = config.get('hidden_size', 200)
        self.min_std = config.get('min_std', 0.1)
        self.kl_strategy = config.get('kl_strategy', 'regular')
        self.free_bits = config.get('free_bits', 1.0)
        self.kl_balance_alpha = config.get('kl_balance_alpha', 0.8)

        # Store for representation_dim property
        self._representation_dim = self.stochastic_size + self.deterministic_size

        super().__init__(config)

    def _build_learner(self):
        """Build RSSM components"""
        # Input to GRU: [stochastic_{t-1}, action_{t-1}]
        # We'll get action_dim from config or infer at runtime
        self.action_dim = self.config.get('action_dim', None)

        # GRU for deterministic path
        # Input: concat(stochastic, action)
        # We'll create this lazily if action_dim not provided
        if self.action_dim is not None:
            self.gru = nn.GRUCell(
                self.stochastic_size + self.action_dim,
                self.deterministic_size
            )
        else:
            self.gru = None  # Create lazily

        # Prior network: h_t -> (mean, std) for s_t
        self.prior_net = build_mlp(
            self.deterministic_size,
            2 * self.stochastic_size,  # mean + log_std
            [self.hidden_size],
            activation='elu'
        )

        # Posterior network: [h_t, features_t] -> (mean, std) for s_t
        # features_dim will be set at runtime
        self.feature_dim = self.config.get('feature_dim', None)
        if self.feature_dim is not None:
            self.posterior_net = build_mlp(
                self.deterministic_size + self.feature_dim,
                2 * self.stochastic_size,  # mean + log_std
                [self.hidden_size],
                activation='elu'
            )
        else:
            self.posterior_net = None  # Create lazily

        # Hidden state for stateful encoding
        self._hidden_state = None
        self._batch_size = None

    def _create_gru_if_needed(self, action_dim: int):
        """Lazily create GRU if not created during init"""
        if self.gru is None:
            self.gru = nn.GRUCell(
                self.stochastic_size + action_dim,
                self.deterministic_size
            ).to(self.device)
            self.action_dim = action_dim

    def _create_posterior_if_needed(self, feature_dim: int):
        """Lazily create posterior network if not created during init"""
        if self.posterior_net is None:
            self.posterior_net = build_mlp(
                self.deterministic_size + feature_dim,
                2 * self.stochastic_size,
                [self.hidden_size],
                activation='elu'
            ).to(self.device)
            self.feature_dim = feature_dim

    def reset_hidden_state(self, batch_size: int):
        """Reset hidden state for new episode"""
        self._batch_size = batch_size
        self._hidden_state = {
            'deterministic': torch.zeros(
                batch_size, self.deterministic_size, device=self.device
            ),
            'stochastic': torch.zeros(
                batch_size, self.stochastic_size, device=self.device
            )
        }

    def _get_dist_params(self, raw_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract mean and std from network output"""
        mean, log_std = torch.chunk(raw_output, 2, dim=-1)
        std = F.softplus(log_std) + self.min_std
        return mean, std

    def _sample_gaussian(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Sample from Gaussian with reparameterization"""
        return mean + std * torch.randn_like(std)

    def _compute_prior(self, deterministic: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Compute prior distribution p(s_t | h_t)"""
        raw_output = self.prior_net(deterministic)
        mean, std = self._get_dist_params(raw_output)

        # Independent Normal distribution
        prior_dist = Independent(Normal(mean, std), 1)

        # Sample
        if self.training:
            stochastic = self._sample_gaussian(mean, std)
        else:
            stochastic = mean  # Use mean during eval

        return stochastic, prior_dist

    def _compute_posterior(
        self,
        deterministic: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, Normal]:
        """Compute posterior distribution q(s_t | h_t, o_t)"""
        # Ensure posterior net exists
        if self.posterior_net is None:
            self._create_posterior_if_needed(features.shape[-1])

        concat_input = torch.cat([deterministic, features], dim=-1)
        raw_output = self.posterior_net(concat_input)
        mean, std = self._get_dist_params(raw_output)

        # Independent Normal distribution
        posterior_dist = Independent(Normal(mean, std), 1)

        # Sample
        if self.training:
            stochastic = self._sample_gaussian(mean, std)
        else:
            stochastic = mean

        return stochastic, posterior_dist

    def _update_deterministic(
        self,
        prev_deterministic: torch.Tensor,
        prev_stochastic: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """Update deterministic state via GRU"""
        # Ensure GRU exists
        if self.gru is None:
            self._create_gru_if_needed(action.shape[-1])

        # GRU input: [stochastic_{t-1}, action_{t-1}]
        gru_input = torch.cat([prev_stochastic, action], dim=-1)
        deterministic = self.gru(gru_input, prev_deterministic)
        return deterministic

    def encode(
        self,
        features: torch.Tensor,
        sequence_context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        Encode features into RSSM states.

        Uses posterior q(s_t | h_t, o_t) when observations available.

        Args:
            features: Shape (B, feature_dim) or (B, T, feature_dim)
            sequence_context: Optional context with prev states and actions

        Returns:
            states: Shape (B, state_dim) or (B, T, state_dim)
                    where state_dim = deterministic_size + stochastic_size
        """
        is_sequence = features.dim() == 3

        if is_sequence:
            return self._encode_sequence(features, sequence_context)
        else:
            return self._encode_single(features, sequence_context)

    def _encode_single(
        self,
        features: torch.Tensor,
        sequence_context: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Encode single timestep (B, feature_dim)"""
        batch_size = features.shape[0]

        # Initialize hidden state if needed
        if self._hidden_state is None or self._batch_size != batch_size:
            self.reset_hidden_state(batch_size)

        # Get previous state and action
        if sequence_context is not None and 'prev_state' in sequence_context:
            prev_det = sequence_context['prev_deterministic']
            prev_sto = sequence_context['prev_stochastic']
            action = sequence_context['action']
        else:
            prev_det = self._hidden_state['deterministic']
            prev_sto = self._hidden_state['stochastic']
            # Default action (zeros) if not provided
            action = torch.zeros(batch_size, self.action_dim or 1, device=self.device)

        # Update deterministic state
        deterministic = self._update_deterministic(prev_det, prev_sto, action)

        # Compute posterior (use observation)
        stochastic, _ = self._compute_posterior(deterministic, features)

        # Update hidden state
        self._hidden_state['deterministic'] = deterministic
        self._hidden_state['stochastic'] = stochastic

        # Return concatenated state
        return torch.cat([deterministic, stochastic], dim=-1)

    def _encode_sequence(
        self,
        features: torch.Tensor,
        sequence_context: Optional[Dict[str, Any]]
    ) -> torch.Tensor:
        """Encode sequence (B, T, feature_dim)"""
        batch_size, seq_len, _ = features.shape

        # Get actions from context
        if sequence_context is not None and 'actions' in sequence_context:
            actions = sequence_context['actions']  # (B, T, action_dim)
        else:
            # Default actions
            action_dim = self.action_dim or 1
            actions = torch.zeros(batch_size, seq_len, action_dim, device=self.device)

        # Initialize hidden state
        self.reset_hidden_state(batch_size)

        states = []
        for t in range(seq_len):
            features_t = features[:, t]
            action_t = actions[:, t]

            # Update deterministic
            deterministic = self._update_deterministic(
                self._hidden_state['deterministic'],
                self._hidden_state['stochastic'],
                action_t
            )

            # Compute posterior
            stochastic, _ = self._compute_posterior(deterministic, features_t)

            # Update hidden state
            self._hidden_state['deterministic'] = deterministic
            self._hidden_state['stochastic'] = stochastic

            # Concatenate
            state_t = torch.cat([deterministic, stochastic], dim=-1)
            states.append(state_t)

        return torch.stack(states, dim=1)

    def imagine_step(
        self,
        prev_state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Imagine next state using prior (no observation).

        Used during imagination rollouts.

        Args:
            prev_state: (B, state_dim)
            action: (B, action_dim)

        Returns:
            next_state: (B, state_dim)
            info: Dict with distributions for loss computation
        """
        # Split state
        prev_det, prev_sto = torch.split(
            prev_state,
            [self.deterministic_size, self.stochastic_size],
            dim=-1
        )

        # Update deterministic
        deterministic = self._update_deterministic(prev_det, prev_sto, action)

        # Use prior (no observation)
        stochastic, prior_dist = self._compute_prior(deterministic)

        # Concatenate
        next_state = torch.cat([deterministic, stochastic], dim=-1)

        info = {
            'prior_dist': prior_dist,
            'deterministic': deterministic,
            'stochastic': stochastic
        }

        return next_state, info

    def representation_loss(
        self,
        features: torch.Tensor,
        sequence_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute KL divergence between posterior and prior.

        This is the key learning signal for RSSM.
        """
        is_sequence = features.dim() == 3

        if not is_sequence:
            # Single step - need sequence for KL
            features = features.unsqueeze(1)

        batch_size, seq_len, _ = features.shape

        # Get actions
        if sequence_context is not None and 'actions' in sequence_context:
            actions = sequence_context['actions']
            if actions.dim() == 2:
                actions = actions.unsqueeze(1)
        else:
            action_dim = self.action_dim or 1
            actions = torch.zeros(batch_size, seq_len, action_dim, device=self.device)

        # Initialize
        self.reset_hidden_state(batch_size)

        kl_losses = []
        for t in range(seq_len):
            features_t = features[:, t]
            action_t = actions[:, t] if t > 0 else torch.zeros_like(actions[:, 0])

            # Update deterministic (from previous step)
            deterministic = self._update_deterministic(
                self._hidden_state['deterministic'],
                self._hidden_state['stochastic'],
                action_t
            )

            # Compute both prior and posterior
            _, prior_dist = self._compute_prior(deterministic)
            stochastic, posterior_dist = self._compute_posterior(deterministic, features_t)

            # KL divergence: D_KL(posterior || prior)
            kl = torch.distributions.kl_divergence(posterior_dist, prior_dist)
            kl_losses.append(kl)

            # Update hidden state with posterior sample
            self._hidden_state['deterministic'] = deterministic
            self._hidden_state['stochastic'] = stochastic

        kl_loss = torch.stack(kl_losses, dim=1).mean(dim=1)  # (B,)

        # Apply KL strategy
        kl_loss = self._apply_kl_strategy(kl_loss)

        return {
            'representation_loss': kl_loss.mean(),
            'kl_loss': kl_loss.mean(),
            'kl_raw': torch.stack(kl_losses, dim=1).mean()
        }

    def _apply_kl_strategy(self, kl_loss: torch.Tensor) -> torch.Tensor:
        """Apply KL balancing strategy"""
        if self.kl_strategy == 'free_bits':
            # Free bits: max(kl, threshold)
            kl_loss = torch.maximum(kl_loss, torch.tensor(self.free_bits, device=self.device))

        elif self.kl_strategy == 'balanced':
            # Balanced KL (Dreamer v2)
            # Mix gradients from prior and posterior
            alpha = self.kl_balance_alpha
            kl_prior = kl_loss.detach()  # No grad through KL value
            kl_post = kl_loss.detach()   # No grad through KL value

            # Create gradients that balance prior and posterior
            # This encourages both to move towards each other
            kl_loss = alpha * kl_prior + (1 - alpha) * kl_post

        # Regular: no modification
        return kl_loss

    @property
    def representation_dim(self) -> int:
        return self._representation_dim

    def get_representation_info(self) -> Dict[str, Any]:
        info = super().get_representation_info()
        info.update({
            'stochastic_size': self.stochastic_size,
            'deterministic_size': self.deterministic_size,
            'total_dim': self.representation_dim,
            'kl_strategy': self.kl_strategy
        })
        return info
```

---

## Configuration Example

### PlaNet Configuration
```yaml
paradigm: world_model
implementation: world_model_mvp

algorithm:
  world_model_lr: 6e-4
  actor_lr: 0  # No actor in PlaNet
  critic_lr: 0  # No critic in PlaNet
  actor_updates_per_batch: 0
  critic_updates_per_batch: 0
  imagination_horizon: 12

  sequence_processing:
    mode: always
    component_inputs:
      representation_learner: sequence
      dynamics_model: sequence
    provide_padding_mask: true

components:
  encoder:
    type: cnn
    config:
      input_channels: 3
      feature_dim: 1024

  representation_learner:
    type: rssm_gaussian
    config:
      stochastic_size: 30
      deterministic_size: 200
      hidden_size: 200
      min_std: 0.1
      kl_strategy: regular
      feature_dim: 1024

  dynamics_model:
    type: stochastic_mlp
    config:
      hidden_dims: [200]

  planner:
    type: cem
    config:
      horizon: 12
      num_candidates: 1000
      num_elite: 100
```

### Dreamer v1 Configuration
```yaml
algorithm:
  world_model_lr: 6e-4
  actor_lr: 8e-5
  critic_lr: 8e-5
  imagination_horizon: 15

  # Same sequence processing as PlaNet

components:
  representation_learner:
    type: rssm_gaussian
    config:
      stochastic_size: 30
      deterministic_size: 200
      hidden_size: 200
      min_std: 0.1
      kl_strategy: regular

  # No planner for Dreamer (uses learned policy)
  planner: null
```

---

## Testing the Implementation

```python
# test_rssm.py
import torch
from src.components.representation_learners.rssm_gaussian import RSSMGaussianRepresentationLearner

# Create RSSM
config = {
    'stochastic_size': 30,
    'deterministic_size': 200,
    'hidden_size': 200,
    'action_dim': 4,
    'feature_dim': 256,
    'device': 'cuda'
}
rssm = RSSMGaussianRepresentationLearner(config).cuda()

# Test single-step encoding
features = torch.randn(32, 256).cuda()  # Batch of 32
context = {
    'action': torch.randn(32, 4).cuda()
}

state = rssm.encode(features, context)
print(f"State shape: {state.shape}")  # Should be (32, 230)

# Test sequence encoding
features_seq = torch.randn(32, 10, 256).cuda()  # Batch of 32, length 10
actions_seq = torch.randn(32, 10, 4).cuda()
context_seq = {'actions': actions_seq}

states_seq = rssm.encode(features_seq, context_seq)
print(f"States sequence shape: {states_seq.shape}")  # Should be (32, 10, 230)

# Test KL loss
losses = rssm.representation_loss(features_seq, context_seq)
print(f"KL loss: {losses['kl_loss'].item():.4f}")

# Test imagination
prev_state = states_seq[:, -1]
action = torch.randn(32, 4).cuda()
next_state, info = rssm.imagine_step(prev_state, action)
print(f"Imagined state shape: {next_state.shape}")
```

---

## Next: Discrete RSSM (Dreamer v2/v3)

The discrete version is similar but uses categorical distributions instead of Gaussians. Key changes:

```python
# Instead of:
stochastic ~ N(μ, σ)

# Use:
stochastic ~ Categorical(logits)  # 32 independent categoricals
stochastic_onehot = F.one_hot(stochastic, num_classes=32).flatten()
```

This will be in `rssm_discrete.py`.

---

## Integration with Your Paradigm

Your `BaseWorldModelParadigm` already handles:
- ✅ Sequence processing
- ✅ Context passing
- ✅ Loss aggregation
- ✅ Separate optimizers

**No changes needed!** Just:
1. Register RSSM component
2. Use appropriate config
3. Run training

The RSSM will automatically:
- Receive sequences via `sequence_processing`
- Compute KL loss via `representation_loss`
- Provide states for dynamics/policy/value

Your architecture is ready! 🎉
