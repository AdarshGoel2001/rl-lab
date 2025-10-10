# Return Computers for World Models

## Overview

Return computers are optional, pluggable components that compute multi-step returns from trajectory sequences in the world model paradigm. Different world model algorithms have different requirements:

- **Dreamer**: Trains critic only on imagined rollouts → No real-data returns needed → Use `none`
- **MuZero**: Uses n-step bootstrapped returns for value targets → Use `n_step`
- **TD-MPC**: Uses TD(λ) returns for temporal difference learning → Use `td_lambda`
- **Offline methods**: May use Monte Carlo returns → Use `discounted`

## Architecture

### Registry-Based Design

Return computers follow the same registry pattern as other components:

```python
from src.utils.registry import register_return_computer

@register_return_computer("my_return_computer")
class MyReturnComputer(BaseReturnComputer):
    def compute_returns(self, rewards, dones, values=None, **kwargs):
        # Your implementation
        return computed_returns
```

### Base Interface

All return computers inherit from `BaseReturnComputer`:

```python
class BaseReturnComputer:
    def compute_returns(
        self,
        rewards: np.ndarray,  # Shape: (B, T)
        dones: np.ndarray,    # Shape: (B, T)
        values: Optional[np.ndarray] = None,  # Shape: (B, T)
        **kwargs
    ) -> np.ndarray:
        """Returns array with shape (B, T) or None"""
```

## Available Return Computers

### 1. None (Default for Dreamer)

**Type**: `"none"`

Returns `None` - no computation performed. Use for Dreamer-style algorithms that train the critic exclusively on imagined trajectories.

**Config**:
```yaml
buffer:
  type: world_model_sequence
  capacity: 100000
  sequence_length: 50
  # No return_computer specified → defaults to no returns
```

Or explicitly:
```yaml
buffer:
  type: world_model_sequence
  capacity: 100000
  sequence_length: 50
  return_computer: "none"  # Explicit specification
```

**When to use**: Dreamer, DreamerV2, DreamerV3

---

### 2. Discounted Returns (Monte Carlo)

**Type**: `"discounted"`

Computes pure discounted returns without bootstrapping:
```
G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^(T-t)*r_T
```

**Config**:
```yaml
buffer:
  type: world_model_sequence
  capacity: 100000
  sequence_length: 50
  return_computer:
    type: discounted
    config:
      gamma: 0.99  # Discount factor
```

**When to use**:
- Episodic tasks with complete trajectories
- Offline RL with full episodes
- When you want unbiased but high-variance estimates

**Notes**:
- Does NOT use value bootstrapping
- Resets at episode boundaries (done flags)
- High variance but unbiased

---

### 3. N-Step Returns (MuZero-style)

**Type**: `"n_step"`

Computes n-step bootstrapped returns:
```
G_t = r_t + γ*r_{t+1} + ... + γ^(n-1)*r_{t+n-1} + γ^n*V(s_{t+n})
```

**Config**:
```yaml
buffer:
  type: world_model_sequence
  capacity: 100000
  sequence_length: 50
  return_computer:
    type: n_step
    config:
      gamma: 0.997     # Discount factor
      n_step: 5        # Number of steps before bootstrapping
```

**When to use**:
- MuZero, MuZero Reanalyze
- Algorithms that benefit from multi-step targets
- When you want to balance bias-variance tradeoff

**Requirements**:
- Requires `values` in trajectory data
- The trainer must compute and store value estimates

**Bias-Variance Tradeoff**:
- `n_step=1`: TD(0) - low variance, high bias
- `n_step=∞`: Monte Carlo - high variance, low bias
- `n_step=5`: Common middle ground

---

### 4. TD(λ) Returns

**Type**: `"td_lambda"`

Computes TD(λ) returns using eligibility traces:
```
G_t^λ = (1-λ) * Σ_{n=1}^{∞} λ^{n-1} * G_t^{(n)}
```

This is a weighted average of all n-step returns.

**Config**:
```yaml
buffer:
  type: world_model_sequence
  capacity: 100000
  sequence_length: 50
  return_computer:
    type: td_lambda
    config:
      gamma: 0.99           # Discount factor
      lambda_coef: 0.95     # Lambda coefficient
      use_gae: false        # Whether to use GAE formulation
```

**When to use**:
- TD-MPC, TD3
- When you want smooth interpolation between TD(0) and Monte Carlo
- When you have good value estimates

**Parameters**:
- `lambda_coef=0.0`: Equivalent to TD(0) (one-step)
- `lambda_coef=1.0`: Equivalent to Monte Carlo
- `lambda_coef=0.95`: Common default (Dreamer uses this for imagined rollouts)
- `use_gae=true`: Use GAE formulation (advantage = δ + γλ*advantage)

**Requirements**:
- Requires `values` in trajectory data
- Optionally uses `next_values` (computed automatically if not provided)

---

## Configuration Examples

### Complete Dreamer Config (No Returns)

```yaml
paradigm: world_model

environment:
  name: CartPole-v1
  wrapper: gym

buffer:
  type: world_model_sequence
  capacity: 100000
  sequence_length: 50
  batch_size: 16
  # No return_computer → trains critic only on imagined rollouts

algorithm:
  gamma: 0.99
  world_model_lr: 3e-4
  actor_lr: 8e-5
  critic_lr: 8e-5

paradigm_config:
  imagination_length: 15
  lambda_return: 0.95  # Used for imagined rollouts, not real data
```

### MuZero-Style Config (N-Step Returns)

```yaml
paradigm: world_model

environment:
  name: Pong-v5
  wrapper: atari

buffer:
  type: world_model_sequence
  capacity: 500000
  sequence_length: 50
  batch_size: 32
  return_computer:
    type: n_step
    config:
      gamma: 0.997
      n_step: 5

algorithm:
  gamma: 0.997
  world_model_lr: 1e-4
  actor_lr: 1e-4
  critic_lr: 1e-4

paradigm_config:
  # MuZero uses planning, so might have planner config here
  use_real_value_loss: true  # Train critic on real data returns
```

### TD-MPC-Style Config (TD-Lambda)

```yaml
paradigm: world_model

environment:
  name: HalfCheetah-v2
  wrapper: gym

buffer:
  type: world_model_sequence
  capacity: 1000000
  sequence_length: 50
  batch_size: 64
  return_computer:
    type: td_lambda
    config:
      gamma: 0.99
      lambda_coef: 0.95
      use_gae: false

algorithm:
  gamma: 0.99
  world_model_lr: 3e-4
  actor_lr: 3e-4
  critic_lr: 3e-4

paradigm_config:
  imagination_length: 5  # TD-MPC uses shorter horizons
```

### Hybrid Config (Train on Both Real and Imagined)

```yaml
buffer:
  type: world_model_sequence
  capacity: 100000
  sequence_length: 50
  batch_size: 16
  return_computer:
    type: td_lambda
    config:
      gamma: 0.99
      lambda_coef: 0.95

paradigm_config:
  imagination_length: 15
  lambda_return: 0.95
  # System will compute value loss on BOTH:
  # 1. Real data returns (from return_computer)
  # 2. Imagined rollouts (from imagination)
  # Final value_loss = real_value_loss + imagined_value_loss
```

## Implementation Details

### How It Works

1. **Buffer Configuration**: Return computer is configured in buffer config
2. **Sampling**: When `buffer.sample()` is called:
   - Sequences are sampled as usual
   - If return computer is configured, `_compute_returns_for_batch()` is called
   - Returns are added to batch dict as `batch["returns"]`
3. **Training**: System checks for returns in batch:
   ```python
   returns_tensor = batch.get("returns")
   if returns_tensor is not None:
       real_value_loss = value_function.value_loss(latent, returns_tensor)
   ```

### Value Requirements

Some return computers require value estimates:
- `"none"`: No values needed ✓
- `"discounted"`: No values needed ✓
- `"n_step"`: Requires values ✗
- `"td_lambda"`: Requires values ✗

If values are required but not available, the return computer will raise an error.

### Storing Values in Trajectories

To use bootstrapping methods, the trainer must store value estimates:

```python
# In trainer collection loop
value_estimate = paradigm.get_value(obs)
trajectory = {
    'observations': obs,
    'actions': actions,
    'rewards': rewards,
    'dones': dones,
    'values': value_estimate.cpu().numpy(),  # Add this!
}
buffer.add(trajectory=trajectory)
```

## Creating Custom Return Computers

### Step 1: Implement the Class

```python
# src/components/world_models/return_computers/my_computer.py

import numpy as np
from .base import BaseReturnComputer
from ....utils.registry import register_return_computer

@register_return_computer("my_custom")
class MyCustomReturnComputer(BaseReturnComputer):
    def __init__(self, config):
        super().__init__(config)
        # Your custom parameters
        self.my_param = config.get("my_param", 1.0)

    def compute_returns(self, rewards, dones, values=None, **kwargs):
        # Your custom logic
        batch_size, time_steps = rewards.shape
        returns = np.zeros_like(rewards)

        # ... your computation ...

        return returns
```

### Step 2: Import in `__init__.py`

```python
# src/components/world_models/return_computers/__init__.py

from .my_computer import MyCustomReturnComputer

__all__ = [..., "MyCustomReturnComputer"]
```

### Step 3: Use in Config

```yaml
buffer:
  return_computer:
    type: my_custom
    config:
      my_param: 2.0
```

## Debugging

### Check if returns are being computed

Add logging in your trainer:

```python
batch = buffer.sample()
if "returns" in batch:
    print(f"Returns shape: {batch['returns'].shape}")
    print(f"Returns mean: {batch['returns'].mean()}")
else:
    print("No returns in batch")
```

### Verify return computer registration

```python
from src.utils.registry import list_registered_components

components = list_registered_components()
print(components['return_computers'])
# Should show: ['none', 'discounted', 'n_step', 'td_lambda', ...]
```

### Check return computation

```python
import numpy as np
from src.utils.registry import get_return_computer

# Create test data
rewards = np.random.randn(4, 10)  # 4 sequences, 10 timesteps
dones = np.zeros((4, 10))
dones[:, -1] = 1  # Episodes end at last step

# Test return computer
rc_cls = get_return_computer("discounted")
rc = rc_cls({"gamma": 0.99})
returns = rc.compute_returns(rewards, dones)

print(f"Returns shape: {returns.shape}")
print(f"Last return should equal last reward: {returns[0, -1]} == {rewards[0, -1]}")
```

## FAQ

**Q: Why does Dreamer not need returns?**

A: Dreamer trains the critic exclusively on imagined trajectories using lambda-returns computed during imagination. Real data is only used to train the world model components (encoder, dynamics, reward predictor).

**Q: Can I train on both real and imagined returns?**

A: Yes! If you configure a return computer, the system will compute:
```python
value_loss = real_value_loss + imagined_value_loss
```
This can help stabilize training in some cases.

**Q: Which n_step value should I use?**

A: Common values:
- MuZero: 5 or 10
- Start with 5 and tune based on your environment
- Longer n makes returns less biased but higher variance

**Q: What's the difference between TD(λ) and n-step?**

A:
- **N-step**: Uses exactly n steps then bootstraps
- **TD(λ)**: Weighted average of ALL n-step returns (smoother)
- TD(λ) with λ close to 1.0 is similar to large n-step

**Q: Do I need to modify the trainer?**

A: Only if you want to use bootstrapping methods (`n_step`, `td_lambda`). You'll need to store value estimates in trajectories. For `none` or `discounted`, no changes needed.

## Summary

| Algorithm | Return Computer | Values Required | Use Case |
|-----------|----------------|-----------------|----------|
| Dreamer | `none` | No | Train critic only on imagination |
| MuZero | `n_step` | Yes | Multi-step bootstrapped targets |
| TD-MPC | `td_lambda` | Yes | Smooth temporal difference learning |
| Offline RL | `discounted` | No | Full episode returns |
| Hybrid | `td_lambda` | Yes | Train on both real + imagined |

Choose based on your algorithm requirements and available data!
