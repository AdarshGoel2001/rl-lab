#!/usr/bin/env python3
"""
Test PPO Paradigm System

Tests the new architecture where PPO paradigm inherits from ModelFreeParadigm
and is registered as an algorithm, providing full trainer compatibility.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.registry import auto_import_modules, get_algorithm

class MockActionSpace:
    def __init__(self, shape, discrete=False):
        self.shape = shape
        self.discrete = discrete
        self.n = shape[0] if discrete else None

class MockObservationSpace:
    def __init__(self, shape):
        self.shape = shape

class MockNetwork:
    """Mock network that mimics the trainer's network initialization."""
    def __init__(self, network_type="mlp", hidden_dims=[64, 64], activation="tanh"):
        self.config = {
            'hidden_dims': hidden_dims,
            'activation': activation
        }
        self._network_type = network_type

    def __class__(self):
        return f"{self._network_type.upper()}Network"

def test_ppo_paradigm_registration():
    """Test that PPO paradigm is registered as algorithm."""
    print("✓ Test 1: PPO Paradigm Registration...")

    # Auto-import modules to populate registries
    auto_import_modules()

    try:
        # Check PPO is registered
        ppo_class = get_algorithm('ppo')
        print(f"  - PPO paradigm found: {ppo_class.__name__}")
        return True
    except Exception as e:
        print(f"  - Error: {e}")
        return False

def test_ppo_paradigm_creation():
    """Test creating PPO paradigm with trainer-compatible config."""
    print("✓ Test 2: PPO Paradigm Creation...")

    # Mock environments
    obs_space = MockObservationSpace((4,))
    action_space = MockActionSpace((2,), discrete=False)

    # Mock networks (same structure as trainer creates)
    networks = {
        'actor': MockNetwork("mlp", [64, 64], "tanh"),
        'critic': MockNetwork("mlp", [64, 64], "relu")
    }

    # Config matching trainer's format exactly
    config = {
        'networks': networks,
        'observation_space': obs_space,
        'action_space': action_space,
        'device': 'cpu',
        # PPO hyperparameters
        'lr': 3e-4,
        'clip_ratio': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'ppo_epochs': 2,
        'minibatch_size': 32,
        'normalize_advantages': True,
        'clip_value_loss': True,
        'max_grad_norm': 0.5
    }

    try:
        ppo_class = get_algorithm('ppo')
        ppo = ppo_class(config)
        print(f"  - PPO paradigm created: {ppo.__class__.__name__}")
        return ppo
    except Exception as e:
        print(f"  - Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trainer_compatibility(ppo):
    """Test that PPO paradigm has all methods trainer expects."""
    print("✓ Test 3: Trainer Compatibility...")

    # Required methods for trainer compatibility
    required_methods = [
        'get_action_and_value',  # Main action sampling
        'act',                   # Deterministic action for evaluation
        'update',                # Training step
        'train',                 # Set training mode
        'eval'                   # Set evaluation mode
    ]

    for method in required_methods:
        if not hasattr(ppo, method):
            print(f"  - Missing method: {method}")
            return False
        print(f"  - Has method: {method}")

    # Required attributes
    required_attrs = [
        'networks',              # For bootstrap value computation
        'device',                # Device compatibility
        'step'                   # Training step tracking
    ]

    for attr in required_attrs:
        if not hasattr(ppo, attr):
            print(f"  - Missing attribute: {attr}")
            return False
        print(f"  - Has attribute: {attr}")

    # Test specific trainer requirements
    assert 'critic' in ppo.networks, "Missing networks['critic'] for bootstrap values"
    print("  - Has networks['critic'] for bootstrap values")

    return True

def test_forward_pass(ppo):
    """Test forward pass methods."""
    print("✓ Test 4: Forward Pass...")

    batch_size = 3
    obs = torch.randn(batch_size, 4)

    try:
        # Test get_action_and_value (main trainer method)
        actions, log_probs, values = ppo.get_action_and_value(obs)

        print(f"  - get_action_and_value: actions {actions.shape}, log_probs {log_probs.shape}, values {values.shape}")

        # Verify shapes
        assert actions.shape == (batch_size, 2), f"Expected actions shape ({batch_size}, 2), got {actions.shape}"
        assert log_probs.shape == (batch_size,), f"Expected log_probs shape ({batch_size},), got {log_probs.shape}"
        assert values.shape == (batch_size,), f"Expected values shape ({batch_size},), got {values.shape}"

        # Test deterministic action (for evaluation)
        det_actions = ppo.act(obs, deterministic=True)
        print(f"  - act(deterministic=True): {det_actions.shape}")
        assert det_actions.shape == (batch_size, 2), f"Expected det_actions shape ({batch_size}, 2), got {det_actions.shape}"

        # Test stochastic action
        stoch_actions = ppo.act(obs, deterministic=False)
        print(f"  - act(deterministic=False): {stoch_actions.shape}")
        assert stoch_actions.shape == (batch_size, 2), f"Expected stoch_actions shape ({batch_size}, 2), got {stoch_actions.shape}"

        return True

    except Exception as e:
        print(f"  - Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_modes(ppo):
    """Test training/eval mode switching."""
    print("✓ Test 5: Training/Eval Modes...")

    try:
        # Test training mode
        ppo.train()
        training_states = []
        for name, component in [('encoder', ppo.encoder),
                               ('policy', ppo.policy_head),
                               ('value', ppo.value_function)]:
            training_states.append((name, component.training))
            print(f"  - {name} training mode: {component.training}")

        # All should be in training mode
        assert all(state for _, state in training_states), "Some components not in training mode"

        # Test eval mode
        ppo.eval()
        eval_states = []
        for name, component in [('encoder', ppo.encoder),
                              ('policy', ppo.policy_head),
                              ('value', ppo.value_function)]:
            eval_states.append((name, component.training))
            print(f"  - {name} eval mode: {not component.training}")

        # All should be in eval mode
        assert not any(state for _, state in eval_states), "Some components still in training mode"

        # Back to training
        ppo.train()

        return True

    except Exception as e:
        print(f"  - Error: {e}")
        return False

def test_update_method(ppo):
    """Test PPO update with multi-epoch training."""
    print("✓ Test 6: Update Method...")

    batch_size = 64

    # Create realistic training batch
    batch = {
        'observations': torch.randn(batch_size, 4),
        'actions': torch.randn(batch_size, 2),
        'old_log_probs': torch.randn(batch_size),
        'advantages': torch.randn(batch_size),
        'returns': torch.randn(batch_size),
        'old_values': torch.randn(batch_size)
    }

    try:
        initial_step = ppo.step
        metrics = ppo.update(batch)
        final_step = ppo.step

        print(f"  - Update completed, step: {initial_step} -> {final_step}")

        # Check required metrics
        required_metrics = ['policy_loss', 'value_loss', 'entropy_loss', 'total_loss', 'grad_norm']
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            print(f"  - {metric}: {metrics[metric]:.6f}")

        # Step should have incremented
        assert final_step == initial_step + 1, f"Step not incremented: {initial_step} -> {final_step}"

        return True

    except Exception as e:
        print(f"  - Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bootstrap_values(ppo):
    """Test bootstrap value computation (used by trainer)."""
    print("✓ Test 7: Bootstrap Values...")

    obs = torch.randn(1, 4)

    try:
        # Test trainer's bootstrap value computation
        critic_network = ppo.networks['critic']

        # Encode observations like trainer does
        features = ppo.encoder(obs)
        representations = ppo.representation_learner.encode(features)
        bootstrap_value = critic_network(representations)

        print(f"  - Bootstrap value shape: {bootstrap_value.shape}")
        print(f"  - Bootstrap value: {bootstrap_value.item():.6f}")

        assert bootstrap_value.shape == (1, 1), f"Expected bootstrap value shape (1, 1), got {bootstrap_value.shape}"

        return True

    except Exception as e:
        print(f"  - Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests for PPO paradigm system."""
    print("🧪 Testing PPO Paradigm System")
    print("=" * 50)

    tests_passed = 0
    total_tests = 7

    try:
        # Test 1: Registration
        if test_ppo_paradigm_registration():
            tests_passed += 1
        else:
            print("❌ PPO registration failed - cannot continue")
            return False

        # Test 2: Creation
        ppo = test_ppo_paradigm_creation()
        if ppo is not None:
            tests_passed += 1
        else:
            print("❌ PPO creation failed - cannot continue")
            return False

        # Test 3: Trainer compatibility
        if test_trainer_compatibility(ppo):
            tests_passed += 1

        # Test 4: Forward pass
        if test_forward_pass(ppo):
            tests_passed += 1

        # Test 5: Training modes
        if test_training_modes(ppo):
            tests_passed += 1

        # Test 6: Update method
        if test_update_method(ppo):
            tests_passed += 1

        # Test 7: Bootstrap values
        if test_bootstrap_values(ppo):
            tests_passed += 1

        print("\n" + "=" * 50)
        print(f"Tests passed: {tests_passed}/{total_tests}")

        if tests_passed == total_tests:
            print("🎉 All tests passed! PPO paradigm system ready for trainer!")
            print("\n📝 Summary:")
            print("✅ PPO paradigm registered as 'ppo' algorithm")
            print("✅ Full trainer compatibility")
            print("✅ Forward pass working")
            print("✅ Training/eval modes working")
            print("✅ Multi-epoch PPO update working")
            print("✅ Bootstrap value computation working")
            print("\n🚀 Trainer can now use PPO paradigm with:")
            print("  algorithm:")
            print("    name: 'ppo'")
            print("    # ... ppo hyperparameters")
            return True
        else:
            print(f"❌ {total_tests - tests_passed} tests failed")
            return False

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)