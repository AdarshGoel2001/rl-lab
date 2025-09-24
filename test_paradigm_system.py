#!/usr/bin/env python3
"""
Test Paradigm System

This script tests the new paradigm architecture by creating and running
simple configurations for both model-free and world model paradigms.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.registry import auto_import_modules, list_registered_components
from src.paradigms.factory import ComponentFactory


def test_registry_system():
    """Test that the registry system works and components are registered."""
    print("🔧 Testing registry system...")

    # Auto-import all modules to populate registries
    auto_import_modules()

    # List all registered components
    components = list_registered_components()

    print("📋 Registered components:")
    for comp_type, comp_list in components.items():
        if comp_list:  # Only show non-empty registries
            print(f"  {comp_type}: {comp_list}")

    # Verify we have the basic components we created
    expected_components = {
        'encoders': ['simple_mlp'],
        'representation_learners': ['identity'],
        'dynamics_models': ['deterministic_mlp'],
        'policy_heads': ['gaussian_mlp'],
        'value_functions': ['mlp_critic'],
        'paradigms': ['model_free', 'world_model']
    }

    for comp_type, expected in expected_components.items():
        for comp_name in expected:
            if comp_name not in components[comp_type]:
                raise ValueError(f"Missing component: {comp_type}.{comp_name}")

    print("✅ Registry system working correctly!")
    return True


def test_component_creation():
    """Test creating individual components."""
    print("\n🧩 Testing component creation...")

    # Test encoder
    encoder_config = {
        'input_dim': (4,),  # CartPole observation space
        'hidden_dims': [64, 64],
        'activation': 'relu',
        'device': 'cpu'
    }
    encoder = ComponentFactory.create_component('encoder', 'simple_mlp', encoder_config)
    print(f"  ✅ Created encoder: {encoder.__class__.__name__}")

    # Test representation learner
    repr_config = {
        'representation_dim': 64,
        'device': 'cpu'
    }
    repr_learner = ComponentFactory.create_component('representation_learner', 'identity', repr_config)
    print(f"  ✅ Created representation learner: {repr_learner.__class__.__name__}")

    # Test policy head
    policy_config = {
        'representation_dim': 64,
        'action_dim': 2,
        'hidden_dims': [64, 64],
        'discrete_actions': False,
        'device': 'cpu'
    }
    policy_head = ComponentFactory.create_component('policy_head', 'gaussian_mlp', policy_config)
    print(f"  ✅ Created policy head: {policy_head.__class__.__name__}")

    # Test value function
    value_config = {
        'representation_dim': 64,
        'hidden_dims': [64, 64],
        'device': 'cpu'
    }
    value_function = ComponentFactory.create_component('value_function', 'mlp_critic', value_config)
    print(f"  ✅ Created value function: {value_function.__class__.__name__}")

    # Test dynamics model
    dynamics_config = {
        'state_dim': 64,
        'action_dim': 2,
        'hidden_dims': [64, 64],
        'device': 'cpu'
    }
    dynamics_model = ComponentFactory.create_component('dynamics_model', 'deterministic_mlp', dynamics_config)
    print(f"  ✅ Created dynamics model: {dynamics_model.__class__.__name__}")

    print("✅ Component creation working correctly!")
    return True


def test_paradigm_creation():
    """Test creating complete paradigms."""
    print("\n🏗️  Testing paradigm creation...")

    # Test Model-Free Paradigm
    print("  Testing Model-Free Paradigm...")
    model_free_config = {
        "paradigm": "model_free",
        "encoder": {
            "type": "simple_mlp",
            "config": {
                "input_dim": (4,),
                "hidden_dims": [64, 64],
                "activation": "relu",
                "device": "cpu"
            }
        },
        "representation_learner": {
            "type": "identity",
            "config": {
                "representation_dim": 64,
                "device": "cpu"
            }
        },
        "policy_head": {
            "type": "gaussian_mlp",
            "config": {
                "representation_dim": 64,
                "action_dim": 2,
                "hidden_dims": [64, 64],
                "discrete_actions": False,
                "device": "cpu"
            }
        },
        "value_function": {
            "type": "mlp_critic",
            "config": {
                "representation_dim": 64,
                "hidden_dims": [64, 64],
                "device": "cpu"
            }
        },
        "paradigm_config": {
            "device": "cpu",
            "entropy_coef": 0.01
        }
    }

    # Validate config
    ComponentFactory.validate_paradigm_config(model_free_config)

    # Create paradigm
    model_free_paradigm = ComponentFactory.create_paradigm(model_free_config)
    print(f"    ✅ Created: {model_free_paradigm.__class__.__name__}")

    # Test World Model Paradigm
    print("  Testing World Model Paradigm...")
    world_model_config = {
        "paradigm": "world_model",
        "encoder": {
            "type": "simple_mlp",
            "config": {
                "input_dim": (4,),
                "hidden_dims": [64, 64],
                "device": "cpu"
            }
        },
        "representation_learner": {
            "type": "identity",
            "config": {
                "representation_dim": 64,
                "device": "cpu"
            }
        },
        "dynamics_model": {
            "type": "deterministic_mlp",
            "config": {
                "state_dim": 64,
                "action_dim": 2,
                "hidden_dims": [64, 64],
                "device": "cpu"
            }
        },
        "policy_head": {
            "type": "gaussian_mlp",
            "config": {
                "representation_dim": 64,
                "action_dim": 2,
                "hidden_dims": [64, 64],
                "device": "cpu"
            }
        },
        "value_function": {
            "type": "mlp_critic",
            "config": {
                "representation_dim": 64,
                "hidden_dims": [64, 64],
                "device": "cpu"
            }
        },
        "paradigm_config": {
            "device": "cpu",
            "imagination_length": 10
        }
    }

    # Validate config
    ComponentFactory.validate_paradigm_config(world_model_config)

    # Create paradigm
    world_model_paradigm = ComponentFactory.create_paradigm(world_model_config)
    print(f"    ✅ Created: {world_model_paradigm.__class__.__name__}")

    print("✅ Paradigm creation working correctly!")
    return model_free_paradigm, world_model_paradigm


def test_paradigm_forward_pass(paradigm, paradigm_name):
    """Test forward pass through a paradigm."""
    print(f"\n🚀 Testing {paradigm_name} forward pass...")

    # Create dummy observations (batch_size=2, obs_dim=4)
    batch_size = 2
    obs_dim = 4
    observations = torch.randn(batch_size, obs_dim)

    # Test forward pass
    with torch.no_grad():
        action_dist = paradigm.forward(observations)
        print(f"  ✅ Forward pass successful, action distribution type: {type(action_dist).__name__}")

        # Sample actions
        actions = paradigm.act(observations, deterministic=False)
        print(f"  ✅ Action sampling successful, action shape: {actions.shape}")

        # Get representations
        representations = paradigm.get_representation(observations)
        print(f"  ✅ Representation extraction successful, shape: {representations.shape}")

    return True


def test_paradigm_loss_computation(paradigm, paradigm_name):
    """Test loss computation for a paradigm."""
    print(f"\n📊 Testing {paradigm_name} loss computation...")

    # Create dummy batch
    batch_size = 4
    batch = {
        'observations': torch.randn(batch_size, 4),
        'actions': torch.randn(batch_size, 2),
        'rewards': torch.randn(batch_size),
        'next_observations': torch.randn(batch_size, 4),
        'dones': torch.zeros(batch_size, dtype=torch.bool),
        'returns': torch.randn(batch_size),
        'advantages': torch.randn(batch_size)
    }

    # Compute losses
    losses = paradigm.compute_loss(batch)
    print(f"  ✅ Loss computation successful")
    print(f"  📈 Loss components: {list(losses.keys())}")

    # Verify all losses are tensors with gradients
    for loss_name, loss_value in losses.items():
        if not isinstance(loss_value, torch.Tensor):
            raise ValueError(f"Loss {loss_name} is not a tensor: {type(loss_value)}")
        print(f"    {loss_name}: {loss_value.item():.4f}")

    return True


def test_checkpoint_system(paradigm, paradigm_name):
    """Test checkpoint save/load functionality."""
    print(f"\n💾 Testing {paradigm_name} checkpoint system...")

    # Save checkpoint
    checkpoint = paradigm.save_checkpoint()
    print(f"  ✅ Checkpoint saved, keys: {list(checkpoint.keys())}")

    # Modify paradigm state (change a parameter)
    original_param = None
    param_name = None
    for name, param in paradigm.named_parameters():
        if param.requires_grad:
            original_param = param.data.clone()
            param_name = name
            param.data.fill_(999.0)  # Change to distinctive value
            break

    if original_param is not None:
        print(f"  🔧 Modified parameter {param_name}")

        # Load checkpoint
        paradigm.load_checkpoint(checkpoint)
        print(f"  ✅ Checkpoint loaded")

        # Verify parameter was restored
        for name, param in paradigm.named_parameters():
            if name == param_name:
                if not torch.allclose(param.data, original_param):
                    raise ValueError("Checkpoint loading failed - parameter not restored")
                print(f"  ✅ Parameter {param_name} correctly restored")
                break

    return True


def main():
    """Run all tests."""
    print("🧪 Testing Paradigm Architecture System")
    print("=" * 50)

    try:
        # Test registry system
        test_registry_system()

        # Test component creation
        test_component_creation()

        # Test paradigm creation
        model_free_paradigm, world_model_paradigm = test_paradigm_creation()

        # Test paradigm functionality
        for paradigm, name in [(model_free_paradigm, "Model-Free"),
                              (world_model_paradigm, "World Model")]:
            test_paradigm_forward_pass(paradigm, name)
            test_paradigm_loss_computation(paradigm, name)
            test_checkpoint_system(paradigm, name)

        print("\n" + "=" * 50)
        print("🎉 All tests passed! Paradigm architecture is working correctly!")
        print("\n📝 Summary:")
        print("✅ Registry system functional")
        print("✅ Component creation working")
        print("✅ Paradigm creation working")
        print("✅ Forward passes working")
        print("✅ Loss computation working")
        print("✅ Checkpoint system working")
        print("\n🚀 Ready for integration with trainer!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)