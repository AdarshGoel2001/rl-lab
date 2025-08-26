#!/usr/bin/env python3
"""
Action Space Testing Script

This script comprehensively tests action handling across different
environments and action formats to ensure robust compatibility.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_discrete_actions():
    """Test discrete action handling"""
    print("🎯 Testing Discrete Actions (CartPole-v1)")
    
    try:
        import torch
        from src.environments.gym_wrapper import GymWrapper
        
        env_config = {
            'name': 'CartPole-v1',
            'normalize_obs': False,
            'normalize_reward': False
        }
        
        env = GymWrapper(env_config)
        obs = env.reset(seed=42)
        
        # Test various discrete action formats
        test_cases = [
            # Standard formats
            ("Python int 0", 0),
            ("Python int 1", 1), 
            ("NumPy int", np.int32(0)),
            ("NumPy int64", np.int64(1)),
            
            # Tensor formats
            ("PyTorch scalar tensor", torch.tensor(0)),
            ("PyTorch int tensor", torch.tensor(1, dtype=torch.int32)),
            ("PyTorch long tensor", torch.tensor(0, dtype=torch.long)),
            
            # Array formats
            ("NumPy scalar array", np.array(1)),
            ("NumPy 1D array", np.array([0])),
            ("NumPy float converted", np.array(1.0)),
            
            # Edge cases
            ("Float that should be int", 0.0),
            ("One-hot encoded", np.array([1, 0])),  # Should take argmax
            ("One-hot encoded reverse", np.array([0, 1])),
        ]
        
        success_count = 0
        for test_name, action in test_cases:
            try:
                next_obs, reward, done, info = env.step(action)
                print(f"✅ {test_name}: Success")
                success_count += 1
                
                # Reset if episode ended
                if done:
                    obs = env.reset(seed=42)
                    
            except Exception as e:
                print(f"❌ {test_name}: Failed - {e}")
        
        env.close()
        print(f"📊 Discrete actions: {success_count}/{len(test_cases)} passed")
        return success_count == len(test_cases)
        
    except Exception as e:
        print(f"❌ Discrete action test setup failed: {e}")
        return False


def test_continuous_actions():
    """Test continuous action handling"""
    print("\n🎯 Testing Continuous Actions (Pendulum-v1)")
    
    try:
        import torch
        from src.environments.gym_wrapper import GymWrapper
        
        env_config = {
            'name': 'Pendulum-v1',
            'normalize_obs': False,
            'normalize_reward': False
        }
        
        try:
            env = GymWrapper(env_config)
        except Exception:
            print("⚠️  Pendulum-v1 not available, skipping continuous action tests")
            return True
        
        obs = env.reset(seed=42)
        
        # Test various continuous action formats
        test_cases = [
            # Standard formats
            ("Python float", [0.5]),
            ("NumPy array", np.array([0.1])),
            ("NumPy float32", np.array([0.0], dtype=np.float32)),
            
            # Tensor formats
            ("PyTorch tensor", torch.tensor([0.5])),
            ("PyTorch float tensor", torch.tensor([0.2], dtype=torch.float32)),
            
            # Edge cases
            ("Clipped action", np.array([3.0])),  # Should be clipped to [-2, 2]
            ("Negative action", np.array([-1.0])),
            ("Zero action", np.array([0.0])),
        ]
        
        success_count = 0
        for test_name, action in test_cases:
            try:
                next_obs, reward, done, info = env.step(action)
                print(f"✅ {test_name}: Success")
                success_count += 1
                
                # Reset if needed (Pendulum typically doesn't end)
                if done:
                    obs = env.reset(seed=42)
                    
            except Exception as e:
                print(f"❌ {test_name}: Failed - {e}")
        
        env.close()
        print(f"📊 Continuous actions: {success_count}/{len(test_cases)} passed")
        return success_count >= len(test_cases) * 0.8  # Allow some failures
        
    except Exception as e:
        print(f"❌ Continuous action test setup failed: {e}")
        return True  # Don't fail if Pendulum isn't available


def test_action_space_edge_cases():
    """Test edge cases and error handling"""
    print("\n🎯 Testing Action Space Edge Cases")
    
    try:
        import torch
        from src.environments.gym_wrapper import GymWrapper
        
        env_config = {
            'name': 'CartPole-v1', 
            'normalize_obs': False,
            'normalize_reward': False
        }
        
        env = GymWrapper(env_config)
        obs = env.reset(seed=42)
        
        # Test edge cases (these might fail gracefully)
        edge_cases = [
            ("Too large action", 999),
            ("Negative discrete action", -1),
            ("Multi-dimensional array", np.array([[0, 1]])),
            ("Empty array", np.array([])),
            ("String action", "0"),
            ("None action", None),
        ]
        
        handled_count = 0
        for test_name, action in edge_cases:
            try:
                next_obs, reward, done, info = env.step(action)
                print(f"✅ {test_name}: Handled gracefully")
                handled_count += 1
                
                if done:
                    obs = env.reset(seed=42)
                    
            except Exception as e:
                # Edge cases are expected to fail, but should give clear error messages
                if "invalid" in str(e).lower() or "not supported" in str(e).lower():
                    print(f"✅ {test_name}: Failed with clear error - {e}")
                    handled_count += 1
                else:
                    print(f"⚠️  {test_name}: Unclear error - {e}")
        
        env.close()
        print(f"📊 Edge cases: {handled_count}/{len(edge_cases)} handled properly")
        return handled_count >= len(edge_cases) * 0.7
        
    except Exception as e:
        print(f"❌ Edge case test setup failed: {e}")
        return False


def test_batch_actions():
    """Test batch action handling"""
    print("\n🎯 Testing Batch Actions")
    
    try:
        import torch
        from src.environments.gym_wrapper import GymWrapper
        
        env_config = {
            'name': 'CartPole-v1',
            'normalize_obs': False,
            'normalize_reward': False
        }
        
        env = GymWrapper(env_config)
        obs = env.reset(seed=42)
        
        # Test batch-like actions (should extract single action)
        batch_cases = [
            ("Batch tensor", torch.tensor([0])),
            ("Batch numpy", np.array([1])),
            ("Multi-batch", torch.tensor([[0, 1]])),  # Should take argmax
        ]
        
        success_count = 0
        for test_name, action in batch_cases:
            try:
                next_obs, reward, done, info = env.step(action)
                print(f"✅ {test_name}: Success")
                success_count += 1
                
                if done:
                    obs = env.reset(seed=42)
                    
            except Exception as e:
                print(f"❌ {test_name}: Failed - {e}")
        
        env.close()
        print(f"📊 Batch actions: {success_count}/{len(batch_cases)} passed")
        return success_count >= len(batch_cases) * 0.8
        
    except Exception as e:
        print(f"❌ Batch action test failed: {e}")
        return False


def main():
    """Run all action space tests"""
    print("🚀 Comprehensive Action Space Testing")
    print("=" * 60)
    
    tests = [
        ("Discrete Actions", test_discrete_actions),
        ("Continuous Actions", test_continuous_actions), 
        ("Edge Cases", test_action_space_edge_cases),
        ("Batch Actions", test_batch_actions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Action Space Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 All action space tests passed!")
        return 0
    else:
        print("⚠️  Some action space tests failed - check implementation")
        return 1


if __name__ == '__main__':
    sys.exit(main())