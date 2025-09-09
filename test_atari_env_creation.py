#!/usr/bin/env python3
"""
Test Atari environment creation to isolate the hanging issue.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.environments.atari_wrapper import AtariEnvironment
from src.utils.config import Config, ConfigManager

def test_env_creation():
    print("Testing Atari environment creation...")
    
    config = {
        'game': 'ALE/Pong-v5',
        'wrapper': 'atari',
        'frame_skip': 4,
        'frame_stack': 4,
        'sticky_actions': 0.25,
        'noop_max': 30,
        'terminal_on_life_loss': False,
        'clip_rewards': True,
        'full_action_space': False,
        'num_environments': 1,
        'parallel_backend': None,
        'start_method': None
    }
    
    print(f"Creating environment with config: {config}")
    
    try:
        env = AtariEnvironment(config)
        print("✅ Environment created successfully")
        
        print("Testing environment reset...")
        obs = env.reset()
        print(f"✅ Environment reset successful, obs shape: {obs.shape}")
        
        print("Testing environment step...")
        action = 0  # No-op action
        next_obs, reward, done, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        print(f"  Next obs shape: {next_obs.shape}")
        print(f"  Info: {info}")
        
        env.close()
        print("✅ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment creation/test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_env_creation()
    if success:
        print("\n✅ All Atari environment tests passed!")
    else:
        print("\n❌ Atari environment tests failed!")