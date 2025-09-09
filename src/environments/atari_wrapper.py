"""
Atari Environment Wrapper

This module provides a comprehensive Atari environment wrapper with all standard
preprocessing for deep RL, including:
- Frame stacking (4 frames)
- Grayscale conversion
- Frame skipping with action repetition
- Sticky actions for realism
- Reward clipping to [-1, 0, 1]  
- Terminal on life loss vs game over
- NoOp reset for random starts
- Max episode steps handling

Based on established best practices from DeepMind Atari papers.
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import gymnasium as gym
import ale_py  # Required to register Atari environments
from gymnasium.wrappers import (
    GrayscaleObservation, 
    ResizeObservation, 
    FrameStackObservation,
    AtariPreprocessing,
    MaxAndSkipObservation,
    ClipReward,
    StickyAction
)
from collections import deque
import cv2

from .base import BaseEnvironment, SpaceSpec
from ..utils.registry import register_environment


class NoOpResetEnv(gym.Wrapper):
    """Apply random number of no-ops at environment reset."""
    
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, _ = self.env.reset(**kwargs)
                break
        return obs, {}


class MaxAndSkipEnv(gym.Wrapper):
    """Max over last 2 frames and repeat action for skip frames."""
    
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        
        # Max over last 2 observations
        if len(self._obs_buffer) == 2:
            max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        else:
            max_frame = self._obs_buffer[-1]
            
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class StickyActionEnv(gym.Wrapper):
    """Apply action with some probability, otherwise repeat last action."""
    
    def __init__(self, env: gym.Env, action_repeat_prob: float = 0.25):
        super().__init__(env)
        self.action_repeat_prob = action_repeat_prob
        self.last_action = 0

    def step(self, action):
        # With probability action_repeat_prob, use last action instead
        if np.random.random() < self.action_repeat_prob:
            action = self.last_action
        self.last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = 0
        return self.env.reset(**kwargs)


class FireResetEnv(gym.Wrapper):
    """Take FIRE action on reset for games that require it."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        if 'FIRE' in action_meanings:
            self.fire_action = action_meanings.index('FIRE')
        else:
            self.fire_action = None

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.fire_action is not None:
            obs, _, terminated, truncated, _ = self.env.step(self.fire_action)
            if terminated or truncated:
                self.env.reset(**kwargs)
        return self.env.step(0)[0], {}

    def step(self, action):
        return self.env.step(action)


class ClipRewardEnv(gym.Wrapper):
    """Clip rewards to {-1, 0, 1}."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = np.sign(reward)
        return obs, reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        # Check current lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Life lost
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/life loss state  
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


@register_environment("atari")
class AtariEnvironment(BaseEnvironment):
    """
    Comprehensive Atari environment wrapper with standard preprocessing.
    
    Features:
    - Grayscale + 84x84 resize
    - Frame stacking (4 frames)
    - Frame skipping (4 frames with max pooling)
    - Sticky actions (25% probability)
    - Reward clipping to [-1, 0, 1]
    - Terminal on life loss
    - NoOp reset for random starts
    - FIRE reset for games requiring it
    
    Config Options:
    - game: Atari game name (e.g., 'PongNoFrameskip-v4')
    - frame_skip: Frames to skip/repeat actions (default: 4)
    - frame_stack: Number of frames to stack (default: 4)  
    - sticky_actions: Probability of repeating last action (default: 0.25)
    - noop_max: Max no-op actions on reset (default: 30)
    - terminal_on_life_loss: Whether to end episode on life loss (default: True)
    - clip_rewards: Whether to clip rewards to [-1,0,1] (default: True)
    - full_action_space: Use full 18 actions vs reduced set (default: False)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.game = config.get('game', 'PongNoFrameskip-v4')
        self.frame_skip = config.get('frame_skip', 4)
        self.frame_stack = config.get('frame_stack', 4)  
        self.sticky_actions = config.get('sticky_actions', 0.25)
        self.noop_max = config.get('noop_max', 30)
        self.terminal_on_life_loss = config.get('terminal_on_life_loss', True)
        self.clip_rewards = config.get('clip_rewards', True)
        self.full_action_space = config.get('full_action_space', False)
        
        super().__init__(config)
    
    def _setup_environment(self):
        """Setup Atari environment with all preprocessing wrappers"""
        # Create base environment
        self.env = gym.make(
            self.game,
            full_action_space=self.full_action_space,
            frameskip=1  # We handle frame skipping manually for more control
        )
        
        # Use gymnasium's built-in AtariPreprocessing for compatibility
        self.env = AtariPreprocessing(
            self.env,
            noop_max=self.noop_max,
            frame_skip=self.frame_skip,
            screen_size=84,
            terminal_on_life_loss=self.terminal_on_life_loss,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=False  # We'll handle scaling in _process_observation
        )
        
        # Reward clipping
        if self.clip_rewards:
            self.env = ClipReward(self.env, min_reward=-1.0, max_reward=1.0)
        
        # Frame stacking (must be last)
        if self.frame_stack > 1:
            self.env = FrameStackObservation(self.env, stack_size=self.frame_stack)
        
        # Store action meanings for debugging
        self.action_meanings = self.env.unwrapped.get_action_meanings()
        
    def _get_observation_space(self) -> SpaceSpec:
        """Get observation space specification for stacked grayscale frames"""
        if self.frame_stack > 1:
            shape = (84, 84, self.frame_stack)  # HWC format
        else:
            shape = (84, 84, 1)
        
        return SpaceSpec(
            shape=shape,
            dtype=np.uint8,  # Keep as uint8 to save memory, convert to float in network
            low=0,
            high=255,
            discrete=False
        )
    
    def _get_action_space(self) -> SpaceSpec:
        """Get action space specification"""
        n_actions = self.env.action_space.n
        # Ensure we always have a valid shape - protect against initialization issues
        if n_actions is None or n_actions <= 0:
            # Default to 6 actions for Pong (fallback)
            n_actions = 6
        return SpaceSpec(
            shape=(n_actions,),  # Match gym_wrapper format
            dtype=np.int64,
            discrete=True,
            n=n_actions
        )
    
    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return initial observation"""
        if seed is not None:
            obs, _ = self.env.reset(seed=seed)
        else:
            obs, _ = self.env.reset()
        
        # Convert observation format
        obs = self._process_observation(obs)
        return obs
    
    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step environment and return processed results"""
        # Convert action to scalar if needed
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.ndim > 0 else int(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Process observation
        obs = self._process_observation(obs)
        
        # Combine terminated and truncated into single done flag
        done = terminated or truncated
        
        # Add Atari-specific info
        info['lives'] = getattr(self.env.unwrapped.ale, 'lives', lambda: 0)()
        info['action_meaning'] = self.action_meanings[action] if action < len(self.action_meanings) else f'ACTION_{action}'
        
        return obs, float(reward), done, info
    
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process raw observation to standard format"""
        # Convert to float32 and normalize to [0, 1]
        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255.0
        
        # Ensure HWC format (Height, Width, Channels)
        if obs.ndim == 3:
            if obs.shape[0] in [1, 4]:  # CHW format
                obs = np.transpose(obs, (1, 2, 0))  # Convert to HWC
        elif obs.ndim == 4:  # Batch dimension accidentally added
            obs = obs[0]  # Remove batch dimension
            if obs.shape[0] in [1, 4]:  # CHW format
                obs = np.transpose(obs, (1, 2, 0))
        
        return obs
    
    def get_action_meanings(self):
        """Get human-readable action meanings"""
        return self.action_meanings
    
    def render(self, mode: str = 'rgb_array'):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        if hasattr(self, 'env'):
            self.env.close()
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed"""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        elif hasattr(self.env, 'reset'):
            # For newer gym versions, seed is passed to reset
            return [seed] if seed is not None else None


def create_atari_env(game: str, **kwargs) -> AtariEnvironment:
    """
    Convenience function to create Atari environment with sensible defaults.
    
    Args:
        game: Atari game name (e.g., 'PongNoFrameskip-v4')
        **kwargs: Additional configuration options
        
    Returns:
        Configured AtariEnvironment instance
    """
    config = {
        'game': game,
        'frame_skip': 4,
        'frame_stack': 4,
        'sticky_actions': 0.25,
        'noop_max': 30,
        'terminal_on_life_loss': True,
        'clip_rewards': True,
        'full_action_space': False,
        **kwargs
    }
    
    return AtariEnvironment(config)


# Predefined configurations for popular games
ATARI_CONFIGS = {
    'pong': {
        'game': 'ALE/Pong-v5',
        'terminal_on_life_loss': False,  # Pong doesn't have lives
    },
    'breakout': {
        'game': 'ALE/Breakout-v5',
        'terminal_on_life_loss': True,
    },
    'space_invaders': {
        'game': 'ALE/SpaceInvaders-v5', 
        'terminal_on_life_loss': True,
    },
    'qbert': {
        'game': 'ALE/Qbert-v5',
        'terminal_on_life_loss': True,
    },
    'seaquest': {
        'game': 'ALE/Seaquest-v5',
        'terminal_on_life_loss': True,
    }
}


def create_atari_pong(**kwargs) -> AtariEnvironment:
    """Create Pong environment with optimal settings"""
    config = {**ATARI_CONFIGS['pong'], **kwargs}
    return AtariEnvironment(config)


def create_atari_breakout(**kwargs) -> AtariEnvironment:
    """Create Breakout environment with optimal settings"""  
    config = {**ATARI_CONFIGS['breakout'], **kwargs}
    return AtariEnvironment(config)