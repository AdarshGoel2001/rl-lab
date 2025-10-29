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

from typing import Dict, Any, Tuple, Optional, Union, List
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        config = dict(config or {})
        if kwargs:
            config.update(kwargs)

        self.game = config.get('game', 'PongNoFrameskip-v4')
        self.frame_skip = config.get('frame_skip', 4)
        self.frame_stack = config.get('frame_stack', 4)
        self.sticky_actions = config.get('sticky_actions', 0.25)
        self.noop_max = config.get('noop_max', 30)
        self.terminal_on_life_loss = config.get('terminal_on_life_loss', True)
        self.clip_rewards = config.get('clip_rewards', True)
        self.full_action_space = config.get('full_action_space', False)
        self.num_environments = int(config.get('num_environments', config.get('num_envs', 1)))
        self.parallel_backend = config.get('parallel_backend', 'sync')
        self.start_method = config.get('start_method', None)

        # Vectorization flags (consumed by BaseEnvironment)
        self.is_vectorized = self.num_environments > 1
        self.num_envs = self.num_environments if self.is_vectorized else 1

        # These will be populated during setup
        self._single_obs_shape: Optional[Tuple[int, ...]] = None
        self._single_obs_dtype: Optional[np.dtype] = None
        self._single_action_n: Optional[int] = None
        self._single_action_shape: Optional[Tuple[int, ...]] = None

        super().__init__(config)
    
    def _setup_environment(self):
        """Setup Atari environment with all preprocessing wrappers"""
        def build_single_env() -> gym.Env:
            env = gym.make(
                self.game,
                full_action_space=self.full_action_space,
                frameskip=1  # We handle frame skipping manually for more control
            )

            env = AtariPreprocessing(
                env,
                noop_max=self.noop_max,
                frame_skip=self.frame_skip,
                screen_size=84,
                terminal_on_life_loss=self.terminal_on_life_loss,
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=False
            )

            if self.clip_rewards:
                env = ClipReward(env, min_reward=-1.0, max_reward=1.0)

            if self.frame_stack > 1:
                env = FrameStackObservation(env, stack_size=self.frame_stack)

            return env

        # Cache action meanings (safe for vectorized envs)
        probe_env = gym.make(
            self.game,
            full_action_space=self.full_action_space,
            frameskip=1
        )
        try:
            self.action_meanings = probe_env.unwrapped.get_action_meanings()
        finally:
            probe_env.close()

        if self.is_vectorized:
            try:
                from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
            except ImportError as e:
                raise ImportError("gymnasium installation is required for vectorized Atari environments") from e

            def make_env_fn(seed_offset: int = 0):
                def _init():
                    env = build_single_env()
                    if self.start_method == 'spawn':
                        # Ensure deterministic seeding when using subprocesses
                        env.reset(seed=(self.config.get('seed', None) or 0) + seed_offset)
                    return env
                return _init

            env_fns = [make_env_fn(i) for i in range(self.num_envs)]
            use_async = self.parallel_backend in ('async', 'auto')

            if use_async and self.parallel_backend != 'sync':
                self.env = AsyncVectorEnv(env_fns, shared_memory=False)
            else:
                self.env = SyncVectorEnv(env_fns)

            self._single_obs_shape = self.env.single_observation_space.shape
            self._single_obs_dtype = getattr(self.env.single_observation_space, 'dtype', np.uint8)
            self._single_action_shape = self.env.single_action_space.shape if hasattr(self.env.single_action_space, 'shape') else ()
            self._single_action_n = getattr(self.env.single_action_space, 'n', None)

            # Track per-environment episode statistics for logging
            self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
            self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
            self._episode_counts = np.zeros(self.num_envs, dtype=np.int64)
        else:
            self.env = build_single_env()
            obs_space = getattr(self.env, 'observation_space', None)
            action_space = getattr(self.env, 'action_space', None)
            self._single_obs_shape = obs_space.shape if obs_space is not None else (84, 84, self.frame_stack)
            self._single_obs_dtype = getattr(obs_space, 'dtype', np.uint8)
            self._single_action_shape = getattr(action_space, 'shape', ())
            self._single_action_n = getattr(action_space, 'n', None)
        
    def _get_observation_space(self) -> SpaceSpec:
        """Get observation space specification for stacked grayscale frames"""
        stack_size = self.frame_stack if self.frame_stack > 1 else 1

        if self._single_obs_shape is not None:
            shape = self._single_obs_shape
            # Ensure frame dimension is last to match downstream networks
            if len(shape) == 3 and shape[0] in [1, 4] and shape[-1] not in [1, 4]:
                shape = (shape[1], shape[2], shape[0])
        else:
            shape = (84, 84, stack_size)

        return SpaceSpec(
            shape=shape,
            dtype=np.float32,
            low=0.0,
            high=1.0,
            discrete=False
        )
    
    def _get_action_space(self) -> SpaceSpec:
        """Get action space specification"""
        n_actions = self._single_action_n
        if n_actions is None or n_actions <= 0:
            n_actions = 6
        return SpaceSpec(
            shape=(n_actions,),
            dtype=np.int64,
            discrete=True,
            n=n_actions
        )
    
    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return initial observation"""
        if self.is_vectorized:
            if seed is not None:
                seeds = [seed + i for i in range(self.num_envs)]
                obs, _ = self.env.reset(seed=seeds)
            else:
                obs, _ = self.env.reset()

            # Reset running episode statistics when the vectorized env resets
            if hasattr(self, '_episode_returns'):
                self._episode_returns.fill(0.0)
                self._episode_lengths.fill(0)
        else:
            if seed is not None:
                obs, _ = self.env.reset(seed=seed)
            else:
                obs, _ = self.env.reset()

        obs = self._process_observation(obs)
        return obs
    
    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step environment and return processed results"""
        if self.is_vectorized:
            actions = np.asarray(action)
            if actions.ndim == 2 and actions.shape[1] == 1:
                actions = actions.squeeze(-1)
            obs, rewards, terminated, truncated, infos = self.env.step(actions)
            obs = self._process_observation(obs)
            rewards = np.asarray(rewards, dtype=np.float32)
            dones = np.logical_or(terminated, truncated)
            infos = self._normalize_vector_infos(infos)

            # Update running episode statistics for downstream logging
            if hasattr(self, '_episode_returns'):
                self._episode_returns += rewards
                self._episode_lengths += 1

            # Annotate infos with action meanings when possible
            if isinstance(actions, np.ndarray) and actions.shape == (self.num_envs,):
                for info, act in zip(infos, actions):
                    if isinstance(info, dict):
                        info.setdefault('action_meaning', self._action_meaning(int(act)))

            # Surface episodic statistics so vectorized trainers can log returns/lengths
            if hasattr(self, '_episode_returns'):
                for idx, (done, info) in enumerate(zip(dones, infos)):
                    if not isinstance(info, dict):
                        continue

                    # Always expose running totals (helpful for streaming dashboards)
                    info.setdefault('episode_return', float(self._episode_returns[idx]))
                    info.setdefault('episode_length', int(self._episode_lengths[idx]))

                    if done:
                        episode_return = float(self._episode_returns[idx])
                        episode_length = int(self._episode_lengths[idx])
                        info['episode_return'] = episode_return
                        info['episode_length'] = episode_length
                        info.setdefault('episode', {'r': episode_return, 'l': episode_length})
                        info['episode_count'] = int(self._episode_counts[idx] + 1)

                        # Reset trackers for next episode and increment counter
                        self._episode_returns[idx] = 0.0
                        self._episode_lengths[idx] = 0
                        self._episode_counts[idx] += 1

            return obs, rewards.astype(float), dones, infos

        # Convert action to scalar if needed for single-environment case
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.ndim > 0 else int(action)

        obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs = self._process_observation(obs)
        done = terminated or truncated

        if isinstance(info, dict):
            # Add Atari-specific info for debugging convenience
            if 'lives' not in info:
                info['lives'] = getattr(self.env.unwrapped.ale, 'lives', lambda: 0)()
            info['action_meaning'] = self._action_meaning(int(action))

        return obs, float(reward), done, info
    
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process raw observation to standard format"""
        obs_array = np.asarray(obs)

        if obs_array.dtype == np.uint8:
            obs_array = obs_array.astype(np.float32) / 255.0

        if self.is_vectorized:
            # Expected format: (num_envs, H, W, C)
            if obs_array.ndim == 4 and obs_array.shape[1] in [1, 4]:
                obs_array = np.transpose(obs_array, (0, 2, 3, 1))
            return obs_array

        if obs_array.ndim == 3 and obs_array.shape[0] in [1, 4]:
            obs_array = np.transpose(obs_array, (1, 2, 0))
        elif obs_array.ndim == 4 and obs_array.shape[1] in [1, 4] and obs_array.shape[0] == 1:
            obs_array = np.transpose(obs_array[0], (1, 2, 0))

        return obs_array

    def _normalize_vector_infos(self, infos: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize Gymnasium vector info outputs to list-of-dicts."""
        if isinstance(infos, list):
            return [dict(info) if isinstance(info, dict) else {} for info in infos]

        if isinstance(infos, dict):
            normalized = []
            keys = list(infos.keys())
            for idx in range(self.num_envs):
                entry = {}
                for key in keys:
                    value = infos[key]
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > idx:
                        entry[key] = value[idx]
                normalized.append(entry)
            return normalized

        return [{} for _ in range(self.num_envs)]

    def _action_meaning(self, action: int) -> str:
        if 0 <= action < len(self.action_meanings):
            return self.action_meanings[action]
        return f'ACTION_{action}'
    
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
