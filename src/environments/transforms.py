"""
Composable Transform System for Environment Observations

This module provides a flexible, composable system for transforming environment observations.
Transforms can be stateless (e.g., to_grayscale, resize) or stateful (e.g., frame_stack).
Each environment instance maintains its own isolated state for stateful transforms.

Key features:
- Plug-and-play transform composition
- Stateful transforms with per-environment isolation
- Registry system for automatic discovery
- Configuration-driven transform pipelines
- Vectorization-friendly design
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformSpec:
    """Specification for a transform operation"""
    name: str
    transform_class: type
    stateful: bool = False


class BaseTransform(ABC):
    """Abstract base class for all transforms"""
    
    def __init__(self, **kwargs):
        """Initialize transform with parameters"""
        self.config = kwargs
        self.name = self.__class__.__name__
    
    @abstractmethod
    def apply(self, obs: np.ndarray) -> np.ndarray:
        """Apply transform to observation"""
        pass
    
    def reset_state(self):
        """Reset transform state (for stateful transforms)"""
        pass


class StatelessTransform(BaseTransform):
    """Base class for stateless transforms"""
    pass


class StatefulTransform(BaseTransform):
    """Base class for stateful transforms that need per-environment isolation"""
    
    @abstractmethod
    def reset_state(self):
        """Reset internal state of the transform"""
        pass


# Global transform registry
TRANSFORM_REGISTRY: Dict[str, TransformSpec] = {}


def register_transform(name: str, stateful: bool = False):
    """
    Decorator to register a transform in the global registry.
    
    Args:
        name: Name to register the transform under
        stateful: Whether the transform maintains internal state
    """
    def decorator(transform_class):
        spec = TransformSpec(
            name=name,
            transform_class=transform_class,
            stateful=stateful
        )
        TRANSFORM_REGISTRY[name] = spec
        logger.debug(f"Registered transform: {name} ({'stateful' if stateful else 'stateless'})")
        return transform_class
    return decorator


def get_transform(name: str, **kwargs) -> BaseTransform:
    """
    Get a transform instance by name.
    
    Args:
        name: Transform name
        **kwargs: Parameters to pass to transform constructor
        
    Returns:
        Transform instance
        
    Raises:
        KeyError: If transform name not found in registry
    """
    if name not in TRANSFORM_REGISTRY:
        available = list(TRANSFORM_REGISTRY.keys())
        raise KeyError(f"Transform '{name}' not found. Available transforms: {available}")
    
    spec = TRANSFORM_REGISTRY[name]
    return spec.transform_class(**kwargs)


# ============================================================================
# Image Processing Transforms
# ============================================================================

@register_transform("to_grayscale", stateful=False)
class ToGrayscale(StatelessTransform):
    """Convert RGB/RGBA image to grayscale"""
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[-1] in (3, 4):
            # RGB or RGBA to grayscale
            if obs.shape[-1] == 4:
                # Remove alpha channel first
                obs = obs[..., :3]
            return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 2:
            # Already grayscale
            return obs
        else:
            logger.warning(f"to_grayscale: unexpected observation shape {obs.shape}, returning unchanged")
            return obs


@register_transform("resize", stateful=False)
class Resize(StatelessTransform):
    """Resize image to specified dimensions"""
    
    def __init__(self, height: int = 84, width: int = 84, interpolation: str = 'linear', **kwargs):
        super().__init__(height=height, width=width, interpolation=interpolation, **kwargs)
        self.height = height
        self.width = width
        
        # Map string interpolation to OpenCV constants
        interp_map = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST, 
            'cubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos4': cv2.INTER_LANCZOS4
        }
        self.interpolation = interp_map.get(interpolation, cv2.INTER_LINEAR)
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim in (2, 3):
            return cv2.resize(obs, (self.width, self.height), interpolation=self.interpolation)
        else:
            logger.warning(f"resize: unexpected observation shape {obs.shape}, returning unchanged")
            return obs


@register_transform("scale_to_float", stateful=False)
class ScaleToFloat(StatelessTransform):
    """Scale uint8 pixel values to [0, 1] float range"""
    
    def __init__(self, scale: float = 255.0, **kwargs):
        super().__init__(scale=scale, **kwargs)
        self.scale = scale
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        if obs.dtype == np.uint8:
            return obs.astype(np.float32) / self.scale
        elif obs.dtype in (np.float32, np.float64):
            # Already float, check if scaling needed
            if obs.max() > 1.0:
                return obs.astype(np.float32) / self.scale
            return obs.astype(np.float32)
        else:
            logger.warning(f"scale_to_float: unexpected dtype {obs.dtype}, returning unchanged")
            return obs.astype(np.float32)


@register_transform("normalize", stateful=False)
class Normalize(StatelessTransform):
    """Normalize observations by mean and std"""
    
    def __init__(self, mean: Union[float, List[float]] = 0.0, std: Union[float, List[float]] = 1.0, **kwargs):
        super().__init__(mean=mean, std=std, **kwargs)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        return (obs - self.mean) / self.std


# ============================================================================
# Temporal Transforms (Stateful)
# ============================================================================

@register_transform("frame_stack", stateful=True)
class FrameStack(StatefulTransform):
    """Stack multiple consecutive frames along channel dimension"""
    
    def __init__(self, n_frames: int = 4, **kwargs):
        super().__init__(n_frames=n_frames, **kwargs)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        # Add frame to history
        self.frames.append(obs)
        
        # If we don't have enough frames yet, repeat the current frame
        while len(self.frames) < self.n_frames:
            self.frames.append(obs)
        
        # Stack frames along last axis
        if obs.ndim == 2:
            # Grayscale: (H, W) -> (H, W, n_frames)
            stacked = np.stack(list(self.frames), axis=-1)
        elif obs.ndim == 3:
            # RGB: (H, W, C) -> (H, W, C * n_frames)
            stacked = np.concatenate(list(self.frames), axis=-1)
        else:
            logger.warning(f"frame_stack: unexpected observation shape {obs.shape}")
            stacked = np.stack(list(self.frames), axis=-1)
        
        return stacked
    
    def reset_state(self):
        """Clear frame history"""
        self.frames.clear()


@register_transform("frame_skip", stateful=True)
class FrameSkip(StatefulTransform):
    """Skip frames by returning every nth frame"""
    
    def __init__(self, skip: int = 4, **kwargs):
        super().__init__(skip=skip, **kwargs)
        self.skip = skip
        self.counter = 0
        self.last_obs = None
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        self.counter += 1
        if self.counter % self.skip == 0:
            self.last_obs = obs
            return obs
        else:
            # Return last observation
            return self.last_obs if self.last_obs is not None else obs
    
    def reset_state(self):
        """Reset frame skip counter"""
        self.counter = 0
        self.last_obs = None


# ============================================================================
# Atari-Specific Transforms
# ============================================================================

@register_transform("atari_preprocessing", stateful=True)
class AtariPreprocessing(StatefulTransform):
    """
    Standard Atari preprocessing including noop actions, life loss handling, and frame skipping.
    This handles the game-logic preprocessing, not the visual transforms.
    """
    
    def __init__(self, noop_max: int = 30, frame_skip: int = 4, terminal_on_life_loss: bool = False, **kwargs):
        super().__init__(noop_max=noop_max, frame_skip=frame_skip, terminal_on_life_loss=terminal_on_life_loss, **kwargs)
        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.terminal_on_life_loss = terminal_on_life_loss
        self.reset_state()
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        # This transform primarily affects environment step logic, not observation processing
        # For observation transforms, it just passes through
        return obs
    
    def reset_state(self):
        """Reset Atari-specific state"""
        self.noop_actions_taken = 0
        self.lives = None


# ============================================================================
# Clipping and Reward Transforms
# ============================================================================

@register_transform("clip", stateful=False)
class Clip(StatelessTransform):
    """Clip observation values to specified range"""
    
    def __init__(self, low: float = -1.0, high: float = 1.0, **kwargs):
        super().__init__(low=low, high=high, **kwargs)
        self.low = low
        self.high = high
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        return np.clip(obs, self.low, self.high)


# ============================================================================
# Transform Pipeline
# ============================================================================

class TransformPipeline:
    """
    Orchestrates a sequence of transforms with proper state isolation.
    
    Each transform maintains its own state, enabling per-environment isolation
    in vectorized settings.
    """
    
    def __init__(self, transform_configs: List[Dict[str, Any]]):
        """
        Initialize transform pipeline from configuration.
        
        Args:
            transform_configs: List of transform configurations, each containing:
                - type: Transform name (must be registered)
                - **kwargs: Parameters for the transform
        """
        self.transforms: List[BaseTransform] = []
        self.stateful_transforms: List[BaseTransform] = []
        
        for config in transform_configs:
            if not isinstance(config, dict):
                raise ValueError(f"Transform config must be dict, got {type(config)}")
            
            config = config.copy()  # Don't modify original
            transform_type = config.pop('type')
            
            try:
                transform = get_transform(transform_type, **config)
                self.transforms.append(transform)
                
                # Track stateful transforms for reset
                if isinstance(transform, StatefulTransform):
                    self.stateful_transforms.append(transform)
                    
                logger.debug(f"Added transform: {transform_type} with config {config}")
                
            except Exception as e:
                logger.error(f"Failed to create transform '{transform_type}': {e}")
                raise
    
    def apply(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply all transforms in sequence.
        
        Args:
            obs: Input observation as numpy array
            
        Returns:
            Transformed observation as numpy array
        """
        for transform in self.transforms:
            try:
                obs = transform.apply(obs)
            except Exception as e:
                logger.error(f"Transform {transform.name} failed: {e}")
                raise
        
        return obs
    
    def reset_states(self):
        """Reset state of all stateful transforms"""
        for transform in self.stateful_transforms:
            transform.reset_state()
        logger.debug(f"Reset state for {len(self.stateful_transforms)} stateful transforms")
    
    def __len__(self) -> int:
        """Return number of transforms in pipeline"""
        return len(self.transforms)
    
    def __repr__(self) -> str:
        """Return string representation of pipeline"""
        transform_names = [t.name for t in self.transforms]
        return f"TransformPipeline({transform_names})"


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_transforms() -> Dict[str, Dict[str, Any]]:
    """
    List all available transforms with their specifications.
    
    Returns:
        Dictionary mapping transform names to their specs
    """
    return {
        name: {
            'class': spec.transform_class.__name__,
            'stateful': spec.stateful,
            'docstring': spec.transform_class.__doc__ or 'No documentation'
        }
        for name, spec in TRANSFORM_REGISTRY.items()
    }


def create_transform_pipeline(configs: List[Dict[str, Any]]) -> Optional[TransformPipeline]:
    """
    Create a transform pipeline from configuration list.
    
    Args:
        configs: List of transform configurations
        
    Returns:
        TransformPipeline instance or None if configs is empty
    """
    if not configs:
        return None
    
    return TransformPipeline(configs)


# ============================================================================
# Preset Configurations
# ============================================================================

# Common transform presets for different environment types
TRANSFORM_PRESETS = {
    'atari_vision': [
        {'type': 'atari_preprocessing', 'noop_max': 30, 'frame_skip': 4, 'terminal_on_life_loss': False},
        {'type': 'to_grayscale'},
        {'type': 'resize', 'height': 84, 'width': 84},
        {'type': 'scale_to_float'},
        {'type': 'frame_stack', 'n_frames': 4}
    ],
    'atari_vision_small': [
        {'type': 'atari_preprocessing', 'noop_max': 30, 'frame_skip': 4, 'terminal_on_life_loss': False},
        {'type': 'to_grayscale'},
        {'type': 'resize', 'height': 42, 'width': 42},
        {'type': 'scale_to_float'},
        {'type': 'frame_stack', 'n_frames': 4}
    ],
    'image_basic': [
        {'type': 'resize', 'height': 64, 'width': 64},
        {'type': 'scale_to_float'}
    ],
    'image_grayscale': [
        {'type': 'to_grayscale'},
        {'type': 'resize', 'height': 64, 'width': 64},
        {'type': 'scale_to_float'}
    ]
}


def expand_preset_configs(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expand preset configurations into individual transform configs.
    
    Args:
        configs: List of configurations that may contain presets
        
    Returns:
        Expanded list with presets replaced by individual transforms
    """
    expanded = []
    
    for config in configs:
        if config.get('type') in TRANSFORM_PRESETS:
            preset_name = config['type']
            logger.info(f"Expanding preset '{preset_name}'")
            expanded.extend(TRANSFORM_PRESETS[preset_name])
        else:
            expanded.append(config)
    
    return expanded


if __name__ == '__main__':
    # Example usage and testing
    print("Available transforms:")
    for name, info in list_available_transforms().items():
        print(f"  {name}: {info['class']} ({'stateful' if info['stateful'] else 'stateless'})")
    
    print("\nAvailable presets:")
    for name, configs in TRANSFORM_PRESETS.items():
        print(f"  {name}: {len(configs)} transforms")
