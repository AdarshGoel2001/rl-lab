"""Scaffold wrapper for ViZDoom scenarios (e.g., Take Cover) used in World Models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseEnvironment, SpaceSpec


class VizDoomWorldModelWrapper(BaseEnvironment):
    """Placeholder wrapper that will encapsulate ViZDoom + scenario logic."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        defaults = {
            "name": "vizdoom_take_cover",
            "scenario_path": None,  # TODO: point to .wad / .cfg assets
            "frame_skip": 4,
            "resolution": (120, 160),  # H, W ordering
            "color_mode": "grayscale",
            "obs_dtype": np.uint8,
            "normalize_obs": False,
            "action_map": None,  # TODO: describe discrete action buttons (strafe, fire, etc.)
            "max_episode_steps": 2100,
        }
        merged = dict(defaults)
        merged.update(config or {})
        super().__init__(merged, **kwargs)
        self.game = None  # type: ignore[assignment]
        self._available_actions: list[list[int]] = []

    # ------------------------------------------------------------------
    # BaseEnvironment hooks
    # ------------------------------------------------------------------
    def _setup_environment(self) -> None:
        """Initialize vizdoom.DoomGame with the requested scenario."""
        # TODO: import vizdoom here, configure screen format/resolution, load scenario, etc.
        raise NotImplementedError("VizDoomWorldModelWrapper._setup_environment homework pending.")

    def _get_observation_space(self) -> SpaceSpec:
        """Return the observation tensor spec (C x H x W or H x W x C)."""
        height, width = self.config["resolution"]
        channels = 1 if self.config.get("color_mode", "grayscale") == "grayscale" else 3
        shape = (height, width, channels)
        return SpaceSpec(shape=shape, dtype=np.uint8, low=0, high=255)

    def _get_action_space(self) -> SpaceSpec:
        """Expose either discrete action indices or multi-binary button vectors."""
        # TODO: Determine if you're using discrete action indices (len(action_map)) or multi-binary of buttons
        return SpaceSpec(shape=(len(self._available_actions) or 3,), dtype=np.float32, discrete=True, n=len(self._available_actions) or 3)

    def _reset_environment(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset the DoomGame instance and return the first frame."""
        # TODO: call self.game.new_episode(), maybe set seed via self.game.set_seed
        raise NotImplementedError("VizDoomWorldModelWrapper._reset_environment homework pending.")

    def _step_environment(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Advance the simulator using either discrete or button-based actions."""
        # TODO: convert action index -> Doom button list, step frame_skip times, collect reward/done/info
        raise NotImplementedError("VizDoomWorldModelWrapper._step_environment homework pending.")

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _configure_actions(self) -> None:
        """Translate config['action_map'] into ViZDoom-friendly button lists."""
        # TODO: populate self._available_actions so controller knows the action_dim
        raise NotImplementedError("VizDoomWorldModelWrapper._configure_actions homework pending.")

    def _convert_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize/reformat raw ViZDoom frame to match the ConvVAE input."""
        # TODO: add resizing, channel ordering, dtype conversions here
        return frame


__all__ = ["VizDoomWorldModelWrapper"]
