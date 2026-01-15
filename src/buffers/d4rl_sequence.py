"""D4RL Sequence Buffer for Diffusion Policy.

Loads D4RL HDF5 datasets and returns action sequences for diffusion policy training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import BaseBuffer


class D4RLSequenceBuffer(BaseBuffer):
    """Buffer that loads D4RL HDF5 files and returns action sequences.

    For diffusion policy training, we need to predict action sequences
    of length `horizon`. This buffer:
    1. Loads D4RL HDF5 data (observations, actions, terminals)
    2. Computes valid sequence start indices (avoiding episode boundaries)
    3. Returns batches with observations (B, obs_dim) and actions (B, H, action_dim)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        merged: Dict[str, Any] = dict(config or {})
        if kwargs:
            merged.update(kwargs)

        self.path = Path(merged.pop("path", "")) if "path" in merged else None
        self.horizon = int(merged.pop("horizon", 16))
        self.shuffle = bool(merged.pop("shuffle", True))
        self.seed = merged.pop("seed", None)
        self.device_override = merged.get("device")

        merged.setdefault("capacity", 0)
        merged.setdefault("batch_size", 256)
        merged.setdefault("device", "cpu")

        super().__init__(merged)

        self._rng: Optional[np.random.Generator] = None
        self._observations: Optional[np.ndarray] = None
        self._actions: Optional[np.ndarray] = None
        self._valid_starts: Optional[np.ndarray] = None
        self._sample_device: Optional[torch.device] = None

    def _setup_storage(self) -> None:
        """Initialize storage (actual loading happens in initialize)."""
        self._observations = None
        self._actions = None
        self._valid_starts = None
        self._size = 0

    def initialize(self, context: Any = None) -> None:
        """Load dataset and compute valid sequence starts."""
        device_hint = None
        if context is not None:
            device_hint = getattr(context, "device", None)
        if self.device_override is not None:
            self._sample_device = torch.device(self.device_override)
        elif device_hint is not None:
            self._sample_device = torch.device(device_hint)
        else:
            self._sample_device = self.device

        self._rng = np.random.default_rng(self.seed)

        if self.path is None or not str(self.path):
            raise ValueError("D4RLSequenceBuffer requires a 'path' configuration.")

        self._load_dataset(self.path)
        self._compute_valid_starts()

    def _load_dataset(self, path: Path) -> None:
        """Load D4RL HDF5 dataset."""
        if not path.exists():
            raise FileNotFoundError(f"D4RL dataset not found at '{path}'.")

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for D4RL datasets. Install: pip install h5py")

        with h5py.File(path, "r") as f:
            self._observations = f["observations"][:].astype(np.float32)
            self._actions = f["actions"][:].astype(np.float32)
            self._terminals = f["terminals"][:].astype(bool)

            # Also handle timeouts if present (both mark episode boundaries)
            if "timeouts" in f:
                timeouts = f["timeouts"][:].astype(bool)
                self._terminals = self._terminals | timeouts

        self._size = len(self._observations)

    def _compute_valid_starts(self) -> None:
        """Compute indices where we can start a sequence without crossing episode boundary.

        A start index i is valid if indices i, i+1, ..., i+horizon-1 are all
        within the same episode (no terminals in that range except possibly at the end).
        """
        N = len(self._observations)
        horizon = self.horizon

        if N < horizon:
            raise ValueError(f"Dataset size {N} smaller than horizon {horizon}")

        # Find episode boundaries
        episode_ends = np.where(self._terminals)[0]

        # Start with all indices as valid
        valid = np.ones(N - horizon + 1, dtype=bool)

        # Mark invalid: any start where the sequence would cross an episode boundary
        for end_idx in episode_ends:
            # If episode ends at end_idx, we can't start a sequence at
            # any index i where i+horizon-1 > end_idx (sequence crosses boundary)
            # i.e., i > end_idx - horizon + 1
            # So invalidate: [end_idx - horizon + 2, ..., end_idx]
            invalid_start = max(0, end_idx - horizon + 2)
            invalid_end = min(N - horizon + 1, end_idx + 1)
            if invalid_start < invalid_end:
                valid[invalid_start:invalid_end] = False

        self._valid_starts = np.where(valid)[0]

        if len(self._valid_starts) == 0:
            raise ValueError("No valid sequence starts found. Episodes may be too short.")

    def add(self, **kwargs: Any) -> None:
        """Raise error - this is a read-only buffer."""
        del kwargs
        raise RuntimeError("D4RLSequenceBuffer is read-only; cannot add new samples.")

    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Sample a batch of observation-action sequence pairs.

        Returns:
            Dict with:
                - observations: (B, obs_dim) - starting observation
                - actions: (B, horizon, action_dim) - action sequence
        """
        if self._valid_starts is None or len(self._valid_starts) == 0:
            raise RuntimeError("Buffer not initialized or no valid sequences.")

        effective_batch = batch_size or self.batch_size
        if effective_batch is None:
            raise ValueError("batch_size must be specified")

        # Sample random start indices
        if self.shuffle:
            idx = self._rng.choice(self._valid_starts, size=effective_batch, replace=True)
        else:
            # Sequential sampling
            start = int(self._rng.integers(0, max(1, len(self._valid_starts))))
            idx = self._valid_starts[start:start + effective_batch]
            if len(idx) < effective_batch:
                # Wrap around
                idx = np.concatenate([idx, self._valid_starts[:effective_batch - len(idx)]])

        # Extract observations at start indices
        observations = self._observations[idx]

        # Extract action sequences: for each start i, get actions[i:i+horizon]
        action_seqs = np.stack([
            self._actions[i:i + self.horizon] for i in idx
        ], axis=0)

        # Convert to tensors
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        action_tensor = torch.as_tensor(action_seqs, dtype=torch.float32)

        if self._sample_device is not None:
            obs_tensor = obs_tensor.to(self._sample_device)
            action_tensor = action_tensor.to(self._sample_device)

        return {
            "observations": obs_tensor,
            "actions": action_tensor,
        }

    def clear(self) -> None:
        """Clear buffer state."""
        self._observations = None
        self._actions = None
        self._valid_starts = None
        self._size = 0

    def ready(self) -> bool:
        """Check if buffer has valid sequences available."""
        return self._valid_starts is not None and len(self._valid_starts) >= self.batch_size

    def _save_buffer_state(self) -> Dict[str, Any]:
        """Save state (minimal for read-only buffer)."""
        return {}

    def _load_buffer_state(self, state: Dict[str, Any]) -> None:
        """Load state (no-op for read-only buffer)."""
        del state

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        stats = super().get_statistics()
        stats.update({
            "horizon": self.horizon,
            "valid_sequences": len(self._valid_starts) if self._valid_starts is not None else 0,
            "obs_dim": self._observations.shape[1] if self._observations is not None else 0,
            "action_dim": self._actions.shape[1] if self._actions is not None else 0,
        })
        return stats
