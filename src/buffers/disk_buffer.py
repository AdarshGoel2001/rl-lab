"""Disk-based sequence buffer for world model training.

This module provides a buffer that can write, append, or read trajectories
from disk - enabling collect-once train-many workflows.

Modes:
    - WRITE: No existing data → creates new dataset
    - APPEND: Existing data + add() called → appends to dataset
    - READ: Existing data + sample() called first → read-only

Path Resolution:
    All relative paths are resolved against the ORIGINAL working directory
    (before Hydra changes it), not the Hydra output directory. This ensures
    datasets are created in predictable locations like project_root/datasets/.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import BaseBuffer


def _get_original_cwd() -> Path:
    """Get original working directory before Hydra changes it.

    Hydra changes the working directory to the output folder. This function
    returns the original directory so paths are resolved correctly.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        # If Hydra is active, use its recorded original cwd
        return Path(HydraConfig.get().runtime.cwd)
    except (ImportError, ValueError):
        # Hydra not available or not initialized - use current directory
        return Path.cwd()


class DiskBuffer(BaseBuffer):
    """Unified disk buffer supporting write, append, and read modes.

    Storage format:
        dataset_dir/
        ├── metadata.json      # Shapes, dtypes, episode info
        ├── observations.bin   # Raw binary data
        ├── actions.bin
        ├── rewards.bin
        ├── dones.bin
        └── index.npy          # Precomputed valid sequence starts

    Usage scenarios:

    1. First run (no dataset_path, auto-generates):
        buffer:
          sequence_length: 15
        phases:
          - name: collect
            workflow_hooks: [collect]
          - name: train
            workflow_hooks: [update_world_model]

    2. Subsequent run (load existing):
        buffer:
          dataset_path: datasets/auto_20241218_143022
        phases:
          - name: train  # No collect phase
            workflow_hooks: [update_world_model]

    3. Append more data to existing:
        buffer:
          dataset_path: datasets/auto_20241218_143022
        phases:
          - name: collect_more  # Has collect phase → appends
            workflow_hooks: [collect]
          - name: train
            workflow_hooks: [update_world_model]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the disk buffer.

        Args:
            config: Configuration dict with:
                - dataset_path: Path to dataset (optional - auto-generates if missing)
                - sequence_length: Length of sequences to sample (default: 16)
                - sequence_stride: Stride for index building (default: 8)
                - batch_size: Number of sequences per batch (default: 32)
                - buffer_flush_size: Steps before flushing to disk (default: 1000)
                - device: PyTorch device (default: cpu)
        """
        merged = dict(config or {})
        merged.update(kwargs)
        config = merged

        # Handle dataset_path - auto-generate if not provided
        # IMPORTANT: Resolve relative paths against original cwd, not Hydra output dir
        original_cwd = _get_original_cwd()
        dataset_path = config.get("dataset_path")

        if dataset_path is None or dataset_path == "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dataset_path = original_cwd / f"datasets/auto_{timestamp}"
            self._auto_generated_path = True
        else:
            path = Path(dataset_path)
            # If path is relative, resolve against original cwd
            if not path.is_absolute():
                self.dataset_path = (original_cwd / path).resolve()
            else:
                self.dataset_path = path.resolve()
            self._auto_generated_path = False

        self.sequence_length = int(config.get("sequence_length", 16))
        self.sequence_stride = int(config.get("sequence_stride", 8))
        self.buffer_flush_size = int(config.get("buffer_flush_size", 1000))

        # Check for existing data
        self.metadata_path = self.dataset_path / "metadata.json"
        self._has_existing_data = self.metadata_path.exists()

        # Mode: "write", "append", "read", or "pending" (decided on first add/sample)
        if self._has_existing_data:
            self._mode = "pending"  # Will become "append" or "read"
            self._load_existing_metadata()
        else:
            self._mode = "write"
            self._init_fresh_state()

        # File handles and memmaps
        self._files: Dict[str, Any] = {}
        self._memmaps: Dict[str, np.memmap] = {}
        self._write_buffer: Dict[str, List[np.ndarray]] = {}
        self._write_buffer_count = 0

        # Index for sampling
        self.index: Optional[np.ndarray] = None

        # Dtypes - only override if explicitly set in config
        # (If loading existing data, dtypes come from metadata)
        if not self._has_existing_data:
            self.obs_dtype = np.dtype(config.get("obs_dtype", "uint8"))
            self.action_dtype = np.dtype(config.get("action_dtype", "float32"))
        else:
            # Keep dtypes from metadata, but allow explicit override
            if "obs_dtype" in config:
                self.obs_dtype = np.dtype(config["obs_dtype"])
            if "action_dtype" in config:
                self.action_dtype = np.dtype(config["action_dtype"])

        # Set dummy capacity for base class
        config["capacity"] = 1000000
        super().__init__(config)

        # If we have existing data and it's read-only scenario, open for reading
        # But we wait until sample() to confirm it's read mode

    def _init_fresh_state(self) -> None:
        """Initialize state for a fresh dataset."""
        self.total_steps = 0
        self.num_envs = 1
        self.obs_shape: Optional[tuple] = None
        self.action_shape: Optional[tuple] = None
        self.episode_boundaries: List[List[int]] = []
        self.shapes: Dict[str, list] = {}
        self.dtypes: Dict[str, str] = {}

    def _load_existing_metadata(self) -> None:
        """Load metadata from existing dataset."""
        with open(self.metadata_path) as f:
            metadata = json.load(f)

        self.total_steps = metadata["total_steps"]
        self.num_envs = metadata["num_envs"]
        self.shapes = metadata["shapes"]
        self.dtypes = metadata["dtypes"]
        self.episode_boundaries = metadata.get("episode_boundaries", [])

        self.obs_shape = tuple(self.shapes["observations"])
        self.action_shape = tuple(self.shapes["actions"])
        self.obs_dtype = np.dtype(self.dtypes["observations"])
        self.action_dtype = np.dtype(self.dtypes["actions"])

    def initialize(self, context: Any = None) -> None:
        """Initialize buffer with context (called by orchestrator).

        Extracts shapes from environment if in write mode.
        """
        if self._mode in ("read", "pending") and self._has_existing_data:
            return  # Shapes come from metadata

        if context is None:
            return

        env = getattr(context, "train_environment", None)
        if env is None:
            return

        # Get observation shape (obs_space.shape is per-env, doesn't include batch dim)
        obs_space = getattr(env, "observation_space", None)
        if obs_space is not None and self.obs_shape is None:
            self.obs_shape = tuple(obs_space.shape)

        # Get action shape (action_space.shape is per-env, doesn't include batch dim)
        action_space = getattr(env, "action_space", None)
        if action_space is not None and self.action_shape is None:
            self.action_shape = tuple(action_space.shape)

        # Get num_envs
        self.num_envs = int(getattr(env, "num_envs", 1) or 1)
        if not self.episode_boundaries:
            self.episode_boundaries = [[] for _ in range(self.num_envs)]

    def _setup_storage(self) -> None:
        """No-op - storage is on disk."""
        pass

    # -------------------------------------------------------------------------
    # Write / Append Mode
    # -------------------------------------------------------------------------

    def add(self, trajectory: Optional[Dict[str, np.ndarray]] = None, **kwargs) -> None:
        """Add trajectory data to disk.

        If existing data: switches to append mode.
        If no existing data: continues write mode.

        Args:
            trajectory: Dict with 'observations', 'actions', 'rewards', 'dones'
                       Each has shape (T, num_envs, ...)
        """
        # Handle mode transition
        if self._mode == "pending":
            self._mode = "append"
            self._open_for_appending()
        elif self._mode == "read":
            # Already finalized and in read mode, ignore new data
            return

        if trajectory is None:
            trajectory = kwargs

        if not trajectory or "observations" not in trajectory:
            return

        # Open files on first write (for fresh datasets)
        if not self._files:
            self._open_for_writing()

        # Shape inference and validation happens in _add_trajectory
        self._add_trajectory(trajectory)

    def _open_for_writing(self) -> None:
        """Open files for fresh writing.

        Files are opened lazily when first trajectory arrives.
        """
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        print(f"DiskBuffer: Creating new dataset at {self.dataset_path}")
        # Files opened lazily per-key in _add_trajectory
        self._write_buffer_count = 0

    def _open_for_appending(self) -> None:
        """Open files for appending to existing dataset.

        Validates that incoming data matches existing dataset schema.
        Files opened lazily per-key when trajectory arrives.
        """
        print(f"DiskBuffer: Appending to existing dataset at {self.dataset_path}")
        print(f"  Existing steps: {self.total_steps}, envs: {self.num_envs}")

        # Delete old index (will rebuild after appending)
        index_path = self.dataset_path / "index.npy"
        if index_path.exists():
            index_path.unlink()

        # Files opened lazily per-key in _add_trajectory
        self._write_buffer_count = 0

    def _add_trajectory(self, trajectory: Dict[str, np.ndarray]) -> None:
        """Add trajectory data step by step - handles arbitrary keys."""
        # Convert all values to numpy arrays
        arrays = {key: np.asarray(value) for key, value in trajectory.items()}
        if not arrays:
            return

        # Get time and env dimensions from first key
        first_key = next(iter(arrays))
        T = arrays[first_key].shape[0]

        # Infer num_envs from first trajectory if not set
        if self.obs_shape is None and "observations" in arrays:
            obs = arrays["observations"]
            if obs.ndim >= 3:
                inferred_num_envs = obs.shape[1]
                inferred_obs_shape = obs.shape[2:]
            else:
                inferred_num_envs = 1
                inferred_obs_shape = ()

            # If appending, validate against existing metadata
            if self._mode == "append":
                if "observations" in self.shapes:
                    expected_obs_shape = tuple(self.shapes["observations"])
                    expected_num_envs = self.num_envs

                    if inferred_obs_shape != expected_obs_shape:
                        raise ValueError(
                            f"Observation shape mismatch when appending: "
                            f"incoming {inferred_obs_shape} vs existing {expected_obs_shape}"
                        )
                    if inferred_num_envs != expected_num_envs:
                        raise ValueError(
                            f"Number of environments mismatch when appending: "
                            f"incoming {inferred_num_envs} vs existing {expected_num_envs}"
                        )

            self.obs_shape = inferred_obs_shape
            self.num_envs = inferred_num_envs
            self.episode_boundaries = [[] for _ in range(self.num_envs)]

        # Infer and store shapes/dtypes for all keys (before flushing clears buffer)
        for key, array in arrays.items():
            # Infer dtype
            if key not in self.dtypes:
                self.dtypes[key] = array.dtype.name

            # Infer shape (per-step, per-env shape)
            if key not in self.shapes:
                # array shape is (T, num_envs, ...), per-env shape is [2:]
                if array.ndim >= 3:
                    self.shapes[key] = list(array.shape[2:])
                else:
                    self.shapes[key] = []

        # Open files lazily for each key
        for key in arrays.keys():
            if key not in self._files:
                filepath = self.dataset_path / f"{key}.bin"
                mode = "ab" if self._mode == "append" else "wb"
                self._files[key] = open(filepath, mode)
            if key not in self._write_buffer:
                self._write_buffer[key] = []

        # Append timestep by timestep
        for t in range(T):
            for key, array in arrays.items():
                # Extract step (handles (T, num_envs, ...) indexing)
                step_data = array[t] if array.ndim > 0 else array
                self._write_buffer[key].append(step_data)

            self._write_buffer_count += 1

            # Track episode boundaries from dones
            if "dones" in arrays:
                done_t = arrays["dones"][t]
                for env_idx in range(self.num_envs):
                    if done_t[env_idx]:
                        self.episode_boundaries[env_idx].append(self.total_steps)

            self.total_steps += 1

            if self._write_buffer_count >= self.buffer_flush_size:
                self._flush_write_buffer()

    def _flush_write_buffer(self) -> None:
        """Flush buffered data to disk - handles arbitrary keys."""
        if self._write_buffer_count == 0:
            return

        for key, buffer_list in self._write_buffer.items():
            if buffer_list:
                data = np.stack(buffer_list, axis=0)
                self._files[key].write(data.tobytes())
                self._write_buffer[key] = []

        self._write_buffer_count = 0

        for f in self._files.values():
            f.flush()

    def finalize(self) -> None:
        """Finalize writing and switch to read mode.

        Called automatically on first sample() if in write/append mode.
        Can also be called explicitly.
        """
        if self._mode not in ("write", "append"):
            return

        if not self._files and self.total_steps == 0:
            return  # Nothing to finalize

        # Flush remaining data
        self._flush_write_buffer()

        # Close file handles
        for f in self._files.values():
            f.close()
        self._files = {}

        # Write metadata (shapes/dtypes already inferred in _add_trajectory)
        metadata = {
            "env_name": "collected",
            "total_steps": self.total_steps,
            "num_envs": self.num_envs,
            "shapes": self.shapes,
            "dtypes": self.dtypes,
            "episode_boundaries": self.episode_boundaries,
            # Store index params - allows rebuilding if they change
            "index_params": {
                "sequence_length": self.sequence_length,
                "sequence_stride": self.sequence_stride,
            },
        }

        temp_path = self.dataset_path / "metadata.json.tmp"
        with open(temp_path, "w") as f:
            json.dump(metadata, f, indent=2)
        temp_path.rename(self.metadata_path)

        # Update internal state
        self.shapes = metadata["shapes"]
        self.dtypes = metadata["dtypes"]

        # Build index
        self._build_index()

        print(f"DiskBuffer: Dataset finalized at {self.dataset_path}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Num envs: {self.num_envs}")
        total_episodes = sum(len(b) for b in self.episode_boundaries)
        print(f"  Total episodes: {total_episodes}")

        # Switch to read mode
        self._mode = "read"
        self._open_for_reading()

    def _build_index(self) -> None:
        """Build index of valid sequence starting points.

        NOTE: This allows sequences to cross episode boundaries. The workflow
        is responsible for resetting hidden states when done=True is encountered
        within a sequence. This maximizes GPU utilization by avoiding masking.
        """
        index = []

        for env_idx in range(self.num_envs):
            # Simple stride across ALL steps, ignoring episode breaks
            for start in range(0, self.total_steps - self.sequence_length + 1, self.sequence_stride):
                index.append((env_idx, start))

        self.index = np.array(index, dtype=np.int64)
        np.save(self.dataset_path / "index.npy", self.index)
        print(f"  Index entries: {len(self.index)}")

    # -------------------------------------------------------------------------
    # Read Mode
    # -------------------------------------------------------------------------

    def _open_for_reading(self) -> None:
        """Open dataset for reading via memory maps - handles arbitrary keys."""
        # Load index
        index_path = self.dataset_path / "index.npy"
        if index_path.exists():
            self.index = np.load(index_path)
        else:
            self._build_index()

        # Open memory maps for all keys in metadata
        for key in self.shapes.keys():
            filepath = self.dataset_path / f"{key}.bin"
            if not filepath.exists():
                print(f"Warning: {key}.bin not found, skipping")
                continue

            dtype = np.dtype(self.dtypes[key])
            shape_suffix = tuple(self.shapes[key])

            if shape_suffix:
                full_shape = (self.total_steps, self.num_envs) + shape_suffix
            else:
                full_shape = (self.total_steps, self.num_envs)

            self._memmaps[key] = np.memmap(
                filepath,
                dtype=dtype,
                mode="r",
                shape=full_shape,
            )

        self._size = len(self.index) if self.index is not None else 0

    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Sample random sequences from disk.

        If in write/append mode, auto-finalizes first.
        If in pending mode (existing data, no add called), switches to read mode.

        Args:
            batch_size: Number of sequences (default: self.batch_size)

        Returns:
            Dict with 'observations', 'actions', 'rewards', 'dones'
            Each tensor has shape (batch_size, sequence_length, ...)
        """
        # Handle mode transitions
        if self._mode == "pending":
            # Existing data, sample called first → read-only mode
            self._mode = "read"
            self._open_for_reading()
        elif self._mode in ("write", "append"):
            # Was writing, now sampling → finalize first
            if self.total_steps > 0:
                self.finalize()
            else:
                raise RuntimeError("No data collected yet")

        if self.index is None or len(self.index) == 0:
            raise ValueError("No valid sequences in dataset")

        if batch_size is None:
            batch_size = self.batch_size

        # Sample random indices
        replace = len(self.index) < batch_size
        sample_indices = np.random.choice(len(self.index), size=batch_size, replace=replace)

        # Gather sequences
        batch = self._gather_sequences(sample_indices)

        batch["sequence_length"] = torch.tensor(self.sequence_length, device=self.device)
        batch["sequence_stride"] = torch.tensor(self.sequence_stride, device=self.device)

        return batch

    def _gather_sequences(self, sample_indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """Gather sequences for given index positions."""
        sequences = {key: [] for key in self._memmaps.keys()}

        for idx in sample_indices:
            env_idx, start = self.index[idx]
            end = start + self.sequence_length

            for key, mmap in self._memmaps.items():
                seq = np.array(mmap[start:end, env_idx])
                sequences[key].append(seq)

        batch = {}
        for key, seqs in sequences.items():
            stacked = np.stack(seqs, axis=0)
            batch[key] = self.to_tensor(stacked)

        return batch

    # -------------------------------------------------------------------------
    # Buffer Interface
    # -------------------------------------------------------------------------

    def ready(self) -> bool:
        """Check if buffer has enough data for sampling."""
        if self._mode in ("write", "append"):
            # Estimate sequences: (steps - seq_len + 1) / stride * num_envs
            # Simplified: need enough steps to form batch_size sequences
            if self.total_steps < self.sequence_length:
                return False
            available_per_env = (self.total_steps - self.sequence_length) // self.sequence_stride + 1
            available_total = available_per_env * self.num_envs
            return available_total >= self.batch_size

        if self._mode == "pending":
            # Have existing data, same estimate
            if self.total_steps < self.sequence_length:
                return False
            available_per_env = (self.total_steps - self.sequence_length) // self.sequence_stride + 1
            available_total = available_per_env * self.num_envs
            return available_total >= self.batch_size

        return self.index is not None and len(self.index) >= self.batch_size

    def clear(self) -> None:
        """Not supported - delete dataset directory manually."""
        raise RuntimeError("Cannot clear disk buffer - delete the dataset directory instead")

    def _save_buffer_state(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            "dataset_path": str(self.dataset_path),
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
            "mode": self._mode,
            "total_steps": self.total_steps,
        }

    def _load_buffer_state(self, state: Dict[str, Any]) -> None:
        """No-op - state is on disk."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        stats = super().get_statistics()
        stats.update({
            "dataset_path": str(self.dataset_path),
            "mode": self._mode,
            "total_steps": self.total_steps,
            "num_envs": self.num_envs,
            "num_sequences": len(self.index) if self.index is not None else 0,
            "sequence_length": self.sequence_length,
        })
        return stats

    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, '_files') and self._files:
            try:
                self._flush_write_buffer()
                for f in self._files.values():
                    f.close()
            except Exception:
                pass

        if hasattr(self, '_memmaps'):
            for mmap in self._memmaps.values():
                try:
                    del mmap
                except Exception:
                    pass

    def __repr__(self) -> str:
        return (
            f"DiskBuffer("
            f"path={self.dataset_path}, "
            f"mode={self._mode}, "
            f"steps={self.total_steps}, "
            f"envs={self.num_envs})"
        )


# Backward compatibility alias
PersistentSequenceBuffer = DiskBuffer
