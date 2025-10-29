"""Sequence buffer tailored for world-model training workloads."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import BaseBuffer

from ..components.world_models.return_computers.base import BaseReturnComputer


class WorldModelSequenceBuffer(BaseBuffer):
    """Stores vectorised experience and returns contiguous sequences on demand."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        merged_config: Dict[str, Any] = dict(config or {})
        if kwargs:
            merged_config.update(kwargs)
        config = merged_config
        config.setdefault("gamma", 0.99)
        config.setdefault("sequence_length", 16)
        config.setdefault("sequence_stride", None)
        config.setdefault("num_envs", 1)
        config.setdefault("pad_end_of_episode", False)

        self.gamma = float(config["gamma"])
        self.sequence_length = int(config["sequence_length"])
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        stride = config.get("sequence_stride")
        if stride is None:
            stride = self.sequence_length
        self.sequence_stride = max(1, int(stride))

        self.num_envs = int(config["num_envs"])
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive for world-model buffer")

        self.pad_end_of_episode = bool(config.get("pad_end_of_episode", False))

        # Return computer setup (optional, for MuZero/TD-MPC style algorithms)
        self.return_computer = self._resolve_return_computer(config.get("return_computer"))

        capacity = int(config.get("capacity", 10000))
        config["capacity"] = max(capacity, self.sequence_length * self.num_envs)

        super().__init__(config)

    # ------------------------------------------------------------------
    # BaseBuffer hooks
    # ------------------------------------------------------------------
    def _setup_storage(self) -> None:
        self.capacity_per_env = max(
            self.sequence_length,
            math.ceil(self.capacity / max(1, self.num_envs)),
        )
        self._env_buffers: List[Dict[str, deque]] = []
        for _ in range(self.num_envs):
            self._env_buffers.append(defaultdict(lambda: deque(maxlen=self.capacity_per_env)))

    def add(self, **kwargs: Any) -> None:
        if "trajectory" not in kwargs:
            raise ValueError("WorldModelSequenceBuffer expects trajectory additions")

        trajectory = kwargs["trajectory"]
        if not isinstance(trajectory, dict):
            raise TypeError("trajectory must be a dict of numpy arrays")

        arrays = {key: np.asarray(value) for key, value in trajectory.items()}
        if not arrays:
            return

        first_key = next(iter(arrays))
        time_dim = arrays[first_key].shape[0]
        env_dim = arrays[first_key].shape[1] if arrays[first_key].ndim >= 2 else 1

        if env_dim > self.num_envs:
            for _ in range(env_dim - self.num_envs):
                self._env_buffers.append(defaultdict(lambda: deque(maxlen=self.capacity_per_env)))
            self.num_envs = env_dim

        for t in range(time_dim):
            for env_idx in range(env_dim):
                step = {
                    key: self._extract_step(value, t, env_idx)
                    for key, value in arrays.items()
                }
                self._append_step(env_idx, step)

        self._size = self._total_steps()

    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if batch_size is None:
            batch_size = self.batch_size

        candidates = self._enumerate_sequences()
        if not candidates:
            raise ValueError("Buffer does not contain enough data to sample sequences")

        replace = len(candidates) < batch_size
        choice = np.random.choice(len(candidates), size=batch_size, replace=replace)

        sequences: Dict[str, List[np.ndarray]] = defaultdict(list)
        env_cache = self._env_arrays_cache()

        for idx in choice:
            env_idx, start = candidates[idx]
            env_arrays = env_cache[env_idx]
            end = start + self.sequence_length
            for key, array in env_arrays.items():
                if array.shape[0] < end:
                    continue
                window = array[start:end]
                sequences[key].append(window)

        batch: Dict[str, torch.Tensor] = {}
        for key, windows in sequences.items():
            stacked = np.stack(windows, axis=0)
            batch[key] = self.to_tensor(stacked)

        batch["sequence_length"] = torch.tensor(self.sequence_length, device=self.device)
        batch["sequence_stride"] = torch.tensor(self.sequence_stride, device=self.device)

        # Compute returns if return computer is configured
        if self.return_computer is not None and "rewards" in batch:
            returns = self._compute_returns_for_batch(batch)
            if returns is not None:
                batch["returns"] = returns

        return batch

    def sample_all(self) -> Dict[str, torch.Tensor]:
        candidates = self._enumerate_sequences()
        if not candidates:
            raise ValueError("No sequences available to sample")
        batch_size = len(candidates)

        env_cache = self._env_arrays_cache()
        sequences: Dict[str, List[np.ndarray]] = defaultdict(list)
        for env_idx, start in candidates:
            env_arrays = env_cache[env_idx]
            end = start + self.sequence_length
            for key, array in env_arrays.items():
                if array.shape[0] < end:
                    continue
                window = array[start:end]
                sequences[key].append(window)

        batch: Dict[str, torch.Tensor] = {}
        for key, windows in sequences.items():
            stacked = np.stack(windows, axis=0)
            batch[key] = self.to_tensor(stacked)

        batch["sequence_length"] = torch.tensor(self.sequence_length, device=self.device)
        batch["sequence_stride"] = torch.tensor(self.sequence_stride, device=self.device)

        # Compute returns if return computer is configured
        if self.return_computer is not None and "rewards" in batch:
            returns = self._compute_returns_for_batch(batch)
            if returns is not None:
                batch["returns"] = returns

        return batch

    def ready(self) -> bool:
        return len(self._enumerate_sequences()) >= self.batch_size

    def clear(self) -> None:
        self._setup_storage()
        self._size = 0

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        state_envs: List[Dict[str, List[np.ndarray]]] = []
        for env_buf in self._env_buffers:
            env_state = {
                key: list(values)
                for key, values in env_buf.items()
            }
            state_envs.append(env_state)
        return {
            "env_buffers": state_envs,
            "num_envs": self.num_envs,
            "gamma": self.gamma,
            "sequence_length": self.sequence_length,
            "sequence_stride": self.sequence_stride,
            "pad_end_of_episode": self.pad_end_of_episode,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.num_envs = int(state.get("num_envs", self.num_envs))
        self.gamma = float(state.get("gamma", self.gamma))
        self.sequence_length = int(state.get("sequence_length", self.sequence_length))
        self.sequence_stride = max(1, int(state.get("sequence_stride", self.sequence_stride)))
        self.pad_end_of_episode = bool(state.get("pad_end_of_episode", self.pad_end_of_episode))

        self._setup_storage()
        env_states: List[Dict[str, List[np.ndarray]]] = state.get("env_buffers", [])
        if env_states:
            self.num_envs = len(env_states)
            self._env_buffers = []
            for env_state in env_states:
                env_buf = defaultdict(lambda: deque(maxlen=self.capacity_per_env))
                for key, values in env_state.items():
                    dq = deque(maxlen=self.capacity_per_env)
                    for item in values:
                        dq.append(np.asarray(item))
                    env_buf[key] = dq
                self._env_buffers.append(env_buf)
        self._size = self._total_steps()

    def _save_buffer_state(self) -> Dict[str, Any]:
        return self.get_state()

    def _load_buffer_state(self, state: Dict[str, Any]) -> None:
        self.set_state(state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _append_step(self, env_idx: int, step: Dict[str, np.ndarray]) -> None:
        env_buf = self._env_buffers[env_idx]
        for key, value in step.items():
            env_buf[key].append(np.asarray(value))

    def _total_steps(self) -> int:
        total = 0
        for env_buf in self._env_buffers:
            observations = env_buf.get("observations")
            if observations is not None:
                total += len(observations)
        return total

    def _enumerate_sequences(self) -> List[Tuple[int, int]]:
        candidates: List[Tuple[int, int]] = []
        for env_idx, env_buf in enumerate(self._env_buffers):
            observations = env_buf.get("observations")
            if observations is None:
                continue
            available = len(observations) - self.sequence_length + 1
            if available <= 0:
                continue
            for start in range(0, available, self.sequence_stride):
                candidates.append((env_idx, start))
        return candidates

    def _env_arrays_cache(self) -> Dict[int, Dict[str, np.ndarray]]:
        cache: Dict[int, Dict[str, np.ndarray]] = {}
        for env_idx, env_buf in enumerate(self._env_buffers):
            env_arrays = {}
            for key, values in env_buf.items():
                if not values:
                    continue
                env_arrays[key] = np.stack(values, axis=0)
            cache[env_idx] = env_arrays
        return cache

    def _build_mask(
        self,
        dones: Optional[np.ndarray],
        start: int,
        end: int,
    ) -> np.ndarray:
        if dones is None:
            return np.ones(self.sequence_length, dtype=np.float32)

        mask = np.ones(self.sequence_length, dtype=np.float32)
        window = dones[start:end]
        if window.dtype != np.bool_:
            window = window.astype(bool)
        terminal_positions = np.where(window)[0]
        if terminal_positions.size > 0 and self.pad_end_of_episode:
            first = terminal_positions[0]
            mask[first + 1 :] = 0.0
        return mask

    def _compute_returns_for_batch(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute returns for a batch using the configured return computer.

        Args:
            batch: Batch dict containing rewards, dones, and optionally values

        Returns:
            Returns tensor with shape (B, T) or None if return computer returns None
        """
        if self.return_computer is None:
            return None

        # Convert tensors to numpy for return computation
        rewards_np = batch["rewards"].cpu().numpy()
        dones_np = batch["dones"].cpu().numpy()

        # Handle boolean dones
        if dones_np.dtype == bool:
            dones_np = dones_np.astype(np.float32)

        # Extract values if available (for bootstrapping methods like n-step, TD-lambda)
        values_np = None
        if "values" in batch:
            values_np = batch["values"].cpu().numpy()

        # Compute returns
        returns_np = self.return_computer.compute_returns(
            rewards=rewards_np,
            dones=dones_np,
            values=values_np,
        )

        # Return None if return computer doesn't compute returns (e.g., "none" type)
        if returns_np is None:
            return None

        # Convert back to tensor
        return self.to_tensor(returns_np)

    @staticmethod
    def _extract_step(array: np.ndarray, t: int, env_idx: int) -> np.ndarray:
        if array.ndim == 0:
            return array
        if array.ndim == 1:
            return array[t]
        if array.ndim >= 2:
            if array.shape[1] == 1:
                return array[t, 0]
            if env_idx >= array.shape[1]:
                raise IndexError(
                    f"Env index {env_idx} out of bounds for array with shape {array.shape}"
                )
            return array[t, env_idx]
        raise ValueError(f"Unsupported array shape {array.shape}")

    def _resolve_return_computer(self, config_value: Any) -> Optional[BaseReturnComputer]:
        if config_value is None:
            return None
        if isinstance(config_value, BaseReturnComputer):
            return config_value
        raise TypeError(
            "return_computer must be None or a pre-instantiated BaseReturnComputer. "
            "Use Hydra to instantiate return computers before passing them to the buffer."
        )
