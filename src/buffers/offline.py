"""Offline dataset buffer that serves batches from a fixed dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from .base import BaseBuffer


class OfflineDatasetBuffer(BaseBuffer):
    """Loads a static dataset from disk and exposes a buffer interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        merged: Dict[str, Any] = dict(config or {})
        if kwargs:
            merged.update(kwargs)

        self.path = Path(merged.pop("path", "")) if "path" in merged else None
        self.explicit_format = merged.pop("format", None)
        self.shuffle = bool(merged.pop("shuffle", True))
        self.replace = bool(merged.pop("replace", True))
        self.seed = merged.pop("seed", None)
        self.device_override = merged.get("device")
        self._batch_size_override = merged.get("batch_size")

        merged.setdefault("capacity", 0)
        merged.setdefault("batch_size", 64)
        merged.setdefault("device", "cpu")

        super().__init__(merged)

        self._rng: Optional[np.random.Generator] = None
        self._data_arrays: Optional[Dict[str, np.ndarray]] = None
        self._size = 0
        self._sample_device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # BaseBuffer interface
    # ------------------------------------------------------------------
    def _setup_storage(self) -> None:
        self._data_arrays = None
        self._size = 0

    def initialize(self, context: Any = None) -> None:
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
            raise ValueError("OfflineDatasetBuffer requires a 'path' configuration.")

        arrays = self._load_dataset(self.path, format_hint=self.explicit_format)
        self._data_arrays = arrays
        self._size = self._infer_size(arrays)

    def add(self, **kwargs: Any) -> None:
        del kwargs
        raise RuntimeError("OfflineDatasetBuffer is read-only; cannot add new samples.")

    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if self._data_arrays is None or self._size == 0:
            raise RuntimeError("Offline dataset not initialised or empty.")

        effective_batch = (
            batch_size
            or self._batch_size_override
            or self.batch_size
        )
        if effective_batch is None:
            raise ValueError("Offline dataset sampling requires 'batch_size' to be specified.")

        if not self.replace and effective_batch > self._size:
            raise ValueError(
                f"Cannot sample {effective_batch} items without replacement from dataset of size {self._size}."
            )

        indices = self._sequential_indices(effective_batch) if not self.shuffle else self._random_indices(effective_batch)

        batch: Dict[str, torch.Tensor] = {}
        for key, array in self._data_arrays.items():
            slice_ = array[indices]
            tensor = torch.as_tensor(slice_)
            if self._sample_device is not None:
                tensor = tensor.to(self._sample_device)
            batch[key] = tensor
        return batch

    def clear(self) -> None:
        self._data_arrays = None
        self._size = 0

    def ready(self) -> bool:
        return self._size > 0

    def _save_buffer_state(self) -> Dict[str, Any]:
        return {}

    def _load_buffer_state(self, state: Dict[str, Any]) -> None:
        del state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _random_indices(self, batch_size: int) -> np.ndarray:
        if self._rng is None:
            raise RuntimeError("OfflineDatasetBuffer RNG not initialised.")
        return self._rng.choice(self._size, size=batch_size, replace=self.replace)

    def _sequential_indices(self, batch_size: int) -> np.ndarray:
        if self._rng is None:
            raise RuntimeError("OfflineDatasetBuffer RNG not initialised.")
        start = int(self._rng.integers(0, max(1, self._size)))
        end = start + batch_size
        if end <= self._size:
            return np.arange(start, end)
        wrapped = np.concatenate([np.arange(start, self._size), np.arange(0, end - self._size)])
        return wrapped.astype(np.int64)

    def _load_dataset(self, path: Path, *, format_hint: Optional[str] = None) -> Dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Offline dataset not found at '{path}'.")

        fmt = (format_hint or path.suffix.lstrip(".")).lower()
        if fmt in {"pt", "pth", "torch"}:
            data = torch.load(path, map_location="cpu")
        elif fmt in {"npz", "numpy"}:
            loaded = np.load(path, allow_pickle=True)
            data = {key: loaded[key] for key in loaded.files}
        elif fmt == "json":
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        else:
            raise ValueError(f"Unsupported offline dataset format '{fmt}' for file '{path}'.")

        return self._normalise_dataset(data)

    def _normalise_dataset(self, data: Any) -> Dict[str, np.ndarray]:
        if isinstance(data, Mapping):
            arrays = {key: self._to_numpy(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            if not data:
                raise ValueError("Offline dataset list is empty.")
            first = data[0]
            if not isinstance(first, Mapping):
                raise TypeError("Offline dataset list entries must be dict-like.")
            keys = list(first.keys())
            stacked: Dict[str, list[np.ndarray]] = {key: [] for key in keys}
            for item in data:
                for key in keys:
                    stacked[key].append(self._to_numpy(item[key]))
            arrays = {key: np.stack(values, axis=0) for key, values in stacked.items()}
        else:
            raise TypeError(f"Unsupported offline dataset structure: {type(data)}")
        return arrays

    def _infer_size(self, arrays: Mapping[str, np.ndarray]) -> int:
        lengths = {key: array.shape[0] for key, array in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Offline dataset tensors must share leading dimension, got {lengths}")
        return next(iter(lengths.values()))

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            return np.asarray(value)
        return np.asarray(value)
