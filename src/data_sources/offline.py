"""Offline dataset adapter exposing the data source interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from . import register_data_source
from .base import DataSource
from ..workflows.world_models.context import WorkflowContext


@register_data_source("offline_tensor")
@register_data_source("offline_dataset")
class OfflineDatasetSource(DataSource):
    """Loads fixed datasets from disk and samples mini-batches for updates."""

    def __init__(
        self,
        path: Optional[str] = None,
        *,
        format: Optional[str] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        shuffle: bool = True,
        replace: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.explicit_format = format
        self.batch_size_config = batch_size
        self.device_override = device
        self.shuffle = shuffle
        self.replace = replace
        self.seed = seed

        self._rng: Optional[np.random.Generator] = None
        self._data_arrays: Optional[Dict[str, np.ndarray]] = None
        self._size = 0
        self._device: Optional[str] = None
        self._fallback_batch_size: Optional[int] = None

    def initialize(self, context: WorkflowContext) -> None:
        self._device = self.device_override or context.device
        self._rng = np.random.default_rng(self.seed)
        self._fallback_batch_size = getattr(context.buffer, "batch_size", None)
        if self.path is None:
            raise ValueError("offline data source requires 'path' configuration.")
        self._data_arrays = self._load_dataset(self.path, format_hint=self.explicit_format)
        self._size = self._infer_size(self._data_arrays)

    def add(self, **kwargs: Any) -> None:
        raise RuntimeError("OfflineDatasetSource is read-only; cannot add new samples.")

    def sample(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if self._data_arrays is None or self._size == 0:
            raise RuntimeError("Offline dataset not initialised or empty.")

        effective_batch = batch_size or self.batch_size_config or self._fallback_batch_size
        if effective_batch is None:
            raise ValueError("Offline dataset sampling requires 'batch_size' to be specified.")

        if not self.replace and effective_batch > self._size:
            raise ValueError(
                f"Cannot sample {effective_batch} items without replacement from dataset of size {self._size}."
            )

        indices = (
            self._sequential_indices(effective_batch)
            if not self.shuffle
            else self._random_indices(effective_batch)
        )

        batch: Dict[str, torch.Tensor] = {}
        for key, array in self._data_arrays.items():
            slice_ = array[indices]
            tensor = torch.as_tensor(slice_)
            if self._device:
                tensor = tensor.to(self._device)
            batch[key] = tensor
        return batch

    def ready(self) -> bool:
        return self._size > 0

    def state_dict(self, *, mode: str = "checkpoint") -> Mapping[str, Any]:
        if mode == "metrics":
            metrics: Dict[str, float] = {
                "offline/size": float(self._size),
            }
            if self.path is not None:
                metrics["offline/has_path"] = 1.0
            return metrics
        if mode != "checkpoint":
            raise ValueError(f"OfflineDatasetSource does not support state_dict mode '{mode}'.")
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _ = state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_indices(self, batch_size: int) -> np.ndarray:
        assert self._rng is not None
        return self._rng.choice(self._size, size=batch_size, replace=self.replace)

    def _sequential_indices(self, batch_size: int) -> np.ndarray:
        assert self._rng is not None
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
        elif fmt in {"json"}:
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
            stacked: Dict[str, Iterable[np.ndarray]] = {key: [] for key in keys}
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
