"""Replay buffer adapter exposing the data source interface."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ..buffers.base import BaseBuffer
from ..utils.registry import get_buffer
from ..workflows.world_models.context import WorkflowContext
from . import register_data_source
from .base import DataSource


@register_data_source("replay")
@register_data_source("world_model_replay")
class ReplayDataSource(DataSource):
    """Thin wrapper delegating to an existing replay buffer implementation."""

    def __init__(
        self,
        buffer: Optional[BaseBuffer] = None,
        *,
        buffer_type: Optional[str] = None,
        buffer_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.buffer = buffer
        self._buffer_type = buffer_type
        self._buffer_config = dict(buffer_config or {})

    def initialize(self, context: WorkflowContext) -> None:
        if self.buffer is None:
            if self._buffer_type:
                cfg = dict(self._buffer_config)
                cfg.setdefault("device", context.device)
                cfg.setdefault("num_envs", getattr(context.train_environment, "num_envs", 1))
                buffer_cls = get_buffer(self._buffer_type)
                self.buffer = buffer_cls(cfg)
            else:
                self.buffer = context.buffer

    def add(self, **kwargs: Any) -> None:
        if self.buffer is None:
            raise RuntimeError("ReplayDataSource requires a buffer before adding data.")
        self.buffer.add(**kwargs)

    def sample(self, batch_size: Optional[int] = None) -> Any:
        if self.buffer is None:
            raise RuntimeError("ReplayDataSource cannot sample before initialization.")
        return self.buffer.sample(batch_size=batch_size)

    def ready(self) -> bool:
        if self.buffer is None:
            return False
        return self.buffer.ready()

    def state_dict(self) -> Mapping[str, Any]:
        if self.buffer is None:
            return {}
        return {"buffer": self.buffer.save_checkpoint()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not state or self.buffer is None:
            return
        checkpoint = state.get("buffer")
        if checkpoint:
            self.buffer.load_checkpoint(checkpoint)
