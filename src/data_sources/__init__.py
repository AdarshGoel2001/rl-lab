"\"\"\"Data source abstractions and registry for world-model workflows.\"\"\""

from typing import Any, Dict, Optional, Type

from .base import DataSource

DATA_SOURCE_REGISTRY: Dict[str, Type[DataSource]] = {}


def register_data_source(name: str):
    """Decorator registering a data source implementation."""

    def decorator(cls: Type[DataSource]) -> Type[DataSource]:
        DATA_SOURCE_REGISTRY[name] = cls
        return cls

    return decorator


def get_data_source(name: str) -> Type[DataSource]:
    if name not in DATA_SOURCE_REGISTRY:
        raise KeyError(f"Unknown data source type '{name}'.")
    return DATA_SOURCE_REGISTRY[name]


def create_data_source(name: str, *, config: Optional[Dict[str, Any]] = None) -> DataSource:
    source_cls = get_data_source(name)
    config = config or {}
    return source_cls(**config)


from .replay import ReplayDataSource  # noqa: E402  (registry side-effect)
from .offline import OfflineDatasetSource  # noqa: E402

__all__ = [
    "DataSource",
    "ReplayDataSource",
    "OfflineDatasetSource",
    "register_data_source",
    "create_data_source",
    "get_data_source",
    "DATA_SOURCE_REGISTRY",
]
