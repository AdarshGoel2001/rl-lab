import pytest

from src.utils.registry import get_environment
from src.utils.config import load_config


def test_unknown_environment_wrapper_raises():
    with pytest.raises(ValueError):
        _ = get_environment("nonexistent_wrapper")


def test_missing_config_file_raises():
    with pytest.raises(Exception):
        _ = load_config("/definitely/missing/config.yaml")

