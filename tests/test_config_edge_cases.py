import os
import tempfile
from pathlib import Path

import pytest

from src.utils.config import load_config, ConfigError, apply_config_overrides


def test_env_var_expansion(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MY_ENV", "expanded")
    cfg = {
        "experiment": {"name": "${MY_ENV}"},
        "algorithm": {"name": "ppo"},
    }
    p = tmp_path / "c.yaml"
    import yaml

    p.write_text(yaml.dump(cfg))
    c = load_config(p)
    assert c.experiment.name == "expanded"


def test_invalid_buffer_capacity_raises(tmp_path: Path):
    import yaml

    cfg = {
        "experiment": {"name": "x"},
        "algorithm": {"name": "ppo"},
        "buffer": {"type": "trajectory", "capacity": -1},
    }
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ConfigError):
        _ = load_config(p)


def test_apply_overrides_dot_notation(tmp_path: Path):
    import yaml
    base = {
        "experiment": {"name": "x", "seed": 1},
        "algorithm": {"name": "ppo", "lr": 1e-3},
    }
    p = tmp_path / "b.yaml"
    p.write_text(yaml.dump(base))
    c = load_config(p)
    c2 = apply_config_overrides(c, {"algorithm.lr": 5e-4, "experiment.seed": 123})
    assert getattr(c2.algorithm, "lr", None) == 5e-4
    assert getattr(c2.experiment, "seed", None) == 123

