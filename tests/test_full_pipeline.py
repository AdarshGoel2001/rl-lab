import numpy as np
import torch
from pathlib import Path

from src.environments.base import BaseEnvironment, SpaceSpec
from src.utils.registry import register_environment
from src.core.trainer import create_trainer_from_config


@register_environment("dummy_single")
class DummySingleEnv(BaseEnvironment):
    def _setup_environment(self):
        self._d = 4
        self._t = 0

    def _get_observation_space(self) -> SpaceSpec:
        return SpaceSpec(shape=(self._d,), dtype=np.float32)

    def _get_action_space(self) -> SpaceSpec:
        return SpaceSpec(shape=(2,), dtype=np.int64, discrete=True, n=2)

    def _reset_environment(self, seed=None):
        self._t = 0
        return np.zeros(self._d, dtype=np.float32)

    def _step_environment(self, action):
        self._t += 1
        obs = np.full(self._d, self._t, dtype=np.float32)
        return obs, 1.0, self._t >= 5, {"t": self._t}


def test_random_trainer_end_to_end(tmp_path: Path):
    import yaml

    cfg = {
        "experiment": {"name": "e2e_random", "seed": 0, "device": "cpu"},
        "algorithm": {"name": "random"},
        "environment": {"name": "Dummy", "wrapper": "dummy_single"},
        "network": {"type": "mlp", "input_dim": 4, "output_dim": 2},
        "buffer": {"type": "trajectory", "capacity": 64, "batch_size": 8},
        "training": {"total_timesteps": 32, "eval_frequency": 16, "checkpoint_frequency": 1000},
        "logging": {"terminal": False, "tensorboard": False, "wandb_enabled": False},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    trainer = create_trainer_from_config(str(p), experiment_dir=str(tmp_path / "exp"))
    results = trainer.train()
    assert isinstance(results, dict)
    trainer.cleanup()

