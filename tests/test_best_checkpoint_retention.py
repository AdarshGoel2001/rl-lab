from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.orchestration.orchestrator import Orchestrator
from src.utils.checkpoint import CheckpointManager
from src.workflows.utils.base import WorldModelWorkflow
from src.workflows.utils.context import WorldModelComponents


class NoopWorkflow(WorldModelWorkflow):
    def initialize(self, context):
        self.context = context

    def update_world_model(self, batch, *, phase):
        return {}


class DummyEnv:
    num_envs = 1

    def reset(self, **kwargs):
        del kwargs
        return torch.zeros(1, 2).numpy()

    def close(self):
        pass


class DummyBuffer:
    def initialize(self, context=None):
        del context

    def ready(self):
        return False

    def add(self, **kwargs):
        del kwargs


def test_checkpoint_cleanup_preserves_best_checkpoint_target(tmp_path):
    manager = CheckpointManager(tmp_path / "run", max_checkpoints=2)

    for step in range(4):
        manager.save({"global_step": step}, step, name=f"step_{step}", is_best=(step == 0))

    best_link = tmp_path / "run" / "checkpoints" / "best.pt"

    assert best_link.is_symlink()
    assert best_link.resolve().exists()
    assert best_link.resolve().name == "step_0.pt"


def test_best_checkpoint_target_is_not_overwritten_by_later_regular_save(tmp_path):
    manager = CheckpointManager(tmp_path / "run", max_checkpoints=10)

    manager.save(
        {"global_step": 10, "value": "best"},
        step=10,
        name="best_step_10",
        is_best=True,
    )
    best_target = (manager.checkpoint_dir / "best.pt").resolve()

    assert best_target.name == "best_step_10.pt"

    manager.save(
        {"global_step": 20, "value": "latest"},
        step=20,
        name="step_20",
        is_best=False,
    )

    assert (manager.checkpoint_dir / "best.pt").resolve() == best_target
    loaded = manager.load(best_target)
    assert loaded["value"] == "best"
    assert loaded["global_step"] == 10


def test_checkpoint_manager_refuses_to_overwrite_existing_checkpoint(tmp_path):
    manager = CheckpointManager(tmp_path / "run", max_checkpoints=10)

    manager.save({"global_step": 10, "value": "first"}, step=10, name="step_10")

    try:
        manager.save({"global_step": 10, "value": "second"}, step=10, name="step_10")
    except FileExistsError as exc:
        assert "Refusing to overwrite checkpoint" in str(exc)
    else:
        raise AssertionError("CheckpointManager should not overwrite immutable checkpoint names")


def test_orchestrator_saves_best_checkpoint_only_when_eval_return_improves(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "best_ckpt_test", "device": "cpu"},
            "logging": {"tensorboard": False, "terminal": False},
            "training": {"total_timesteps": 1, "phases": []},
        }
    )
    calls = []
    orchestrator = Orchestrator(
        cfg,
        NoopWorkflow(),
        experiment_dir=tmp_path / "run",
        components=WorldModelComponents({}),
        optimizers={},
        controllers={},
        buffers={"replay": DummyBuffer()},
        train_environment=DummyEnv(),
        eval_environment=DummyEnv(),
    )

    def record_save(*, name=None, is_best=False):
        calls.append((name, is_best))

    orchestrator._save_checkpoint = record_save  # type: ignore[method-assign]

    orchestrator._maybe_save_best_checkpoint({"return_mean": 10.0})
    orchestrator._maybe_save_best_checkpoint({"return_mean": 9.0})
    orchestrator._maybe_save_best_checkpoint({"return_mean": 11.0})

    assert calls == [("best_step_0", True), ("best_step_0_1", True)]
    assert orchestrator.best_eval_return == 11.0


def test_orchestrator_checkpoint_retention_is_configurable(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "retention_test", "device": "cpu"},
            "logging": {"tensorboard": False, "terminal": False},
            "training": {"total_timesteps": 1, "phases": [], "max_checkpoints": 17},
        }
    )

    orchestrator = Orchestrator(
        cfg,
        NoopWorkflow(),
        experiment_dir=tmp_path / "run",
        components=WorldModelComponents({}),
        optimizers={},
        controllers={},
        buffers={"replay": DummyBuffer()},
        train_environment=DummyEnv(),
        eval_environment=DummyEnv(),
    )

    assert orchestrator.checkpoint_manager.max_checkpoints == 17
