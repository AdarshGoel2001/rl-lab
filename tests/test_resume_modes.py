from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.train import load_resume_checkpoint
from src.orchestration.orchestrator import Orchestrator
from src.workflows.utils.base import WorldModelWorkflow
from src.workflows.utils.context import ControllerManager, WorldModelComponents


class NoopWorkflow(WorldModelWorkflow):
    def __init__(self):
        super().__init__()
        self.restored_state = None

    def initialize(self, context):
        self.context = context

    def update_world_model(self, batch, *, phase):
        return {}

    def get_state(self):
        return {"workflow_live": 1}

    def set_state(self, state):
        self.restored_state = dict(state)


class DummyEnv:
    num_envs = 1

    def reset(self, **kwargs):
        del kwargs
        return np.zeros((1, 2), dtype=np.float32)

    def close(self):
        pass


class DummyBuffer:
    def __init__(self):
        self.restored_state = None

    def initialize(self, context=None):
        del context

    def ready(self):
        return False

    def add(self, **kwargs):
        del kwargs

    def get_state(self):
        return {"buffer_live": 1}

    def set_state(self, state):
        self.restored_state = dict(state)


class NonRestorableBuffer:
    def initialize(self, context=None):
        del context

    def ready(self):
        return False


def _make_cfg():
    return OmegaConf.create(
        {
            "experiment": {"name": "resume_test", "device": "cpu", "seed": 1},
            "logging": {"tensorboard": False, "terminal": False},
            "algorithm": {"world_model_lr": 1e-3},
            "training": {
                "total_timesteps": 10,
                "phases": [
                    {
                        "name": "train_again",
                        "type": "offline",
                        "duration_updates": 2,
                        "workflow_hooks": ["update_world_model"],
                    }
                ],
            },
        }
    )


def _optimizer_with_state(module):
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    loss = module(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return optimizer


def _write_checkpoint(path: Path, *, model, controller, optimizer):
    torch.save(
        {
            "version": 1,
            "global_step": 99,
            "workflow_name": "NoopWorkflow",
            "phase_state": {
                "current_index": 1,
                "pending_hooks": [],
                "steps_done": 0,
                "updates_done": 9,
                "cycles_done": 1,
                "finished": True,
            },
            "components": {"model": model.state_dict()},
            "controllers": {"actor": controller.state_dict()},
            "optimizers": {"world_model": optimizer.state_dict()},
            "workflow_custom": {"restored": True},
            "buffers": {"replay": {"restored_buffer": True}},
        },
        path,
    )


def _make_orchestrator(tmp_path):
    cfg = _make_cfg()
    model = nn.Linear(2, 1)
    controller = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    workflow = NoopWorkflow()
    orchestrator = Orchestrator(
        cfg,
        workflow,
        experiment_dir=tmp_path / "run",
        components=WorldModelComponents({"model": model}),
        optimizers={"world_model": optimizer},
        controllers={"actor": controller},
        controller_manager=ControllerManager({"actor": controller}),
        buffers={"replay": DummyBuffer()},
        train_environment=DummyEnv(),
        eval_environment=DummyEnv(),
    )
    orchestrator.ensure_context()
    return orchestrator, model, controller, optimizer, workflow


def test_exact_resume_restores_training_state(tmp_path):
    source_model = nn.Linear(2, 1)
    source_controller = nn.Linear(2, 1)
    source_optimizer = _optimizer_with_state(source_model)
    ckpt = tmp_path / "checkpoint.pt"
    _write_checkpoint(ckpt, model=source_model, controller=source_controller, optimizer=source_optimizer)

    orchestrator, model, controller, optimizer, workflow = _make_orchestrator(tmp_path)
    replay_buffer = orchestrator.buffers["replay"]

    orchestrator.load_checkpoint(ckpt, mode="exact")

    assert orchestrator.global_step == 99
    assert orchestrator.scheduler.is_finished()
    assert workflow.restored_state == {"restored": True}
    assert replay_buffer.restored_state == {"restored_buffer": True}
    assert optimizer.state_dict()["state"]
    for loaded, expected in zip(model.parameters(), source_model.parameters()):
        assert torch.allclose(loaded, expected)
    for loaded, expected in zip(controller.parameters(), source_controller.parameters()):
        assert torch.allclose(loaded, expected)


def test_warm_start_resets_scheduler_and_skips_optimizer(tmp_path):
    source_model = nn.Linear(2, 1)
    source_controller = nn.Linear(2, 1)
    source_optimizer = _optimizer_with_state(source_model)
    ckpt = tmp_path / "checkpoint.pt"
    _write_checkpoint(ckpt, model=source_model, controller=source_controller, optimizer=source_optimizer)

    orchestrator, model, controller, optimizer, workflow = _make_orchestrator(tmp_path)

    orchestrator.load_checkpoint(ckpt, mode="warm_start")

    assert orchestrator.global_step == 0
    assert not orchestrator.scheduler.is_finished()
    assert workflow.restored_state is None
    assert optimizer.state_dict()["state"] == {}
    for loaded, expected in zip(model.parameters(), source_model.parameters()):
        assert torch.allclose(loaded, expected)
    for loaded, expected in zip(controller.parameters(), source_controller.parameters()):
        assert torch.allclose(loaded, expected)


def test_warm_start_optimizer_resets_scheduler_and_keeps_optimizer(tmp_path):
    source_model = nn.Linear(2, 1)
    source_controller = nn.Linear(2, 1)
    source_optimizer = _optimizer_with_state(source_model)
    ckpt = tmp_path / "checkpoint.pt"
    _write_checkpoint(ckpt, model=source_model, controller=source_controller, optimizer=source_optimizer)

    orchestrator, model, controller, optimizer, workflow = _make_orchestrator(tmp_path)

    orchestrator.load_checkpoint(ckpt, mode="warm_start_optimizer")

    assert orchestrator.global_step == 0
    assert not orchestrator.scheduler.is_finished()
    assert workflow.restored_state is None
    assert optimizer.state_dict()["state"]
    for loaded, expected in zip(model.parameters(), source_model.parameters()):
        assert torch.allclose(loaded, expected)
    for loaded, expected in zip(controller.parameters(), source_controller.parameters()):
        assert torch.allclose(loaded, expected)


def test_exact_resume_fails_if_buffer_state_cannot_be_restored(tmp_path):
    source_model = nn.Linear(2, 1)
    source_controller = nn.Linear(2, 1)
    source_optimizer = _optimizer_with_state(source_model)
    ckpt = tmp_path / "checkpoint.pt"
    _write_checkpoint(ckpt, model=source_model, controller=source_controller, optimizer=source_optimizer)

    cfg = _make_cfg()
    model = nn.Linear(2, 1)
    controller = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    workflow = NoopWorkflow()
    orchestrator = Orchestrator(
        cfg,
        workflow,
        experiment_dir=tmp_path / "run",
        components=WorldModelComponents({"model": model}),
        optimizers={"world_model": optimizer},
        controllers={"actor": controller},
        controller_manager=ControllerManager({"actor": controller}),
        buffers={"replay": NonRestorableBuffer()},
        train_environment=DummyEnv(),
        eval_environment=DummyEnv(),
    )
    orchestrator.ensure_context()

    try:
        orchestrator.load_checkpoint(ckpt, mode="exact")
    except RuntimeError as exc:
        assert "Exact resume requires buffer 'replay' to implement set_state()" in str(exc)
    else:
        raise AssertionError("exact resume should fail when buffer state cannot be restored")


def test_exact_resume_fails_if_checkpoint_has_no_buffer_state(tmp_path):
    source_model = nn.Linear(2, 1)
    source_controller = nn.Linear(2, 1)
    source_optimizer = _optimizer_with_state(source_model)
    ckpt = tmp_path / "checkpoint.pt"
    _write_checkpoint(ckpt, model=source_model, controller=source_controller, optimizer=source_optimizer)
    checkpoint_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    checkpoint_data.pop("buffers")
    torch.save(checkpoint_data, ckpt)

    orchestrator, _, _, _, _ = _make_orchestrator(tmp_path)

    try:
        orchestrator.load_checkpoint(ckpt, mode="exact")
    except RuntimeError as exc:
        assert "Exact resume requires checkpoint buffer state for 'replay'" in str(exc)
    else:
        raise AssertionError("exact resume should fail when checkpoint has no buffer state")


def test_missing_resume_path_fails_by_default(tmp_path):
    cfg = _make_cfg()
    cfg.training.resume_path = str(tmp_path / "missing.pt")
    cfg.training.resume_mode = "warm_start_optimizer"

    class ResumeRecorder:
        def load_checkpoint(self, path, *, mode):
            raise AssertionError("load_checkpoint should not be called for a missing path")

    try:
        load_resume_checkpoint(ResumeRecorder(), cfg)
    except FileNotFoundError as exc:
        assert "training.resume_path does not exist" in str(exc)
        assert str(tmp_path / "missing.pt") in str(exc)
    else:
        raise AssertionError("missing resume path should fail by default")


def test_missing_resume_path_can_be_explicitly_allowed(tmp_path):
    cfg = _make_cfg()
    cfg.training.resume_path = str(tmp_path / "missing.pt")
    cfg.training.resume_mode = "warm_start_optimizer"
    cfg.training.allow_missing_resume = True

    class ResumeRecorder:
        called = False

        def load_checkpoint(self, path, *, mode):
            self.called = True

    recorder = ResumeRecorder()

    load_resume_checkpoint(recorder, cfg)

    assert recorder.called is False
