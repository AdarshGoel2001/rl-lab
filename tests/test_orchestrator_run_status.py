import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.orchestration.orchestrator import Orchestrator
from src.workflows.utils.base import CollectResult, WorldModelWorkflow
from src.workflows.utils.context import ControllerManager, WorldModelComponents


class CollectOnceWorkflow(WorldModelWorkflow):
    def initialize(self, context):
        self.context = context

    def collect_step(self, step, *, phase):
        del step, phase
        return CollectResult(
            steps=1,
            metrics={"reward": 2.5},
            trajectory={
                "observations": np.zeros((1, 1, 2), dtype=np.float32),
                "actions": np.zeros((1, 1, 1), dtype=np.float32),
                "rewards": np.ones((1, 1), dtype=np.float32),
                "dones": np.zeros((1, 1), dtype=bool),
            },
        )

    def update_world_model(self, batch, *, phase):
        del batch, phase
        return {}


class DummyEnv:
    num_envs = 1

    def reset(self, **kwargs):
        del kwargs
        return np.zeros((1, 2), dtype=np.float32)

    def close(self):
        pass


class RecordingBuffer:
    def __init__(self):
        self.trajectories = []

    def initialize(self, context=None):
        del context

    def ready(self):
        return False

    def add(self, **kwargs):
        self.trajectories.append(kwargs["trajectory"])


def test_orchestrator_writes_run_status_file_with_hook_metrics(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "status_test", "device": "cpu", "seed": 1},
            "logging": {"tensorboard": False, "terminal": False},
            "algorithm": {"world_model_lr": 1e-3},
            "training": {
                "total_timesteps": 1,
                "checkpoint_frequency": 0,
                "eval_frequency": 0,
                "phases": [
                    {
                        "name": "collect_for_status",
                        "type": "online",
                        "duration_steps": 1,
                        "workflow_hooks": ["collect"],
                    }
                ],
            },
        }
    )
    model = nn.Linear(2, 1)
    controller = nn.Linear(2, 1)
    buffer = RecordingBuffer()
    orchestrator = Orchestrator(
        cfg,
        CollectOnceWorkflow(),
        experiment_dir=tmp_path / "run",
        components=WorldModelComponents({"model": model}),
        optimizers={"world_model": torch.optim.Adam(model.parameters(), lr=1e-3)},
        controllers={"actor": controller},
        controller_manager=ControllerManager({"actor": controller}),
        buffers={"replay": buffer},
        train_environment=DummyEnv(),
        eval_environment=DummyEnv(),
    )

    orchestrator.run()

    status_path = tmp_path / "run" / "run_status.json"
    assert status_path.exists()

    status = json.loads(status_path.read_text())
    assert status["schema_version"] == 1
    assert status["run_id"] == "run"
    assert status["pid"] == os.getpid()
    assert status["command"] == " ".join(sys.argv)
    assert status["experiment_dir"] == str(tmp_path / "run")
    assert status["workflow_name"] == "CollectOnceWorkflow"
    assert status["status"] == "completed"
    assert status["phase"] == "collect_for_status"
    assert status["action"] == "collect"
    assert status["hook_state"] == "completed"
    assert status["global_step"] == 1
    assert status["last_metrics"] == {"collect/reward": 2.5}
    assert status["started_at"]
    assert status["updated_at"]
    assert buffer.trajectories


def test_orchestrator_writes_run_summary_file(tmp_path):
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "summary_test", "device": "cpu", "seed": 1},
            "logging": {"tensorboard": False, "terminal": False},
            "algorithm": {"world_model_lr": 1e-3},
            "training": {
                "total_timesteps": 1,
                "checkpoint_frequency": 0,
                "eval_frequency": 0,
                "phases": [
                    {
                        "name": "collect_for_summary",
                        "type": "online",
                        "duration_steps": 1,
                        "workflow_hooks": ["collect"],
                    }
                ],
            },
        }
    )
    model = nn.Linear(2, 1)
    controller = nn.Linear(2, 1)
    orchestrator = Orchestrator(
        cfg,
        CollectOnceWorkflow(),
        experiment_dir=tmp_path / "run",
        components=WorldModelComponents({"model": model}),
        optimizers={"world_model": torch.optim.Adam(model.parameters(), lr=1e-3)},
        controllers={"actor": controller},
        controller_manager=ControllerManager({"actor": controller}),
        buffers={"replay": RecordingBuffer()},
        train_environment=DummyEnv(),
        eval_environment=DummyEnv(),
    )

    result = orchestrator.run()

    summary_path = tmp_path / "run" / "run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == 1
    assert summary["run_id"] == "run"
    assert summary["status"] == "completed"
    assert summary["global_step"] == 1
    assert summary["workflow_name"] == "CollectOnceWorkflow"
    assert summary["final_metrics"]["wall_time"] == result["wall_time"]
    assert summary["workflow_metrics"] == {}
    assert summary["checkpoints"]["final"].endswith("checkpoints/final.pt")
    assert summary["checkpoints"]["latest"].endswith("checkpoints/final.pt")
    assert summary["checkpoints"]["best"] is None
