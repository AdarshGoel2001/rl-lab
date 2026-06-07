import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.orchestration.orchestrator import Orchestrator
from src.workflows.utils.base import WorldModelWorkflow
from src.workflows.utils.context import ControllerManager, WorldModelComponents


class DummyEnv:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs

    def reset(self, **kwargs):
        del kwargs
        return np.zeros((1, 2), dtype=np.float32)

    def close(self):
        pass


class DummyBuffer:
    def initialize(self, context=None):
        del context

    def ready(self):
        return False

    def add(self, **kwargs):
        del kwargs


class NoEvalWorkflow(WorldModelWorkflow):
    def initialize(self, context):
        self.context = context

    def update_world_model(self, batch, *, phase):
        del batch, phase
        return {}


class EvalWorkflow(NoEvalWorkflow):
    def __init__(self):
        super().__init__()
        self.calls = []

    def evaluate(self, *, num_eval_batches, max_steps_per_episode, deterministic):
        self.calls.append(
            {
                "num_eval_batches": num_eval_batches,
                "max_steps_per_episode": max_steps_per_episode,
                "deterministic": deterministic,
            }
        )
        return {"return_mean": 12.0}


def _make_cfg():
    return OmegaConf.create(
        {
            "experiment": {"name": "eval_contract_test", "device": "cpu", "seed": 1},
            "logging": {"tensorboard": False, "terminal": False},
            "algorithm": {"world_model_lr": 1e-3},
            "training": {
                "total_timesteps": 10,
                "num_eval_episodes": 3,
                "max_eval_steps": 27,
                "phases": [],
            },
        }
    )


def _make_orchestrator(tmp_path, workflow, *, cfg=None, eval_num_envs=1):
    cfg = cfg or _make_cfg()
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    controller = nn.Linear(2, 1)
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
        eval_environment=DummyEnv(num_envs=eval_num_envs),
    )
    orchestrator.ensure_context()
    return orchestrator


def test_orchestrator_converts_total_eval_episodes_to_workflow_batches(tmp_path):
    cfg = _make_cfg()
    cfg.training.num_eval_episodes = 6
    workflow = EvalWorkflow()
    orchestrator = _make_orchestrator(tmp_path, workflow, cfg=cfg, eval_num_envs=3)

    metrics = orchestrator._run_evaluation({"name": "scheduled_eval"})

    assert metrics == {
        "return_mean": 12.0,
        "episodes": 6.0,
        "eval_episode_batches": 2.0,
        "eval_num_envs": 3.0,
        "eval_total_episodes": 6.0,
    }
    assert workflow.calls == [
        {
            "num_eval_batches": 2,
            "max_steps_per_episode": 27,
            "deterministic": True,
        }
    ]


def test_orchestrator_rejects_eval_episode_count_not_divisible_by_num_envs(tmp_path):
    cfg = _make_cfg()
    cfg.training.num_eval_episodes = 5
    orchestrator = _make_orchestrator(tmp_path, EvalWorkflow(), cfg=cfg, eval_num_envs=3)

    with pytest.raises(ValueError, match="training.num_eval_episodes"):
        orchestrator._run_evaluation({"name": "scheduled_eval"})


def test_orchestrator_requires_workflow_evaluate_for_eval(tmp_path):
    orchestrator = _make_orchestrator(tmp_path, NoEvalWorkflow())

    with pytest.raises(NotImplementedError, match="must implement evaluate"):
        orchestrator._run_evaluation({"name": "scheduled_eval"})
