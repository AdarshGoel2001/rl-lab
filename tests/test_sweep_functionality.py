import os
from pathlib import Path

import pytest

from tests._helpers import safe_import


pytestmark = pytest.mark.skipif(not safe_import("optuna"), reason="optuna not available")


def test_sweep_orchestrator_runs_minimal_trial(tmp_path: Path, monkeypatch):
    import yaml
    from src.core import sweep as sweep_mod

    # Stub trainer that returns deterministic metric
    class StubTrainer:
        def __init__(self, *args, **kwargs):
            self.step = 0

        def _evaluation_step(self):
            self.step += 1
            return {"eval_return_mean": 1.0 + 0.01 * self.step}

        def train(self):
            # Quick "training" that returns final metric
            return {"eval_return_mean": 1.5}

    # Mock the Trainer import where it's actually imported
    monkeypatch.setattr("src.core.trainer.Trainer", StubTrainer)

    # Minimal sweep config
    cfg = {
        "sweep": {
            "name": "unit_test_sweep",
            "study_name": "unit_test_study",
            "storage": f"sqlite:///{tmp_path/'studies.db'}",
            "direction": "maximize",
            "sampler": {"type": "Random"},
            "pruner": {"type": None},
        },
        "base_config": str(Path("configs/experiments/ppo_cartpole.yaml")),
        "parameters": {"algorithm.lr": {"type": "uniform", "low": 1e-4, "high": 1e-3}},
        "execution": {"n_trials": 1, "n_jobs": 1},
        "objective": {"metric": "eval_return_mean"},
        "advanced": {"experiment_root": str(tmp_path / "sweeps"), "load_if_exists": True},
    }
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.dump(cfg))

    orch = sweep_mod.SweepOrchestrator(str(sweep_path))
    study = orch.create_study()
    assert study is not None
    # Run a very small sweep
    study.optimize(lambda t: orch.objective_function(t), n_trials=1)
    assert len(study.trials) >= 1

