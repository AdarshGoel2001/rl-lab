import pytest

from tests._helpers import safe_import

pytestmark = pytest.mark.skipif(not safe_import("optuna"), reason="optuna not available")


def _orchestrator(tmp_path, sampler_type: str):
    import yaml
    from src.core.sweep import SweepOrchestrator

    # Ensure directory exists
    tmp_path.mkdir(parents=True, exist_ok=True)

    cfg = {
        "sweep": {
            "name": f"sampler_{sampler_type}",
            "study_name": f"study_{sampler_type}",
            "storage": f"sqlite:///{tmp_path/'s.db'}",
            "sampler": {"type": sampler_type},
            "pruner": {"type": "MedianPruner"},
        },
        "base_config": str(tmp_path / "base.yaml"),
        "parameters": {},
        "objective": {"metric": "eval_return_mean"},
        "advanced": {"experiment_root": str(tmp_path / "sweeps")},
    }
    (tmp_path / "base.yaml").write_text("experiment: {name: test}\nalgorithm: {name: ppo}\n")
    p = tmp_path / "sweep.yaml"
    p.write_text(yaml.dump(cfg))
    return SweepOrchestrator(str(p))


def test_sampler_selection(tmp_path):
    from optuna.samplers import TPESampler, RandomSampler
    orch_tpe = _orchestrator(tmp_path / "tpe", "TPE")
    assert isinstance(orch_tpe._create_sampler(), TPESampler)
    orch_rand = _orchestrator(tmp_path / "rand", "Random")
    assert isinstance(orch_rand._create_sampler(), RandomSampler)

