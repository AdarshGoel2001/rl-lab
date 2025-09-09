import pytest

from tests._helpers import safe_import

pytestmark = pytest.mark.skipif(not safe_import("optuna"), reason="optuna not available")


def test_sweep_resume_and_list(tmp_path):
    import optuna
    from src.core.sweep import list_studies, resume_sweep

    storage = f"sqlite:///{tmp_path/'studies.db'}"
    study_name = "unit_list_resume"
    optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)
    studies = list_studies(storage)
    assert study_name in studies
    _ = resume_sweep(study_name, storage)

