from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "contracts" / "run_artifacts.md"


def test_run_artifact_contract_names_required_files():
    assert DOC.exists(), f"Missing {DOC}"
    text = DOC.read_text()

    for phrase in [
        ".hydra/config.yaml",
        "TensorBoard",
        "checkpoints/best.pt",
        "best_step_<step>.pt",
        "run_status.json",
        "run_summary.json",
        "pre_run/working_tree.diff",
        "reports/world_model_runs.csv",
        "diagnostics",
    ]:
        assert phrase in text


def test_run_artifact_contract_explains_agent_inspection_order():
    text = DOC.read_text()

    for phrase in [
        "read `run_summary.json`",
        "read `run_status.json`",
        "inspect the resolved config",
        "export TensorBoard scalars",
        "verify checkpoint links",
        "update `reports/world_model_runs.csv`",
    ]:
        assert phrase in text


def test_run_artifact_contract_documents_pull_and_code_movement_tools():
    text = DOC.read_text()

    for phrase in [
        "gpu_sync_patch.sh --paths",
        "gpu_run.sh --session",
        "gpu_pull_latest.sh --run",
        "gpu_pull_patch.sh",
        "--analyze",
    ]:
        assert phrase in text
