import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reports" / "world_model_runs.csv"
DOC = ROOT / "docs" / "roadmap" / "run_manifest.md"

REQUIRED_COLUMNS = [
    "run_id",
    "status",
    "chapter",
    "paper",
    "year",
    "model_family",
    "implementation",
    "config",
    "environment",
    "modality",
    "eval_layer",
    "budget_tier",
    "remote_host",
    "gpu",
    "run_dir",
    "artifact_dir",
    "seed",
    "env_steps",
    "updates",
    "wall_time_seconds",
    "eval_return_mean",
    "eval_return_std",
    "best_eval_return_mean",
    "primary_loss",
    "claim",
    "failure_notes",
    "next_action",
]


def test_world_model_run_manifest_has_stable_schema():
    assert MANIFEST.exists(), f"Missing {MANIFEST}"

    with MANIFEST.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == REQUIRED_COLUMNS
        rows = list(reader)

    assert rows, "Manifest should contain at least one row to anchor the format."
    for row in rows:
        assert row["run_id"]
        assert row["status"] in {"planned", "running", "complete", "failed", "superseded"}
        assert row["paper"]
        assert row["config"]
        assert row["environment"]
        assert row["eval_layer"]


def test_world_model_run_manifest_is_not_git_ignored():
    result = subprocess.run(
        ["git", "check-ignore", str(MANIFEST.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout or result.stderr


def test_run_manifest_doc_explains_how_to_use_csv():
    assert DOC.exists(), f"Missing {DOC}"
    text = DOC.read_text()

    assert "reports/world_model_runs.csv" in text
    assert "run_id" in text
    assert "status" in text
    assert "eval_return_mean" in text
    assert "artifact_dir" in text
    assert "Do not use the manifest as a substitute for logs" in text
