import csv
import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")


ROOT = Path(__file__).resolve().parents[1]


def test_train_tiny():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "+experiment=og_wm_carracing",
            "budget=tiny",
            "experiment.device=cpu",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout
    latest_file = ROOT / "runs" / "latest_og_wm_run.txt"
    assert latest_file.exists()
    run_dir = Path(latest_file.read_text().strip())
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    metrics_csv = run_dir / "metrics.csv"
    assert metrics_csv.exists()

    with metrics_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 1
