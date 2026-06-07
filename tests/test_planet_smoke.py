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
            "+experiment=planet_cartpole",
            "budget=planet_tiny",
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
    experiment_dirs = sorted((ROOT / "experiments").glob("planet_cartpole_tiny-*"))
    assert experiment_dirs
    run_dir = experiment_dirs[-1]
    assert (run_dir / "checkpoints" / "final.pt").exists()
    assert (run_dir / "run_summary.json").exists()
