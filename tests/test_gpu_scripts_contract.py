from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = ROOT / "scripts" / "GPU"


def test_gpu_scripts_exist_and_are_executable():
    expected = [
        "gpu_common.sh",
        "gpu_status.sh",
        "gpu_run.sh",
        "gpu_tail.sh",
        "gpu_metrics.sh",
        "gpu_run_snapshot.sh",
        "gpu_run_snapshot_remote.py",
        "gpu_sync_patch.sh",
        "gpu_pull_latest.sh",
        "gpu_pull_patch.sh",
    ]

    for name in expected:
        script = GPU_DIR / name
        assert script.exists(), f"Missing {script}"
        if name not in {"gpu_common.sh", "gpu_run_snapshot_remote.py"}:
            assert script.stat().st_mode & 0o111, f"{script} should be executable"


def test_gpu_scripts_keep_remote_loop_simple():
    status = (GPU_DIR / "gpu_status.sh").read_text()
    run = (GPU_DIR / "gpu_run.sh").read_text()
    tail = (GPU_DIR / "gpu_tail.sh").read_text()
    metrics = (GPU_DIR / "gpu_metrics.sh").read_text()
    snapshot = (GPU_DIR / "gpu_run_snapshot.sh").read_text()
    snapshot_remote = (GPU_DIR / "gpu_run_snapshot_remote.py").read_text()
    sync = (GPU_DIR / "gpu_sync_patch.sh").read_text()
    pull = (GPU_DIR / "gpu_pull_latest.sh").read_text()
    pull_patch = (GPU_DIR / "gpu_pull_patch.sh").read_text()
    common = (GPU_DIR / "gpu_common.sh").read_text()

    assert "gpu_ssh" in common
    assert "gpu_scp" in common
    assert "gpu_rsync_ssh" in common
    assert "ssh -T" in common
    assert "RequestTTY=no" in common
    assert "BatchMode=yes" in common
    assert "ConnectTimeout" in common
    assert "RL_LAB_GPU_SSH_HOST" in common

    assert "gpu_common.sh" in status
    assert "gpu_ssh" in status and "wsl" in common
    assert "/usr/lib/wsl/lib/nvidia-smi" in status
    assert "tmux ls" in status
    assert "run_status.json" in status
    assert "RL_LAB_GPU_STATUS_GIT" in status
    assert status.index("== gpu ==") < status.index("== git ==")

    assert "tmux new-session" in run
    assert "gpu_common.sh" in run
    assert "gpu_ssh" in run
    assert "--override" in run
    assert "HYDRA_OVERRIDES" in run
    assert "RL_LAB_GPU_RUN_LOG" in run
    assert ".agent_runs" in run
    assert "--dry-run" in run
    assert "scripts/train.py" in run
    assert "planet_dmc_cartpole_swingup" in run

    assert "tail" in tail
    assert "gpu_common.sh" in tail
    assert "gpu_ssh" in tail
    assert "train.log" in tail

    assert "--run" in metrics
    assert "RUN_PATH" in metrics
    assert "gpu_common.sh" in metrics
    assert "gpu_ssh" in metrics
    assert "events.out.tfevents" in metrics
    assert "checkpoints" in metrics
    assert "run_status.json" in metrics
    assert 'find \\"\\$latest\\" -type f' in metrics
    assert 'find \\"\\$latest\\" runs' not in metrics

    assert "--run" in snapshot
    assert "RUN_PATH" in snapshot
    assert "gpu_run_snapshot_remote.py" in snapshot
    assert "gpu_common.sh" in snapshot
    assert "gpu_ssh" in snapshot
    assert "base64" in snapshot
    assert "dd if=" in snapshot
    assert "run_status.json" in snapshot_remote
    assert "train.log" in snapshot_remote
    assert "Evaluation complete" in snapshot_remote
    assert "checkpoint_files" in snapshot_remote

    assert "git diff --binary HEAD" in sync
    assert "git ls-files --others --exclude-standard" in sync
    assert "--paths" in sync
    assert "PATH_FILTERS" in sync
    assert "--exclude" in sync
    assert "EXCLUDE_FILTERS" in sync
    assert "--dry-run" in sync
    assert "scp" in sync
    assert "gpu_common.sh" in sync
    assert "gpu_ssh" in sync
    assert "gpu_scp" in sync
    assert "git apply --check" in sync
    assert "--reset-remote" in sync
    assert "git reset --hard" in sync
    assert not re.search(r"(^|[^A-Za-z0-9_])status=", sync)

    assert "rsync" in pull
    assert "base64" in pull
    assert "falling back to base64 file copy" in pull
    assert "--run" in pull
    assert "RUN_PATH" in pull
    assert "--analyze" in pull
    assert "export_tensorboard_run.py" in pull
    assert "gpu_common.sh" in pull
    assert "gpu_ssh" in pull
    assert "gpu_rsync_ssh" in pull
    assert "train.log" in pull
    assert "logs/bash_output.log" in pull
    assert ".hydra/config.yaml" in pull
    assert "run_status.json" in pull
    assert "run_summary.json" in pull
    assert "/pre_run/" in pull
    assert "/diagnostics/" in pull
    assert "checkpoints/final.json" in pull
    assert "checkpoint_manifest.json" in pull
    assert "/checkpoints/latest.pt" in pull
    assert "/checkpoints/best.pt" in pull
    assert "/checkpoints/best_step_*.pt" in pull
    assert "/checkpoints/step_*.pt" in pull
    assert "--checkpoint" in pull
    assert "--analyze" in pull
    analyze_case = re.search(r"--analyze\)(.*?)shift", pull, re.S)
    assert analyze_case is not None
    assert "INCLUDE_CHECKPOINT=1" not in analyze_case.group(1)
    assert "/runs/" in pull
    assert "/runs/**/events.out.tfevents*" in pull

    assert "git diff --binary HEAD" in pull_patch
    assert "git ls-files --others --exclude-standard" in pull_patch
    assert "git apply --check" in pull_patch
    assert "--apply" in pull_patch
    assert "remote_code_patches" in pull_patch
    assert "--paths" in pull_patch
    assert "--exclude" in pull_patch
    assert not re.search(r"(^|[^A-Za-z0-9_])status=", pull_patch)
