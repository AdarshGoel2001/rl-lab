import json
import os
import sys
import time
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class IssueLogger:
    """Thread-safe issue logger writing JSONL to repo-level test_issues.log"""

    def __init__(self, log_path: Optional[Path] = None):
        repo_root = Path(__file__).resolve().parent.parent
        self.log_path = log_path or (repo_root / "test_issues.log")
        self._lock = Lock()

        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        category: str,
        severity: str,
        message: str,
        file: Optional[str] = None,
        line: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "category": category,
            "severity": severity,
            "message": message,
            "file": file,
            "line": line,
            "extra": extra or {},
        }
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")


issue_logger = IssueLogger()


def safe_import(module: str) -> bool:
    """Try importing a module; return True if available, else log and return False."""
    try:
        __import__(module)
        return True
    except Exception as e:
        issue_logger.log(
            category="dependency",
            severity="info",
            message=f"Optional dependency not available: {module}",
            extra={"error": str(e)},
        )
        return False


def repo_files(suffix: str = ".py"):
    root = Path(__file__).resolve().parent.parent
    for path in root.rglob(f"*{suffix}"):
        # Skip venvs, experiments, and non-source folders
        parts = {p.name for p in path.parents}
        if any(k in parts for k in {"experiments", ".git", ".venv", "wandb", "tensorboard"}):
            continue
        yield path


def record_exception(category: str, exc: BaseException, file: Optional[str] = None):
    issue_logger.log(
        category=category,
        severity="error",
        message=f"Exception: {exc.__class__.__name__}: {exc}",
        file=file,
        extra={"traceback": traceback.format_exc()},
    )


def has_torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def has_torch_mps() -> bool:
    try:
        import torch
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def pytest_configure(config):
    """PyTest hook: mark custom categories if desired."""
    # No-op: markers can be added here if needed later
    return None

