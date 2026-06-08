#!/usr/bin/env python3
"""Emit a tiny JSON snapshot for one RL Lab run.

This runs on the GPU host. Keep it dependency-free so it can run with system
Python even when the training venv is not active.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive for corrupt run files.
        return {"error": str(exc)}


def read_tail(path: Path, limit: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def parse_train_log(text: str, *, max_evals: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    eval_rows: list[dict[str, Any]] = []
    final_metrics: dict[str, Any] | None = None
    for line in text.splitlines():
        eval_match = re.search(r"Evaluation complete: mean_return=([-+0-9.eE]+)", line)
        if eval_match:
            eval_rows.append({"mean_return": float(eval_match.group(1))})
        if "Training complete:" in line:
            payload = line.split("Training complete:", 1)[1].strip()
            try:
                parsed = ast.literal_eval(payload)
                if isinstance(parsed, dict):
                    final_metrics = parsed
            except Exception:
                final_metrics = {"raw": payload}
    return eval_rows[-max_evals:], final_metrics


def checkpoint_files(run_path: Path) -> list[dict[str, Any]]:
    checkpoint_dir = run_path / "checkpoints"
    rows: list[dict[str, Any]] = []
    if not checkpoint_dir.exists():
        return rows
    for item in sorted(checkpoint_dir.rglob("*")):
        if not (item.is_file() or item.is_symlink()):
            continue
        try:
            stat_result = item.stat()
            row: dict[str, Any] = {
                "path": str(item.relative_to(run_path)),
                "size_bytes": stat_result.st_size,
                "kind": "symlink" if item.is_symlink() else "file",
            }
            if item.is_symlink():
                row["link_target"] = os.readlink(item)
            rows.append(row)
        except OSError as exc:
            rows.append({"path": str(item.relative_to(run_path)), "error": str(exc)})
    return rows


def build_snapshot(run_path: Path, *, tail_bytes: int, max_evals: int) -> dict[str, Any]:
    train_tail = read_tail(run_path / "train.log", tail_bytes)
    eval_rows, final_metrics = parse_train_log(train_tail, max_evals=max_evals)
    return {
        "schema_version": 1,
        "run_path": str(run_path),
        "run_name": run_path.name,
        "run_status": read_json(run_path / "run_status.json"),
        "run_summary": read_json(run_path / "run_summary.json"),
        "train_log": {
            "exists": (run_path / "train.log").exists(),
            "tail_bytes": tail_bytes,
            "recent_eval_returns": eval_rows,
            "final_metrics": final_metrics,
        },
        "checkpoint_files": checkpoint_files(run_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Absolute remote run path.")
    parser.add_argument("--tail-bytes", type=int, default=262144)
    parser.add_argument("--max-evals", type=int, default=20)
    args = parser.parse_args()

    snapshot = build_snapshot(Path(args.run), tail_bytes=args.tail_bytes, max_evals=args.max_evals)
    print(json.dumps(snapshot, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
