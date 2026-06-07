#!/usr/bin/env python3
"""Export TensorBoard scalar data and plots for one experiment run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from tensorboard.backend.event_processing import event_accumulator


ScalarRow = Dict[str, Any]


def find_event_files(run_dir: Path) -> List[Path]:
    """Return TensorBoard event files under an experiment or runs directory."""
    return sorted(path for path in run_dir.rglob("events.out.tfevents*") if path.is_file())


def read_scalar_rows(event_files: Iterable[Path]) -> List[ScalarRow]:
    """Read scalar events from TensorBoard event files."""
    rows: List[ScalarRow] = []
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(str(event_file))
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                rows.append(
                    {
                        "tag": tag,
                        "step": int(event.step),
                        "wall_time": float(event.wall_time),
                        "value": float(event.value),
                        "event_file": str(event_file),
                    }
                )
    rows.sort(key=lambda row: (row["tag"], row["step"], row["wall_time"]))
    return rows


def summarize_scalars(rows: Iterable[ScalarRow], event_files: Iterable[Path]) -> Dict[str, Any]:
    """Build a compact per-tag scalar summary."""
    by_tag: Dict[str, List[ScalarRow]] = defaultdict(list)
    for row in rows:
        by_tag[str(row["tag"])].append(row)

    tags: Dict[str, Dict[str, Any]] = {}
    for tag, tag_rows in sorted(by_tag.items()):
        values = [float(row["value"]) for row in tag_rows]
        steps = [int(row["step"]) for row in tag_rows]
        tags[tag] = {
            "count": len(tag_rows),
            "first_step": steps[0],
            "last_step": steps[-1],
            "first_value": values[0],
            "last_value": values[-1],
            "min": min(values),
            "max": max(values),
        }

    return {
        "event_files": [str(path) for path in event_files],
        "tags": tags,
        "groups": {
            "loss": _select_loss_tags(tags),
            "collect": _select_collect_tags(tags),
            "eval": _select_eval_tags(tags),
            "eval_return": _select_eval_return_tags(tags),
        },
    }


def write_scalars_csv(rows: Iterable[ScalarRow], path: Path) -> None:
    """Write scalar events to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tag", "step", "wall_time", "value", "event_file"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_json(summary: Mapping[str, Any], path: Path) -> None:
    """Write scalar summary to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def plot_scalar_groups(rows: Iterable[ScalarRow], groups: Mapping[str, List[str]], out_dir: Path) -> List[Path]:
    """Plot configured scalar groups to PNG files."""
    _prepare_matplotlib_cache()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_tag: Dict[str, List[ScalarRow]] = defaultdict(list)
    for row in rows:
        by_tag[str(row["tag"])].append(row)

    written: List[Path] = []
    for group_name, tags in groups.items():
        if not tags:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for tag in tags:
            tag_rows = sorted(by_tag[tag], key=lambda row: (row["step"], row["wall_time"]))
            steps = [int(row["step"]) for row in tag_rows]
            values = [float(row["value"]) for row in tag_rows]
            ax.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=tag)
        ax.set_title(f"{group_name.title()} Curves")
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        output = out_dir / f"{group_name}_curves.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        written.append(output)
    return written


def export_tensorboard_run(run_dir: str | Path, *, out_dir: str | Path | None = None) -> Dict[str, Any]:
    """Export scalar summaries, CSV rows, and grouped curve plots for a run."""
    run_path = Path(run_dir)
    output_dir = Path(out_dir) if out_dir is not None else run_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    event_files = find_event_files(run_path)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under '{run_path}'.")

    rows = read_scalar_rows(event_files)
    summary = summarize_scalars(rows, event_files)

    write_summary_json(summary, output_dir / "scalar_summary.json")
    write_scalars_csv(rows, output_dir / "scalars.csv")
    plot_scalar_groups(rows, summary["groups"], output_dir)
    return summary


def _select_loss_tags(tags: Mapping[str, Any]) -> List[str]:
    return [tag for tag in tags if "loss" in tag.lower()]


def _select_collect_tags(tags: Mapping[str, Any]) -> List[str]:
    return [tag for tag in tags if tag.startswith("collect/")]


def _select_eval_tags(tags: Mapping[str, Any]) -> List[str]:
    return [tag for tag in tags if tag.startswith("eval/")]


def _select_eval_return_tags(tags: Mapping[str, Any]) -> List[str]:
    return [tag for tag in tags if tag.startswith("eval/return")]


def _prepare_matplotlib_cache() -> None:
    """Point matplotlib/fontconfig caches at a writable temp directory."""
    cache_root = Path(tempfile.gettempdir()) / "rl-lab-matplotlib-cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Experiment directory or directory containing TensorBoard event files.")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to <run_dir>/analysis.")
    args = parser.parse_args(argv)

    summary = export_tensorboard_run(args.run_dir, out_dir=args.out)
    print(json.dumps({"tags": list(summary["tags"].keys()), "output": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
