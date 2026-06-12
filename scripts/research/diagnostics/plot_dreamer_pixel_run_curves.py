#!/usr/bin/env python3
"""Plot a compact curve sheet for one Dreamer pixel run."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator


DEFAULT_TAGS = (
    "eval/return_mean",
    "world_model/total_loss",
    "world_model/observation_loss",
    "world_model/reward_loss",
    "world_model/continue_loss",
    "world_model/kl_loss",
    "actor/loss",
    "critic/loss",
    "actor/action_abs_mean",
    "actor/action_abs_max",
)


def export_dreamer_pixel_curves(
    run_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    tags: tuple[str, ...] = DEFAULT_TAGS,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    output_dir = Path(out_dir) if out_dir is not None else run_path / "diagnostics" / "dreamer_pixel_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_selected_scalars(run_path, set(tags))
    summary = _summarize(rows)
    summary["run_dir"] = str(run_path)
    summary["output_dir"] = str(output_dir)

    summary_path = output_dir / "dreamer_pixel_curve_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_paths = _plot_groups(rows, output_dir)
    summary["plots"] = [str(path) for path in plot_paths]
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _read_selected_scalars(run_path: Path, tags: set[str]) -> dict[str, list[tuple[int, float]]]:
    by_tag: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for event_file in sorted(run_path.rglob("events.out.tfevents*")):
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if tag not in tags:
                continue
            for event in accumulator.Scalars(tag):
                by_tag[tag].append((int(event.step), float(event.value)))
    for tag_rows in by_tag.values():
        tag_rows.sort(key=lambda row: row[0])
    return dict(sorted(by_tag.items()))


def _summarize(rows: dict[str, list[tuple[int, float]]]) -> dict[str, Any]:
    tags = {}
    for tag, values in rows.items():
        scalars = [value for _step, value in values]
        steps = [step for step, _value in values]
        tags[tag] = {
            "count": len(values),
            "first_step": steps[0],
            "last_step": steps[-1],
            "first_value": scalars[0],
            "last_value": scalars[-1],
            "min": min(scalars),
            "max": max(scalars),
        }
    return {"tags": tags}


def _plot_groups(rows: dict[str, list[tuple[int, float]]], output_dir: Path) -> list[Path]:
    _prepare_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {
        "eval_return": ("eval/return_mean",),
        "world_model_losses": (
            "world_model/total_loss",
            "world_model/observation_loss",
            "world_model/reward_loss",
            "world_model/continue_loss",
            "world_model/kl_loss",
        ),
        "controller_losses": ("actor/loss", "critic/loss"),
        "actor_actions": ("actor/action_abs_mean", "actor/action_abs_max"),
    }

    written: list[Path] = []
    for group_name, tags in groups.items():
        present = [tag for tag in tags if tag in rows]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for tag in present:
            steps = [step for step, _value in rows[tag]]
            values = [value for _step, value in rows[tag]]
            ax.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=tag)
        ax.set_title(group_name.replace("_", " ").title())
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        output = output_dir / f"{group_name}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        written.append(output)
    return written


def _prepare_matplotlib_cache() -> None:
    import os

    cache_root = Path(tempfile.gettempdir()) / "rl-lab-dreamer-pixel-curves"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(json.dumps(export_dreamer_pixel_curves(args.run_dir, out_dir=args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
