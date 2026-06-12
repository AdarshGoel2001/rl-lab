#!/usr/bin/env python3
"""Summarize Dreamer TensorBoard scalars into actionable diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from scripts.research.export_tensorboard_run import find_event_files, read_scalar_rows


ScalarRow = Dict[str, Any]


DREAMER_TAGS = {
    "eval_return": "eval/return_mean",
    "actor_loss": "controller/actor_loss",
    "critic_loss": "controller/critic_loss",
    "lambda_return_mean": "controller/lambda_return_mean",
    "lambda_return_std": "controller/lambda_return_std",
    "imagined_reward_mean": "controller/imagined_reward_mean",
    "imagined_reward_std": "controller/imagined_reward_std",
    "imagined_continue_mean": "controller/imagined_continue_mean",
    "imagined_value_mean": "controller/imagined_value_mean",
    "critic_target_mean": "controller/critic_target_mean",
    "critic_target_std": "controller/critic_target_std",
    "critic_value_mean": "controller/critic_value_mean",
    "critic_value_std": "controller/critic_value_std",
    "action_abs_mean": "controller/action_abs_mean",
    "action_abs_max": "controller/action_abs_max",
    "action_std": "controller/action_std",
    "world_model_total_loss": "train/world_model/total_loss",
    "world_model_reward_loss": "train/world_model/reward_loss",
    "world_model_kl_loss": "train/world_model/kl_loss",
    "world_model_observation_loss": "train/world_model/observation_loss",
}


def load_scalar_rows(path: str | Path) -> List[ScalarRow]:
    """Load scalar rows from analysis/scalars.csv or TensorBoard event files."""
    run_path = Path(path)
    csv_path = run_path / "analysis" / "scalars.csv"
    if run_path.name == "scalars.csv":
        csv_path = run_path
    elif run_path.name == "analysis":
        csv_path = run_path / "scalars.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [_normalize_row(row) for row in rows]

    event_files = find_event_files(run_path)
    if not event_files:
        raise FileNotFoundError(f"No scalars.csv or TensorBoard event files found under '{run_path}'.")
    return read_scalar_rows(event_files)


def summarize_dreamer_scalars(rows: Iterable[ScalarRow]) -> Dict[str, Any]:
    """Build a compact health summary from Dreamer scalar rows."""
    rows_by_tag: Dict[str, List[ScalarRow]] = defaultdict(list)
    for row in rows:
        rows_by_tag[str(row["tag"])].append(row)
    for tag_rows in rows_by_tag.values():
        tag_rows.sort(key=lambda row: (int(row["step"]), float(row.get("wall_time", 0.0))))

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "missing_tags": [],
        "eval": {},
        "world_model": {},
        "controller": {},
        "warnings": [],
    }

    for name, tag in DREAMER_TAGS.items():
        if tag not in rows_by_tag:
            summary["missing_tags"].append(tag)

    eval_rows = rows_by_tag.get(DREAMER_TAGS["eval_return"], [])
    if eval_rows:
        best = max(eval_rows, key=lambda row: float(row["value"]))
        latest = eval_rows[-1]
        summary["eval"] = {
            "count": len(eval_rows),
            "latest_return_mean": float(latest["value"]),
            "latest_step": int(latest["step"]),
            "best_return_mean": float(best["value"]),
            "best_step": int(best["step"]),
        }

    for name in [
        "world_model_total_loss",
        "world_model_reward_loss",
        "world_model_kl_loss",
        "world_model_observation_loss",
    ]:
        _copy_latest(rows_by_tag, DREAMER_TAGS[name], summary["world_model"], name)

    for name in [
        "actor_loss",
        "critic_loss",
        "lambda_return_mean",
        "lambda_return_std",
        "imagined_reward_mean",
        "imagined_reward_std",
        "imagined_continue_mean",
        "imagined_value_mean",
        "critic_target_mean",
        "critic_target_std",
        "critic_value_mean",
        "critic_value_std",
        "action_abs_mean",
        "action_abs_max",
        "action_std",
    ]:
        _copy_latest(rows_by_tag, DREAMER_TAGS[name], summary["controller"], name)

    controller = summary["controller"]
    if "critic_target_mean" in controller and "critic_value_mean" in controller:
        gap = float(controller["critic_target_mean"]) - float(controller["critic_value_mean"])
        controller["critic_target_value_gap"] = gap
        controller["critic_target_value_abs_gap"] = abs(gap)

    action_abs_mean = controller.get("action_abs_mean")
    action_abs_max = controller.get("action_abs_max")
    if action_abs_mean is not None and float(action_abs_mean) > 0.9:
        summary["warnings"].append("actor_action_abs_mean_high")
    if action_abs_max is not None and float(action_abs_max) > 0.995:
        summary["warnings"].append("actor_action_abs_max_saturated")

    continue_mean = controller.get("imagined_continue_mean")
    if continue_mean is not None and not 0.01 <= float(continue_mean) <= 1.0:
        summary["warnings"].append("imagined_continue_mean_out_of_range")

    for group_name in ("world_model", "controller"):
        for key, value in list(summary[group_name].items()):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                summary["warnings"].append(f"{group_name}_{key}_nonfinite")

    return summary


def write_dreamer_diagnostics(summary: Mapping[str, Any], out_dir: str | Path) -> Dict[str, Path]:
    """Write Dreamer diagnostics as JSON and Markdown."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dreamer_diagnostics_summary.json"
    report_path = output_dir / "dreamer_diagnostics_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")
    return {"summary": summary_path, "report": report_path}


def render_markdown_report(summary: Mapping[str, Any]) -> str:
    """Render a concise Markdown report for humans and agents."""
    lines = ["# Dreamer TensorBoard Diagnostics", ""]
    eval_summary = summary.get("eval", {})
    if eval_summary:
        lines.extend(
            [
                "## Eval",
                "",
                f"- latest return mean: {eval_summary.get('latest_return_mean')} at step {eval_summary.get('latest_step')}",
                f"- best return mean: {eval_summary.get('best_return_mean')} at step {eval_summary.get('best_step')}",
                "",
            ]
        )
    for section in ("world_model", "controller"):
        values = summary.get(section, {})
        if not values:
            continue
        lines.extend([f"## {section.replace('_', ' ').title()}", ""])
        for key, value in sorted(values.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")
    warnings = summary.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")
    missing = summary.get("missing_tags", [])
    if missing:
        lines.extend(["## Missing Tags", ""])
        for tag in missing:
            lines.append(f"- {tag}")
        lines.append("")
    return "\n".join(lines)


def _copy_latest(
    rows_by_tag: Mapping[str, List[ScalarRow]],
    tag: str,
    out: Dict[str, Any],
    name: str,
) -> None:
    rows = rows_by_tag.get(tag)
    if not rows:
        return
    latest = rows[-1]
    out[name] = float(latest["value"])
    out[f"{name}_step"] = int(latest["step"])


def _normalize_row(row: Mapping[str, Any]) -> ScalarRow:
    return {
        "tag": str(row["tag"]),
        "step": int(row["step"]),
        "wall_time": float(row.get("wall_time", 0.0)),
        "value": float(row["value"]),
        "event_file": str(row.get("event_file", "")),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory, analysis directory parent, or analysis/scalars.csv path.")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to <run_dir>/diagnostics/dreamer_tensorboard.")
    args = parser.parse_args(argv)

    run_path = Path(args.run_dir)
    rows = load_scalar_rows(run_path)
    summary = summarize_dreamer_scalars(rows)
    if args.out is not None:
        out_dir = Path(args.out)
    elif run_path.name == "scalars.csv":
        out_dir = run_path.parent / "dreamer_tensorboard"
    else:
        out_dir = run_path / "diagnostics" / "dreamer_tensorboard"
    outputs = write_dreamer_diagnostics(summary, out_dir)
    print(json.dumps({"summary": str(outputs["summary"]), "report": str(outputs["report"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
