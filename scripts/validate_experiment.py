#!/usr/bin/env python3
"""Cheap experiment contract check for humans and coding agents.

This intentionally does not run training. It verifies that a Hydra experiment
resolves, points at importable Python targets, and exposes the phase sequence an
agent should expect before attempting a model or workflow change.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _target_importable(target: str) -> tuple[bool, str | None]:
    module_name, _, attr_name = target.rpartition(".")
    if not module_name or not attr_name:
        return False, "target must be a dotted module path with an attribute name"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - exact import errors vary by optional dependency
        return False, f"could not import module {module_name}: {exc}"
    if not hasattr(module, attr_name):
        return False, f"module {module_name} has no attribute {attr_name}"
    return True, None


def _collect_targets(cfg: DictConfig) -> dict[str, str]:
    targets: dict[str, str] = {}
    for key in ("workflow", "environment", "buffer"):
        node = cfg.get(key)
        if node is not None and "_target_" in node:
            targets[key] = str(node["_target_"])

    for group_name in ("components", "controllers", "buffers"):
        group = cfg.get(group_name)
        if group is None:
            continue
        for name, node in group.items():
            if node is not None and "_target_" in node:
                targets[f"{group_name}.{name}"] = str(node["_target_"])
    return targets


def validate_experiment_config(
    experiment: str,
    *,
    budget: str | None = None,
    extra_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve an experiment config and return an agent-readable validation report."""

    overrides = [f"+experiment={experiment}"]
    if budget:
        overrides.append(f"budget={budget}")
    overrides.extend(extra_overrides or [])

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    targets = _collect_targets(cfg)
    errors: list[str] = []
    warnings: list[str] = []

    required_targets = ("workflow", "environment")
    for key in required_targets:
        if key not in targets:
            errors.append(f"missing required target: {key}")

    if "components" not in cfg or not cfg.components:
        errors.append("missing components group")
    if "controllers" not in cfg or not cfg.controllers:
        warnings.append("no controllers configured")
    if "buffer" not in targets and ("buffers" not in cfg or not cfg.buffers):
        errors.append("missing buffer or buffers configuration")

    for key, target in sorted(targets.items()):
        ok, error = _target_importable(target)
        if not ok and error:
            errors.append(f"{key}: {error}")

    phases = []
    if "training" in cfg and "phases" in cfg.training:
        phases = [str(phase.get("name", phase.get("type", ""))) for phase in cfg.training.phases]
    else:
        warnings.append("training.phases is not configured; PhaseScheduler will use its default online phase")

    return {
        "ok": not errors,
        "experiment": experiment,
        "budget": budget,
        "overrides": overrides,
        "targets": targets,
        "phases": phases,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Hydra experiment name, e.g. planet_cartpole")
    parser.add_argument("--budget", default=None, help="Optional budget group, e.g. planet_tiny")
    args, overrides = parser.parse_known_args(argv)
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    report = validate_experiment_config(
        args.experiment,
        budget=args.budget,
        extra_overrides=overrides,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
