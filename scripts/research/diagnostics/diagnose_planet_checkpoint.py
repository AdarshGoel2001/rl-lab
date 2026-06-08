#!/usr/bin/env python3
"""Diagnose PlaNet reward calibration and open-loop rollout accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import (  # noqa: E402
    build_controllers,
    build_optimizers,
    build_world_model_components,
    set_seeds,
)
from src.components.representation_learners import RSSMState  # noqa: E402
from src.orchestration import Orchestrator  # noqa: E402
from src.utils.config import resolve_device  # noqa: E402


MetricDict = Dict[str, float]
CsvRow = Dict[str, Any]
PLANET_REQUIRED_CHECKPOINT_COMPONENTS = ("representation_learner", "reward_predictor")


def compute_regression_metrics(predicted: np.ndarray, actual: np.ndarray, *, prefix: str) -> MetricDict:
    """Compute flat prediction-vs-target regression metrics."""
    pred = np.asarray(predicted, dtype=np.float64).reshape(-1)
    target = np.asarray(actual, dtype=np.float64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f"predicted and actual must have the same flat shape, got {pred.shape} and {target.shape}.")
    if pred.size == 0:
        raise ValueError("Cannot compute regression metrics for empty arrays.")

    error = pred - target
    metrics: MetricDict = {
        f"{prefix}/count": float(pred.size),
        f"{prefix}/mse": float(np.mean(error**2)),
        f"{prefix}/mae": float(np.mean(np.abs(error))),
        f"{prefix}/bias": float(np.mean(error)),
        f"{prefix}/predicted_mean": float(np.mean(pred)),
        f"{prefix}/actual_mean": float(np.mean(target)),
    }
    if float(np.std(pred)) < 1e-12 or float(np.std(target)) < 1e-12:
        metrics[f"{prefix}/corr"] = float("nan")
    else:
        metrics[f"{prefix}/corr"] = float(np.corrcoef(pred, target)[0, 1])
    return metrics


def compute_horizon_return_metrics(
    predicted_rewards: np.ndarray,
    actual_rewards: np.ndarray,
    *,
    horizons: Sequence[int],
) -> Tuple[List[CsvRow], MetricDict]:
    """Compare predicted and actual cumulative rewards at each rollout horizon."""
    pred = np.asarray(predicted_rewards, dtype=np.float64)
    actual = np.asarray(actual_rewards, dtype=np.float64)
    if pred.shape != actual.shape:
        raise ValueError(f"predicted_rewards and actual_rewards must match, got {pred.shape} and {actual.shape}.")
    if pred.ndim != 2:
        raise ValueError(f"Expected reward arrays with shape [windows, horizon], got {pred.shape}.")
    if pred.shape[0] == 0:
        raise ValueError("Cannot compute horizon metrics without rollout windows.")

    max_len = pred.shape[1]
    rows: List[CsvRow] = []
    summary: MetricDict = {}
    for horizon in horizons:
        h = int(horizon)
        if h <= 0 or h > max_len:
            continue
        pred_return = np.sum(pred[:, :h], axis=1)
        actual_return = np.sum(actual[:, :h], axis=1)
        metrics = compute_regression_metrics(pred_return, actual_return, prefix=f"open_loop/h{h}_return")
        row = {
            "horizon": h,
            "count": int(metrics[f"open_loop/h{h}_return/count"]),
            "return_mse": metrics[f"open_loop/h{h}_return/mse"],
            "return_mae": metrics[f"open_loop/h{h}_return/mae"],
            "return_bias": metrics[f"open_loop/h{h}_return/bias"],
            "return_corr": metrics[f"open_loop/h{h}_return/corr"],
            "predicted_return_mean": metrics[f"open_loop/h{h}_return/predicted_mean"],
            "actual_return_mean": metrics[f"open_loop/h{h}_return/actual_mean"],
        }
        rows.append(row)
        summary.update(
            {
                f"open_loop/h{h}_return_count": float(row["count"]),
                f"open_loop/h{h}_return_mse": float(row["return_mse"]),
                f"open_loop/h{h}_return_mae": float(row["return_mae"]),
                f"open_loop/h{h}_return_bias": float(row["return_bias"]),
                f"open_loop/h{h}_return_corr": float(row["return_corr"]),
                f"open_loop/h{h}_predicted_return_mean": float(row["predicted_return_mean"]),
                f"open_loop/h{h}_actual_return_mean": float(row["actual_return_mean"]),
            }
        )
    return rows, summary


def write_diagnostic_outputs(
    *,
    out_dir: str | Path,
    summary: Mapping[str, Any],
    reward_rows: Iterable[Mapping[str, Any]],
    horizon_rows: Iterable[Mapping[str, Any]],
    tensorboard_logdir: str | Path | None = None,
    tensorboard_step: int = 0,
) -> Dict[str, Path]:
    """Persist diagnostics as JSON, CSV, plots, and optional TensorBoard events."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_rows_list = [dict(row) for row in reward_rows]
    horizon_rows_list = [dict(row) for row in horizon_rows]

    summary_path = output_dir / "diagnostics_summary.json"
    reward_csv_path = output_dir / "reward_calibration.csv"
    horizon_csv_path = output_dir / "open_loop_horizon_metrics.csv"
    reward_plot_path = output_dir / "reward_pred_vs_actual.png"
    horizon_plot_path = output_dir / "open_loop_errors.png"

    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(reward_csv_path, reward_rows_list)
    _write_csv(horizon_csv_path, horizon_rows_list)
    plot_reward_calibration(reward_rows_list, reward_plot_path)
    plot_open_loop_errors(horizon_rows_list, horizon_plot_path)

    outputs = {
        "summary": summary_path,
        "reward_csv": reward_csv_path,
        "horizon_csv": horizon_csv_path,
        "reward_plot": reward_plot_path,
        "horizon_plot": horizon_plot_path,
    }
    if tensorboard_logdir is not None:
        tb_dir = Path(tensorboard_logdir)
        write_tensorboard_outputs(
            logdir=tb_dir,
            summary=summary,
            horizon_rows=horizon_rows_list,
            reward_plot=reward_plot_path,
            horizon_plot=horizon_plot_path,
            checkpoint_step=_tensorboard_checkpoint_step(summary, fallback=tensorboard_step),
        )
        outputs["tensorboard_logdir"] = tb_dir
    return outputs


def diagnose_planet_checkpoint(
    *,
    checkpoint: str | Path,
    experiment: str,
    budget: str | None = None,
    out_dir: str | Path | None = None,
    steps: int = 256,
    open_loop_horizon: int = 12,
    horizons: Sequence[int] | None = None,
    policy: str = "random",
    device_override: str | None = None,
    num_envs: int | None = None,
    seed: int | None = None,
    overrides: Sequence[str] | None = None,
    tensorboard: bool = True,
    tensorboard_logdir: str | Path | None = None,
) -> Dict[str, Any]:
    """Load a PlaNet checkpoint and write offline diagnostics."""
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint_metadata = validate_planet_checkpoint_schema(checkpoint_path)
    checkpoint_global_step = read_checkpoint_global_step(checkpoint_path)
    output_dir = Path(out_dir) if out_dir is not None else checkpoint_path.parent.parent / "diagnostics"

    cfg = compose_experiment_config(
        experiment=experiment,
        budget=budget,
        device_override=device_override,
        num_envs=num_envs,
        seed=seed,
        overrides=overrides or (),
    )
    orchestrator = build_initialized_orchestrator(
        cfg,
        checkpoint_path,
        experiment_dir=diagnostic_orchestrator_dir(checkpoint=checkpoint_path, out_dir=output_dir),
    )
    workflow = orchestrator.workflow

    trajectory = collect_diagnostic_trajectory(
        workflow,
        steps=steps,
        policy=policy,
        seed=seed,
    )
    reward_rows, reward_metrics = compute_reward_calibration(workflow, trajectory)
    horizon_rows, horizon_metrics = compute_open_loop_diagnostics(
        workflow,
        trajectory,
        open_loop_horizon=open_loop_horizon,
        horizons=horizons,
    )

    summary: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "experiment": experiment,
        "budget": budget,
        "policy": policy,
        "steps": int(steps),
        "num_envs": int(trajectory["observations"].shape[1]),
        "open_loop_horizon": int(open_loop_horizon),
        "checkpoint_global_step": float(checkpoint_global_step),
        "checkpoint_metadata": checkpoint_metadata,
    }
    summary.update(reward_metrics)
    summary.update(horizon_metrics)

    tb_logdir = None
    if tensorboard:
        tb_logdir = (
            Path(tensorboard_logdir)
            if tensorboard_logdir is not None
            else checkpoint_path.parent.parent / "runs" / "diagnostics_planet_reward_open_loop"
        )

    outputs = write_diagnostic_outputs(
        out_dir=output_dir,
        summary=summary,
        reward_rows=reward_rows,
        horizon_rows=horizon_rows,
        tensorboard_logdir=tb_logdir,
    )
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    outputs["summary"].write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")

    cleanup = getattr(orchestrator, "cleanup", None)
    if callable(cleanup):
        cleanup()
    return summary


def compose_experiment_config(
    *,
    experiment: str,
    budget: str | None,
    device_override: str | None,
    num_envs: int | None,
    seed: int | None,
    overrides: Sequence[str],
) -> DictConfig:
    config_dir = str(ROOT / "configs")
    hydra_overrides = [f"+experiment={experiment}"]
    if budget:
        hydra_overrides.append(f"budget={budget}")
    if device_override:
        hydra_overrides.append(f"experiment.device={device_override}")
    if num_envs is not None:
        hydra_overrides.append(f"++environment.num_envs={int(num_envs)}")
    if seed is not None:
        hydra_overrides.append(f"experiment.seed={int(seed)}")
        hydra_overrides.append(f"++environment.seed={int(seed)}")
    hydra_overrides.extend(overrides)

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="config", overrides=hydra_overrides)


def read_checkpoint_global_step(checkpoint_path: Path) -> int:
    """Read the training global_step from a checkpoint without restoring it."""
    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return 0
    try:
        return int(checkpoint.get("global_step", 0))
    except (TypeError, ValueError):
        return 0


def validate_planet_checkpoint_schema(
    checkpoint_path: str | Path,
    *,
    required_components: Sequence[str] = PLANET_REQUIRED_CHECKPOINT_COMPONENTS,
) -> Dict[str, Any]:
    """Fail before diagnostics if the checkpoint cannot contain learned PlaNet weights."""
    path = Path(checkpoint_path).expanduser()
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"Could not read checkpoint for diagnostics: {path}") from exc

    components = checkpoint.get("components") if isinstance(checkpoint, Mapping) else None
    if not isinstance(components, Mapping):
        raise ValueError(f"Checkpoint {path} is missing required component weights: components")

    component_names = sorted(str(name) for name in components.keys())
    missing = [name for name in required_components if name not in components]
    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing required component weights: {', '.join(missing)}. "
            f"Found: {', '.join(component_names) if component_names else 'none'}."
        )

    empty = [
        name
        for name in required_components
        if isinstance(components.get(name), Mapping) and len(components.get(name, {})) == 0
    ]
    if empty:
        raise ValueError(
            f"Checkpoint {path} has empty required component weights: {', '.join(empty)}."
        )

    return {
        "global_step": int(checkpoint.get("global_step", 0)),
        "component_names": component_names,
        "required_component_names": list(required_components),
    }


def diagnostic_orchestrator_dir(*, checkpoint: str | Path, out_dir: str | Path) -> Path:
    """Put diagnostic orchestrator side effects under diagnostics, not experiments/."""
    _ = Path(checkpoint)
    return Path(out_dir) / "_orchestrator"


def build_initialized_orchestrator(
    cfg: DictConfig,
    checkpoint_path: Path,
    *,
    experiment_dir: str | Path | None = None,
) -> Orchestrator:
    device = resolve_device(cfg.experiment.device)
    set_seeds(cfg.experiment.get("seed", None))

    workflow = instantiate(cfg.workflow)
    components = build_world_model_components(cfg, device)
    controllers, controller_manager = build_controllers(cfg, device)
    optimizers = build_optimizers(cfg, components, controllers)
    train_environment = instantiate(cfg.environment)
    eval_environment = instantiate(cfg.evaluation) if "evaluation" in cfg and cfg.evaluation is not None else None

    buffer_num_envs = getattr(train_environment, "num_envs", None)
    if buffer_num_envs is None:
        buffer_num_envs = cfg.environment.get("num_envs", 1)

    buffers: Dict[str, Any] = {}
    if "buffers" in cfg and cfg.buffers is not None:
        for name, buffer_cfg in cfg.buffers.items():
            buffers[name] = _instantiate_buffer(buffer_cfg, device=device, num_envs=buffer_num_envs)
    if not buffers:
        buffers["replay"] = _instantiate_buffer(cfg.buffer, device=device, num_envs=buffer_num_envs)

    orchestrator = Orchestrator(
        cfg,
        workflow,
        experiment_dir=Path(experiment_dir) if experiment_dir is not None else None,
        components=components,
        optimizers=optimizers,
        controllers=controllers,
        controller_manager=controller_manager,
        buffers=buffers,
        train_environment=train_environment,
        eval_environment=eval_environment,
    )
    orchestrator.load_checkpoint(checkpoint_path, mode="warm_start")
    orchestrator.initialize()
    workflow.rssm.eval()
    workflow.reward_predictor.eval()
    if getattr(workflow, "continue_predictor", None) is not None:
        workflow.continue_predictor.eval()
    if getattr(workflow, "observation_predictor", None) is not None:
        workflow.observation_predictor.eval()
    return orchestrator


def collect_diagnostic_trajectory(
    workflow: Any,
    *,
    steps: int,
    policy: str,
    seed: int | None,
) -> Dict[str, np.ndarray]:
    """Collect a short real trajectory for offline diagnostics."""
    env = workflow.eval_environment or workflow.environment
    if env is None:
        raise RuntimeError("PlaNet diagnostics require a train or eval environment.")
    reset = env.reset(seed=seed)
    obs = reset[0] if isinstance(reset, tuple) else reset
    obs = _ensure_batch(np.asarray(obs, dtype=np.float32))
    num_envs = int(obs.shape[0])
    action_dim = int(getattr(workflow, "action_dim"))
    prev_action = torch.zeros(num_envs, action_dim, device=workflow.device)
    done = np.zeros(num_envs, dtype=bool)

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[np.ndarray] = []
    dones: List[np.ndarray] = []

    for _ in range(int(steps)):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=workflow.device)
        with torch.no_grad():
            action_tensor = select_diagnostic_action(
                workflow,
                obs_tensor=obs_tensor,
                prev_action=prev_action,
                done=done,
                policy=policy,
            )
        next_obs, reward, next_done, _info = env.step(action_tensor)
        observations.append(obs.copy())
        actions.append(action_tensor.detach().cpu().numpy().copy())
        rewards.append(np.asarray(reward, dtype=np.float32).reshape(num_envs).copy())
        dones.append(np.asarray(next_done, dtype=bool).reshape(num_envs).copy())
        obs = _ensure_batch(np.asarray(next_obs, dtype=np.float32))
        done = np.asarray(next_done, dtype=bool).reshape(num_envs)
        prev_action = action_tensor.detach()

    if not observations:
        raise ValueError("No diagnostic trajectory was collected.")
    return {
        "observations": np.stack(observations, axis=0),
        "actions": np.stack(actions, axis=0),
        "rewards": np.stack(rewards, axis=0),
        "dones": np.stack(dones, axis=0),
    }


def select_diagnostic_action(
    workflow: Any,
    *,
    obs_tensor: torch.Tensor,
    prev_action: torch.Tensor,
    done: np.ndarray,
    policy: str,
) -> torch.Tensor:
    """Select action for diagnostics without mutating training buffers."""
    policy_name = str(policy).lower()
    if policy_name == "random":
        low, high = workflow.get_action_bounds()
        low = low.to(workflow.device).view(1, -1)
        high = high.to(workflow.device).view(1, -1)
        return low + torch.rand(obs_tensor.shape[0], low.shape[-1], device=workflow.device) * (high - low)
    if policy_name == "actor":
        actor = workflow.controllers.get("actor")
        if actor is None:
            raise RuntimeError("policy='actor' requested but no actor controller is configured.")
        return actor.act(obs_tensor)
    if policy_name == "planner":
        planner = workflow.controllers.get("planner")
        if planner is None:
            raise RuntimeError("policy='planner' requested but no planner controller is configured.")
        latent_step = workflow.rssm.observe(
            obs_tensor,
            prev_action=prev_action,
            reset_mask=torch.as_tensor(done, dtype=torch.bool, device=workflow.device),
            detach_posteriors=True,
        )
        return planner.act(latent_step.posterior.to_tensor(), workflow=workflow, deterministic=True)
    raise ValueError("policy must be one of: random, actor, planner")


def compute_reward_calibration(workflow: Any, trajectory: Mapping[str, np.ndarray]) -> Tuple[List[CsvRow], MetricDict]:
    """Teacher-force the RSSM and compare reward-head predictions to real rewards."""
    observations, actions, rewards, dones = _trajectory_to_tensors(trajectory, workflow.device)
    with torch.no_grad():
        sequence = workflow.rssm.observe_sequence(observations, actions=actions, dones=dones)
        predicted = workflow.reward_predictor(sequence.posterior.to_tensor()).squeeze(-1)

    pred_np = predicted.detach().cpu().numpy()
    actual_np = rewards.detach().cpu().numpy()
    metrics = compute_regression_metrics(pred_np, actual_np, prefix="reward")
    rows = _reward_rows(pred_np, actual_np)
    return rows, metrics


def compute_open_loop_diagnostics(
    workflow: Any,
    trajectory: Mapping[str, np.ndarray],
    *,
    open_loop_horizon: int,
    horizons: Sequence[int] | None = None,
) -> Tuple[List[CsvRow], MetricDict]:
    """Roll the RSSM forward using real future actions and compare rewards."""
    observations, actions, rewards, dones = _trajectory_to_tensors(trajectory, workflow.device)
    max_horizon = int(open_loop_horizon)
    if max_horizon <= 0:
        raise ValueError("open_loop_horizon must be positive.")
    if observations.shape[1] <= max_horizon:
        raise ValueError(
            f"Need more collected steps than open_loop_horizon; got steps={observations.shape[1]}, "
            f"horizon={max_horizon}."
        )

    with torch.no_grad():
        sequence = workflow.rssm.observe_sequence(observations, actions=actions, dones=dones)
        posterior = sequence.posterior

        predicted_windows: List[np.ndarray] = []
        actual_windows: List[np.ndarray] = []
        for start in range(0, int(observations.shape[1]) - max_horizon):
            valid_mask = ~torch.any(dones[:, start : start + max_horizon], dim=1)
            if not bool(torch.any(valid_mask)):
                continue
            state = _slice_rssm_state(posterior, start, valid_mask)
            action_sequence = actions[valid_mask, start : start + max_horizon]
            rollout = workflow.imagine(
                latent=state,
                horizon=max_horizon,
                action_sequence=action_sequence,
                deterministic=True,
            )
            predicted = rollout["rewards"].squeeze(-1).detach().cpu().numpy()
            actual = rewards[valid_mask, start : start + max_horizon].detach().cpu().numpy()
            predicted_windows.append(predicted)
            actual_windows.append(actual)

    if not predicted_windows:
        raise ValueError("No valid open-loop windows found. Collect more steps or reduce horizon.")
    predicted_np = np.concatenate(predicted_windows, axis=0)
    actual_np = np.concatenate(actual_windows, axis=0)
    selected_horizons = list(horizons or _default_horizons(max_horizon))
    return compute_horizon_return_metrics(predicted_np, actual_np, horizons=selected_horizons)


def write_tensorboard_outputs(
    *,
    logdir: str | Path,
    summary: Mapping[str, Any],
    horizon_rows: Sequence[Mapping[str, Any]],
    reward_plot: str | Path,
    horizon_plot: str | Path,
    checkpoint_step: int = 0,
) -> None:
    """Write PlaNet diagnostic metrics and plots to a TensorBoard event file."""
    _prepare_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg

    tb_dir = Path(logdir)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    try:
        for key in ("reward/mse", "reward/mae", "reward/bias", "reward/corr"):
            numeric_value = _as_finite_float(summary.get(key))
            if numeric_value is None:
                continue
            writer.add_scalar(f"diagnostics/{key}", numeric_value, int(checkpoint_step))

        for row in horizon_rows:
            horizon = _as_positive_int(row.get("horizon"))
            if horizon is None:
                continue
            for metric_name in (
                "return_mae",
                "return_mse",
                "return_bias",
                "return_corr",
                "predicted_return_mean",
                "actual_return_mean",
            ):
                numeric_value = _as_finite_float(row.get(metric_name))
                if numeric_value is None:
                    continue
                writer.add_scalar(f"diagnostics/open_loop/{metric_name}", numeric_value, horizon)

        reward_image = mpimg.imread(str(reward_plot))
        horizon_image = mpimg.imread(str(horizon_plot))
        writer.add_image(
            "diagnostics/images/reward_pred_vs_actual",
            reward_image,
            int(checkpoint_step),
            dataformats="HWC",
        )
        writer.add_image(
            "diagnostics/images/open_loop_errors",
            horizon_image,
            int(checkpoint_step),
            dataformats="HWC",
        )
    finally:
        writer.flush()
        writer.close()


def plot_reward_calibration(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    _prepare_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = [float(row["predicted_reward"]) for row in rows]
    actual = [float(row["actual_reward"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, s=10, alpha=0.45)
    if actual and predicted:
        lo = min(min(actual), min(predicted))
        hi = max(max(actual), max(predicted))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Reward Prediction Calibration")
    ax.set_xlabel("actual reward")
    ax.set_ylabel("predicted reward")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_open_loop_errors(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    _prepare_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = [int(row["horizon"]) for row in rows]
    mae = [float(row["return_mae"]) for row in rows]
    mse = [float(row["return_mse"]) for row in rows]

    fig, ax_mae = plt.subplots(figsize=(8, 5))
    ax_mae.plot(horizons, mae, marker="o", label="return MAE")
    ax_mae.set_xlabel("rollout horizon")
    ax_mae.set_ylabel("return MAE")
    ax_mae.grid(True, alpha=0.25)
    ax_mse = ax_mae.twinx()
    ax_mse.plot(horizons, mse, marker="s", color="tab:red", label="return MSE")
    ax_mse.set_ylabel("return MSE")
    lines, labels = ax_mae.get_legend_handles_labels()
    extra_lines, extra_labels = ax_mse.get_legend_handles_labels()
    ax_mae.legend(lines + extra_lines, labels + extra_labels, loc="best")
    ax_mae.set_title("Open-Loop Return Error")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _instantiate_buffer(buffer_cfg: Any, *, device: str, num_envs: int) -> Any:
    try:
        return instantiate(buffer_cfg, device=device, num_envs=num_envs)
    except TypeError:
        try:
            return instantiate(buffer_cfg, device=device)
        except TypeError:
            return instantiate(buffer_cfg)


def _trajectory_to_tensors(
    trajectory: Mapping[str, np.ndarray],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.as_tensor(np.swapaxes(trajectory["observations"], 0, 1), dtype=torch.float32, device=device)
    actions = torch.as_tensor(np.swapaxes(trajectory["actions"], 0, 1), dtype=torch.float32, device=device)
    rewards = torch.as_tensor(np.swapaxes(trajectory["rewards"], 0, 1), dtype=torch.float32, device=device)
    dones = torch.as_tensor(np.swapaxes(trajectory["dones"], 0, 1), dtype=torch.bool, device=device)
    return observations, actions, rewards, dones


def _slice_rssm_state(state: RSSMState, time_index: int, batch_mask: torch.Tensor) -> RSSMState:
    return RSSMState(
        deterministic=state.deterministic[batch_mask, time_index],
        stochastic=state.stochastic[batch_mask, time_index],
        mean=state.mean[batch_mask, time_index],
        std=state.std[batch_mask, time_index],
    )


def _reward_rows(predicted: np.ndarray, actual: np.ndarray) -> List[CsvRow]:
    rows: List[CsvRow] = []
    pred = np.asarray(predicted)
    target = np.asarray(actual)
    for env_id in range(pred.shape[0]):
        for step in range(pred.shape[1]):
            rows.append(
                {
                    "step": step,
                    "env": env_id,
                    "predicted_reward": float(pred[env_id, step]),
                    "actual_reward": float(target[env_id, step]),
                    "error": float(pred[env_id, step] - target[env_id, step]),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _default_horizons(max_horizon: int) -> List[int]:
    candidates = [1, 3, 5, 10, 12, max_horizon]
    return sorted({h for h in candidates if 0 < h <= max_horizon})


def _ensure_batch(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 1:
        return obs[None, :]
    return obs


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    if integer <= 0:
        return None
    return integer


def _tensorboard_checkpoint_step(summary: Mapping[str, Any], *, fallback: int = 0) -> int:
    step = _as_positive_int(summary.get("checkpoint_global_step"))
    if step is not None:
        return step
    return max(int(fallback), 0)


def _prepare_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "rl-lab-matplotlib-cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def parse_horizons(value: str | None) -> List[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a PlaNet checkpoint .pt file.")
    parser.add_argument("--experiment", required=True, help="Hydra experiment name, e.g. planet_dmc_cartpole_swingup.")
    parser.add_argument("--budget", default=None, help="Optional budget config name.")
    parser.add_argument("--out", default=None, help="Output directory. Defaults to <run>/diagnostics.")
    parser.add_argument("--steps", type=int, default=256, help="Real environment steps to collect for diagnostics.")
    parser.add_argument("--open-loop-horizon", type=int, default=12, help="Maximum open-loop rollout horizon.")
    parser.add_argument("--horizons", default=None, help="Comma-separated horizons to report, e.g. 1,3,5,12.")
    parser.add_argument("--policy", choices=["random", "actor", "planner"], default="random")
    parser.add_argument("--device", default=None, help="Override experiment.device.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override environment.num_envs if supported.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment/environment seed.")
    parser.add_argument(
        "--tensorboard-logdir",
        default=None,
        help="TensorBoard logdir. Defaults to <checkpoint run>/runs/diagnostics_planet_reward_open_loop.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Do not write diagnostic scalars/images to TensorBoard.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be repeated.",
    )
    args = parser.parse_args(argv)

    if not OmegaConf.has_resolver("add"):
        OmegaConf.register_new_resolver("add", lambda x, y: x + y)

    summary = diagnose_planet_checkpoint(
        checkpoint=args.checkpoint,
        experiment=args.experiment,
        budget=args.budget,
        out_dir=args.out,
        steps=args.steps,
        open_loop_horizon=args.open_loop_horizon,
        horizons=parse_horizons(args.horizons),
        policy=args.policy,
        device_override=args.device,
        num_envs=args.num_envs,
        seed=args.seed,
        overrides=args.override,
        tensorboard=not args.no_tensorboard,
        tensorboard_logdir=args.tensorboard_logdir,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
