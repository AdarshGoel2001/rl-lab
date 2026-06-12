#!/usr/bin/env python3
"""Generate pixel Dreamer reconstruction diagnostics from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from scripts.train import build_world_model_components


def summarize_reconstructions(actual: torch.Tensor, reconstruction: torch.Tensor) -> dict[str, Any]:
    """Return scalar reconstruction diagnostics for normalized image tensors."""
    actual = _normalize_images(actual)
    reconstruction = reconstruction.to(torch.float32).clamp(0.0, 1.0)
    diff = reconstruction - actual
    return {
        "schema_version": 1,
        "num_frames": int(actual.shape[0] * actual.shape[1]) if actual.dim() == 5 else int(actual.shape[0]),
        "reconstruction_mse": float(diff.pow(2).mean().item()),
        "reconstruction_mae": float(diff.abs().mean().item()),
        "actual_min": float(actual.min().item()),
        "actual_max": float(actual.max().item()),
        "reconstruction_min": float(reconstruction.min().item()),
        "reconstruction_max": float(reconstruction.max().item()),
    }


def build_reconstruction_grid(
    actual: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    max_frames: int = 8,
) -> np.ndarray:
    """Create an HWC image grid with actual frames above reconstructions."""
    actual_frames = _flatten_frames(_normalize_images(actual))[:max_frames]
    recon_frames = _flatten_frames(reconstruction.to(torch.float32).clamp(0.0, 1.0))[:max_frames]
    if actual_frames.shape != recon_frames.shape:
        raise ValueError(f"Actual/reconstruction frame shapes differ: {actual_frames.shape} vs {recon_frames.shape}.")
    top = torch.cat([frame for frame in actual_frames], dim=2)
    bottom = torch.cat([frame for frame in recon_frames], dim=2)
    grid = torch.cat([top, bottom], dim=1)
    return grid.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def diagnose_pixel_checkpoint(
    *,
    checkpoint: str | Path,
    experiment: str,
    budget: str,
    out_dir: str | Path | None = None,
    steps: int = 32,
    device_override: str | None = None,
    seed: int | None = 0,
    overrides: Sequence[str] = (),
    tensorboard: bool = True,
    tensorboard_logdir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a pixel Dreamer checkpoint, collect frames, and save recon diagnostics."""
    checkpoint_path = Path(checkpoint)
    run_dir = checkpoint_path.parents[1] if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    output_dir = Path(out_dir) if out_dir is not None else run_dir / "diagnostics" / "dreamer_pixel_reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _compose_config(experiment=experiment, budget=budget, overrides=overrides)
    if device_override is not None:
        cfg.experiment.device = device_override
    device = str(cfg.experiment.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        cfg.experiment.seed = int(seed)

    components = build_world_model_components(cfg, device)
    _load_checkpoint_components(checkpoint_path, components.components)
    for module in components.components.values():
        if hasattr(module, "eval"):
            module.eval()

    observations, actions, dones = _collect_random_pixel_sequence(cfg, steps=steps, device=device)
    encoder = getattr(components, "observation_encoder", None)
    rssm = components.representation_learner
    decoder = components.observation_predictor
    if encoder is None:
        raise RuntimeError("Pixel diagnostics require components.observation_encoder.")

    with torch.no_grad():
        features = encoder(observations)
        sequence = rssm.observe_sequence(features, actions=actions, dones=dones)
        reconstruction = decoder(sequence.posterior.to_tensor())

    summary = summarize_reconstructions(observations, reconstruction)
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "experiment": experiment,
            "budget": budget,
            "steps": int(steps),
        }
    )
    grid = build_reconstruction_grid(observations, reconstruction)
    summary_path = output_dir / "dreamer_pixel_reconstruction_summary.json"
    grid_path = output_dir / "dreamer_pixel_reconstruction_grid.png"
    report_path = output_dir / "dreamer_pixel_reconstruction_report.md"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    _write_png(grid_path, grid)
    report_path.write_text(_render_report(summary, grid_path), encoding="utf-8")

    if tensorboard:
        tb_dir = Path(tensorboard_logdir) if tensorboard_logdir else output_dir / "tensorboard"
        write_tensorboard_outputs(tb_dir, summary, grid)

    return {
        **summary,
        "summary_path": str(summary_path),
        "grid_path": str(grid_path),
        "report_path": str(report_path),
    }


def write_tensorboard_outputs(logdir: str | Path, summary: Mapping[str, Any], grid: np.ndarray) -> None:
    """Write reconstruction scalars and image grid to TensorBoard."""
    writer = SummaryWriter(log_dir=str(logdir))
    try:
        step = int(summary.get("steps", 0))
        writer.add_scalar("diagnostics/pixel/reconstruction_mse", float(summary["reconstruction_mse"]), step)
        writer.add_scalar("diagnostics/pixel/reconstruction_mae", float(summary["reconstruction_mae"]), step)
        writer.add_image("diagnostics/images/pixel_reconstruction_grid", grid, step, dataformats="HWC")
    finally:
        writer.flush()
        writer.close()


def _collect_random_pixel_sequence(cfg: DictConfig, *, steps: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    env = instantiate(cfg.environment)
    try:
        obs = env.reset(seed=int(cfg.experiment.get("seed", 0)))
        obs_arr = np.asarray(obs)
        action_dim = int(cfg._dims.action)
        observations = []
        actions = []
        dones = []
        for _ in range(int(steps)):
            action = np.random.uniform(-1.0, 1.0, size=(1, action_dim)).astype(np.float32)
            observations.append(obs_arr.copy())
            actions.append(action.copy())
            next_obs, _reward, done, _info = env.step(action)
            done_arr = np.asarray(done, dtype=bool).reshape(1)
            dones.append(done_arr.copy())
            obs_arr = np.asarray(next_obs)
            if bool(done_arr[0]):
                obs_arr = np.asarray(env.reset())
        obs_tensor = torch.as_tensor(np.stack(observations, axis=1), device=device)
        action_tensor = torch.as_tensor(np.stack(actions, axis=1), dtype=torch.float32, device=device)
        done_tensor = torch.as_tensor(np.stack(dones, axis=1), dtype=torch.bool, device=device)
        return obs_tensor, action_tensor, done_tensor
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _compose_config(*, experiment: str, budget: str, overrides: Sequence[str]) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[3] / "configs"
    all_overrides = [f"+experiment={experiment}", f"budget={budget}", *overrides]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=list(all_overrides))


def _load_checkpoint_components(checkpoint_path: Path, components: Mapping[str, Any]) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    component_states = checkpoint.get("components", {})
    missing = []
    for name, module in components.items():
        state = component_states.get(name)
        if state is None:
            missing.append(name)
            continue
        module.load_state_dict(state)
    if missing:
        raise RuntimeError(f"Checkpoint is missing component states required for diagnostics: {missing}")


def _normalize_images(images: torch.Tensor) -> torch.Tensor:
    images = images.to(torch.float32)
    if images.numel() and float(images.detach().max().item()) > 1.5:
        images = images / 255.0
    return images.clamp(0.0, 1.0)


def _flatten_frames(images: torch.Tensor) -> torch.Tensor:
    if images.dim() == 5:
        batch, time, channels, height, width = images.shape
        return images.reshape(batch * time, channels, height, width)
    if images.dim() == 4:
        return images
    raise ValueError(f"Expected image tensor [B,T,C,H,W] or [B,C,H,W], got {tuple(images.shape)}.")


def _write_png(path: Path, grid: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.clip(grid, 0.0, 1.0))


def _render_report(summary: Mapping[str, Any], grid_path: Path) -> str:
    return "\n".join(
        [
            "# Dreamer Pixel Reconstruction Diagnostics",
            "",
            f"- checkpoint: {summary.get('checkpoint')}",
            f"- reconstruction MSE: {summary.get('reconstruction_mse')}",
            f"- reconstruction MAE: {summary.get('reconstruction_mae')}",
            f"- frames: {summary.get('num_frames')}",
            f"- grid: {grid_path.name}",
            "",
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a pixel Dreamer checkpoint .pt file.")
    parser.add_argument("--experiment", default="dreamer_dmc_cartpole_swingup")
    parser.add_argument("--budget", default="dreamer_pixel_100ep")
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--tensorboard-logdir", default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args(argv)

    if not OmegaConf.has_resolver("add"):
        OmegaConf.register_new_resolver("add", lambda x, y: x + y)

    result = diagnose_pixel_checkpoint(
        checkpoint=args.checkpoint,
        experiment=args.experiment,
        budget=args.budget,
        out_dir=args.out,
        steps=args.steps,
        device_override=args.device,
        seed=args.seed,
        overrides=args.override,
        tensorboard=not args.no_tensorboard,
        tensorboard_logdir=args.tensorboard_logdir,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
