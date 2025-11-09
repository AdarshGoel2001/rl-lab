#!/usr/bin/env python3
"""
Memory Monitoring Harness for RL Lab

Scenarios:
- buffer-eviction-ram: Stress EpisodeReplayBuffer eviction and sampling; track RAM vs accounted bytes
- gpu-update: Simulate GPU forward/backward on synthetic batches; track VRAM baseline without eviction
- gpu-eviction: Combine buffer sampling with a small CUDA model to emulate world-model updates under eviction
- weakref: Ensure batch tensors die after use (no lingering references)

Outputs: CSV logs under experiments/memory_monitor_<timestamp> and optional CUDA memory snapshots
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # optional

from src.buffers.episode_replay import EpisodeReplayBuffer


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def process_rss_mb() -> float:
    if psutil is None:
        return -1.0
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


def gpu_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"gpu_alloc_mb": -1.0, "gpu_reserved_mb": -1.0, "gpu_max_alloc_mb": -1.0, "gpu_max_reserved_mb": -1.0}
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    return {
        "gpu_alloc_mb": float(alloc),
        "gpu_reserved_mb": float(reserved),
        "gpu_max_alloc_mb": float(max_alloc),
        "gpu_max_reserved_mb": float(max_reserved),
    }


def make_atari_traj(T: int, E: int, A: int = 6) -> Dict[str, np.ndarray]:
    obs = (np.random.randint(0, 256, size=(T, E, 84, 84, 4), dtype=np.uint8)).astype(np.uint8)
    actions = np.random.randn(T, E, A).astype(np.float32)
    rewards = np.random.randn(T, E).astype(np.float32)
    # Mark a done at the last timestep for each env to finalize episodes
    dones = np.zeros((T, E), dtype=bool)
    dones[-1, :] = True
    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def scenario_buffer_eviction_ram(args: argparse.Namespace, log_path: Path) -> None:
    buf = EpisodeReplayBuffer(
        config={
            "capacity": args.capacity,
            "batch_size": args.batch_size,
            "sequence_length": args.seq_len,
            "burn_in_length": 0,
            "recent_ratio": 0.8,
            "num_envs": args.num_envs,
        }
    )
    writer = csv.DictWriter(open(log_path, "w", newline=""), fieldnames=[
        "step", "rss_mb", "obs_bytes", "aux_bytes", "num_episodes", "total_steps"
    ])
    writer.writeheader()
    step = 0
    while step < args.steps:
        traj = make_atari_traj(args.traj_T, args.num_envs, A=args.action_dim)
        buf.add(trajectory=traj)
        if buf.ready():
            _ = buf.sample(args.batch_size)
        st = buf.stats()
        if step % args.log_interval == 0:
            writer.writerow({
                "step": step,
                "rss_mb": process_rss_mb(),
                "obs_bytes": st.get("obs_bytes", -1),
                "aux_bytes": st.get("aux_bytes", -1),
                "num_episodes": st.get("num_episodes", -1),
                "total_steps": st.get("total_steps", -1),
            })
        step += args.traj_T * args.num_envs


def small_cuda_model(latent_dim: int = 256) -> torch.nn.Module:
    # Simple MLP decoder-like model to simulate memory pressure
    return torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 1024),
        torch.nn.ELU(),
        torch.nn.Linear(1024, 84 * 84 * 4),
    ).cuda()


def scenario_gpu_update(args: argparse.Namespace, log_path: Path) -> None:
    assert torch.cuda.is_available(), "CUDA required for gpu-update scenario"
    model = small_cuda_model(args.latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, L = args.batch_size, args.seq_len
    latent = torch.randn(B * L, args.latent_dim, device="cuda")
    target = torch.randn(B * L, 84 * 84 * 4, device="cuda")
    writer = csv.DictWriter(open(log_path, "w", newline=""), fieldnames=[
        "iter", "gpu_alloc_mb", "gpu_reserved_mb", "gpu_max_alloc_mb", "gpu_max_reserved_mb"
    ])
    writer.writeheader()
    for i in range(args.iters):
        out = model(latent)
        loss = torch.nn.functional.mse_loss(out, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i + 1) % args.log_interval == 0:
            stats = gpu_stats()
            stats_row = {"iter": i + 1, **stats}
            writer.writerow(stats_row)


def scenario_gpu_eviction(args: argparse.Namespace, log_dir: Path) -> None:
    assert torch.cuda.is_available(), "CUDA required for gpu-eviction scenario"
    # CPU buffer + small CUDA model to emulate update under eviction churn
    buf = EpisodeReplayBuffer(
        config={
            "capacity": args.capacity,
            "batch_size": args.batch_size,
            "sequence_length": args.seq_len,
            "burn_in_length": 0,
            "recent_ratio": 0.8,
            "num_envs": args.num_envs,
        }
    )
    model = small_cuda_model(args.latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, L = args.batch_size, args.seq_len
    # Log file
    writer = csv.DictWriter(open(log_dir / "gpu_eviction.csv", "w", newline=""), fieldnames=[
        "global_step", "gpu_alloc_mb", "gpu_reserved_mb", "gpu_max_alloc_mb", "gpu_max_reserved_mb", "rss_mb", "num_episodes", "total_steps"
    ])
    writer.writeheader()
    global_steps = 0
    while global_steps < args.steps:
        traj = make_atari_traj(args.traj_T, args.num_envs, A=args.action_dim)
        buf.add(trajectory=traj)
        if buf.ready():
            batch = buf.sample(args.batch_size)
            obs = batch["observations"].to("cuda", non_blocking=False)
            x = obs.reshape(B * L, -1)
            target = torch.zeros_like(x)
            out = model(torch.randn(B * L, args.latent_dim, device="cuda"))
            loss = torch.nn.functional.mse_loss(out, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_steps += B * L
        st = buf.stats()
        if global_steps % max(args.log_interval, 1) == 0:
            row = {"global_step": global_steps, **gpu_stats(), "rss_mb": process_rss_mb(),
                   "num_episodes": st.get("num_episodes", -1), "total_steps": st.get("total_steps", -1)}
            writer.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=[
        "buffer-eviction-ram", "gpu-update", "gpu-eviction"
    ])
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--capacity", type=int, default=50_000)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    ap.add_argument("--seq-len", dest="seq_len", type=int, default=50)
    ap.add_argument("--num-envs", dest="num_envs", type=int, default=2)
    ap.add_argument("--traj-T", dest="traj_T", type=int, default=100)
    ap.add_argument("--action-dim", dest="action_dim", type=int, default=6)
    ap.add_argument("--latent-dim", dest="latent_dim", type=int, default=256)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--log-interval", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("experiments") / f"memory_monitor_{now_ts()}" )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "buffer-eviction-ram":
        log_path = out_dir / "buffer_eviction_ram.csv"
        scenario_buffer_eviction_ram(args, log_path)
        print(f"buffer-eviction-ram logs: {log_path}")
    elif args.mode == "gpu-update":
        log_path = out_dir / "gpu_update.csv"
        scenario_gpu_update(args, log_path)
        print(f"gpu-update logs: {log_path}")
    elif args.mode == "gpu-eviction":
        scenario_gpu_eviction(args, out_dir)
        print(f"gpu-eviction logs: {out_dir/'gpu_eviction.csv'}")


if __name__ == "__main__":
    main()

