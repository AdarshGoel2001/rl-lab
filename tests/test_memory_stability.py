import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.buffers.episode_replay import EpisodeReplayBuffer


def _rss_mb():
    try:
        import psutil  # type: ignore
    except Exception:
        return -1.0
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)


def _make_traj(T, E, A=6):
    obs = (np.random.randint(0, 256, size=(T, E, 84, 84, 4), dtype=np.uint8)).astype(np.uint8)
    actions = np.random.randn(T, E, A).astype(np.float32)
    rewards = np.random.randn(T, E).astype(np.float32)
    dones = np.zeros((T, E), dtype=bool)
    dones[-1, :] = True
    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def test_buffer_eviction_ram_stability():
    buf = EpisodeReplayBuffer({
        "capacity": 5000,  # small for test
        "batch_size": 8,
        "sequence_length": 16,
        "num_envs": 2,
        "burn_in_length": 0,
        "recent_ratio": 0.8,
    })

    baseline = _rss_mb()
    steps = 0
    last_rss = baseline
    for _ in range(200):  # ~200 episodes
        traj = _make_traj(50, 2)
        buf.add(trajectory=traj)
        if buf.ready():
            _ = buf.sample(8)
        steps += 100
        if steps % 500 == 0:
            st = buf.stats()
            rss = _rss_mb()
            # RSS should not grow unbounded beyond accounted bytes + reasonable overhead
            accounted_mb = (st.get("obs_bytes", 0) + st.get("aux_bytes", 0)) / (1024 ** 2)
            assert rss <= accounted_mb + 500, f"RSS {rss}MB > accounted {accounted_mb}MB + overhead"
            last_rss = rss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_update_baseline_stable():
    # Synthetic small model to check no baseline creep between steps
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 1024), torch.nn.ELU(), torch.nn.Linear(1024, 84 * 84 * 4)
    ).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, L = 8, 16
    latent = torch.randn(B * L, 256, device="cuda")
    target = torch.randn(B * L, 84 * 84 * 4, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    for i in range(200):
        out = model(latent)
        loss = torch.nn.functional.mse_loss(out, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i + 1) % 50 == 0:
            torch.cuda.synchronize()
            cur_alloc = torch.cuda.memory_allocated()
            # Allow small drift, but not monotonic creep
            assert cur_alloc <= base_alloc * 1.2 + 50 * 1024 ** 2

