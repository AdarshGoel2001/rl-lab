import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.set_num_threads(1)

from src.buffers.world_model_sequence import WorldModelSequenceBuffer


def _make_step(num_envs: int = 2):
    obs = np.random.randn(1, num_envs, 4).astype(np.float32)
    next_obs = np.random.randn(1, num_envs, 4).astype(np.float32)
    actions = np.random.randint(0, 3, size=(1, num_envs))
    rewards = np.random.randn(1, num_envs).astype(np.float32)
    dones = np.zeros((1, num_envs), dtype=bool)
    return {
        "observations": obs,
        "next_observations": next_obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def test_world_model_sequence_buffer_sampling():
    config = {
        "capacity": 64,
        "batch_size": 4,
        "sequence_length": 5,
        "sequence_stride": 2,
        "num_envs": 2,
    }
    buffer = WorldModelSequenceBuffer(config)

    for t in range(20):
        traj = _make_step(num_envs=2)
        if t % 7 == 6:
            traj["dones"][0, 0] = True
        buffer.add(trajectory=traj)

    assert buffer.ready()

    batch = buffer.sample(batch_size=4)
    obs = batch["observations"]
    next_obs = batch["next_observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    mask = batch["sequence_mask"]

    assert obs.shape[0] == 4
    assert obs.shape[1] == config["sequence_length"]
    assert next_obs.shape == obs.shape
    assert actions.shape[:2] == obs.shape[:2]
    assert rewards.shape[:2] == obs.shape[:2]
    assert mask.shape[:2] == obs.shape[:2]

    # Ensure mask zeros out timesteps after terminal transitions
    mask_np = mask.cpu().numpy()
    assert np.all((mask_np[:, 1:] <= mask_np[:, :-1]) | (mask_np[:, :-1] == 1.0))

    # Ensure tensors live on configured device
    assert obs.device == buffer.device
    assert mask.device == buffer.device
