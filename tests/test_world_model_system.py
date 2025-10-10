import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paradigms.factory import ComponentFactory
from src.utils.registry import auto_import_modules


def _dreamer_cartpole_config():
    return {
        "paradigm": "world_model",
        "encoder": {
            "type": "mlp",
            "config": {
                "input_dim": 4,
                "hidden_dims": [64, 64],
                "activation": "elu",
            },
        },
        "representation_learner": {
            "type": "rssm",
            "config": {
                "latent_dim": 16,
                "hidden_dim": 64,
                "min_std": 0.1,
            },
        },
        "dynamics_model": {
            "type": "rssm",
            "config": {
                "hidden_dim": 64,
                "min_std": 0.1,
            },
        },
        "reward_predictor": {
            "type": "mlp",
            "config": {
                "hidden_dims": [64, 64],
                "activation": "elu",
            },
        },
        "observation_decoder": {
            "type": "mlp",
            "config": {
                "hidden_dims": [64, 64],
                "activation": "elu",
                "output_dim": 4,
            },
        },
        "policy_head": {
            "type": "categorical_mlp",
            "config": {
                "hidden_dims": [64, 64],
                "activation": "elu",
                "discrete_actions": True,
                "action_dim": 2,
            },
        },
        "value_function": {
            "type": "critic_mlp",
            "config": {
                "hidden_dims": [64, 64],
                "activation": "elu",
            },
        },
        "paradigm_config": {
            "imagination_length": 3,
            "gamma": 0.99,
            "lambda_return": 0.95,
            "entropy_coef": 0.0,
        },
    }


def test_world_model_loss_smoke():
    auto_import_modules()
    factory = ComponentFactory()
    paradigm = factory.create_paradigm(_dreamer_cartpole_config())

    batch_size = 8
    observations = torch.randn(batch_size, 4)
    next_observations = torch.randn(batch_size, 4)
    actions = torch.randint(0, 2, (batch_size,))
    rewards = torch.randn(batch_size)
    returns = torch.randn(batch_size)

    batch = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "rewards": rewards,
        "returns": returns,
    }

    losses = paradigm.compute_loss(batch)

    assert "total_loss" in losses
    for value in losses.values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
