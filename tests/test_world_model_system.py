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

from src.orchestration.factory import ComponentFactory
from src.utils.registry import auto_import_modules


def _dreamer_cartpole_config():
    return {
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
                "feature_dim": 64,
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
                "representation_dim": 64,
                "hidden_dims": [64, 64],
                "activation": "elu",
                "discrete_actions": True,
                "action_dim": 2,
            },
        },
        "value_function": {
            "type": "critic_mlp",
            "config": {
                "representation_dim": 64,
                "hidden_dims": [64, 64],
                "activation": "elu",
            },
        },
    }


def test_world_model_components_smoke():
    auto_import_modules()
    bundle = ComponentFactory.create_world_model_components(_dreamer_cartpole_config(), device="cpu")
    components = bundle.as_dict()

    assert components["encoder"] is not None
    assert components["representation_learner"] is not None
    assert components["dynamics_model"] is not None

    observations = torch.randn(4, 4)
    encoder = components["encoder"]
    with torch.no_grad():
        features = encoder(observations)
    assert isinstance(features, torch.Tensor)
    assert torch.isfinite(features).all()
