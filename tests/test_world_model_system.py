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

from omegaconf import OmegaConf

from scripts.train import build_world_model_components, build_optimizers


def _dreamer_cartpole_config():
    return OmegaConf.create(
        {
            "_dims": {
                "observation": 4,
                "action": 2,
                "encoder_output": 64,
                "deterministic": 32,
                "stochastic": 16,
                "representation": 48,
            },
            "components": {
                "encoder": {
                    "_target_": "src.components.encoders.simple_mlp.MLPEncoder",
                    "input_dim": 4,
                    "hidden_dims": [64, 64],
                    "activation": "elu",
                },
                "representation_learner": {
                    "_target_": "src.components.world_models.representation_learners.rssm.RSSMRepresentationLearner",
                    "feature_dim": 64,
                    "latent_dim": 16,
                    "deterministic_dim": 32,
                    "stochastic_dim": 16,
                    "hidden_dim": 64,
                    "min_std": 0.1,
                    "action_dim": 2,
                },
                "dynamics_model": {
                    "_target_": "src.components.world_models.representation_learners.rssm.RSSMRepresentationLearner",
                    "feature_dim": 64,
                    "latent_dim": 16,
                    "deterministic_dim": 32,
                    "stochastic_dim": 16,
                    "hidden_dim": 64,
                    "min_std": 0.1,
                    "action_dim": 2,
                },
            },
            "algorithm": {
                "world_model_lr": 2e-4,
            },
            "controllers": {
                "actor": {
                    "discrete_actions": True,
                }
            },
        }
    )


def test_world_model_components_smoke():
    cfg = _dreamer_cartpole_config()
    components = build_world_model_components(cfg, device="cpu")
    optimizers = build_optimizers(cfg, components, controllers={})
    component_dict = components.as_dict()["components"]

    assert component_dict["encoder"] is not None
    assert component_dict["representation_learner"] is not None
    assert component_dict["dynamics_model"] is not None
    assert "world_model" in optimizers

    observations = torch.randn(4, 4)
    encoder = component_dict["encoder"]
    with torch.no_grad():
        features = encoder(observations)
    assert isinstance(features, torch.Tensor)
    assert torch.isfinite(features).all()
