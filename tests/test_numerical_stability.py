import torch
import math

from src.networks.mlp import ContinuousActorMLP


def test_continuous_actor_output_finite():
    net = ContinuousActorMLP({"input_dim": 8, "output_dim": 3, "device": "cpu", "use_tanh_squashing": True})
    x = torch.randn(64, 8) * 10  # large inputs
    out = net(x)
    assert torch.isfinite(out).all()

