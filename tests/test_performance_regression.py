import time

import torch

from src.networks.mlp import MLP
from tests._helpers import issue_logger


def test_simple_performance_baseline_logging():
    net = MLP({"input_dim": 128, "output_dim": 64, "hidden_dims": [128, 128], "device": "cpu"})
    x = torch.randn(512, 128)
    t0 = time.time()
    with torch.no_grad():
        _ = net(x)
    dt = time.time() - t0
    issue_logger.log(
        category="regression",
        severity="info",
        message="Forward pass baseline",
        extra={"elapsed_s": dt},
    )
    assert True

