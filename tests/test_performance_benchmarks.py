import time

import torch

from src.networks.mlp import MLP
from tests._helpers import issue_logger


def test_mlp_forward_speed():
    net = MLP({"input_dim": 64, "output_dim": 32, "hidden_dims": [64, 64], "device": "cpu"})
    x = torch.randn(1024, 64)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(200):
            _ = net(x)
    dt = time.time() - t0
    issue_logger.log(
        category="benchmark",
        severity="info",
        message="MLP forward benchmark",
        extra={"iterations": 200, "batch": 1024, "elapsed_s": dt},
    )
    assert True

