import torch

from src.networks.mlp import MLP
from tests._helpers import issue_logger, has_torch_cuda


def test_network_device_move_cpu():
    net = MLP({"input_dim": 8, "output_dim": 2, "device": "cpu"})
    x = torch.randn(2, 8)
    y = net(x)
    assert y.shape == (2, 2)


def test_network_device_move_cuda_if_available():
    if not has_torch_cuda():
        return
    net = MLP({"input_dim": 8, "output_dim": 2, "device": "cuda"})
    x = torch.randn(2, 8, device=torch.device("cuda"))
    y = net(x)
    assert y.is_cuda

