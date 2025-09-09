from typing import Dict, Any

import torch
import torch.nn as nn

from src.utils.registry import register_network, get_network


@register_network("test_linear")
class LinearForRegistry(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        in_d = int(config.get("input_dim", 4))
        out_d = int(config.get("output_dim", 2))
        self.net = nn.Linear(in_d, out_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def test_component_swapping_and_registration():
    Net = get_network("test_linear")
    m = Net({"input_dim": 4, "output_dim": 3, "device": "cpu"})
    x = torch.randn(5, 4)
    y = m(x)
    assert y.shape == (5, 3)
