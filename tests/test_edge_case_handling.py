import pytest

from src.networks.base import BaseNetwork


class _TmpNet(BaseNetwork):
    def _build_network(self):
        import torch.nn as nn
        self.l = nn.Identity()

    def forward(self, x):
        return x


def test_invalid_activation_name_raises():
    # Construct normally, then explicitly request an invalid activation
    net = _TmpNet({"input_dim": 1, "output_dim": 1})
    import pytest
    with pytest.raises(ValueError):
        _ = net.get_activation_function("not_an_activation")
