import torch

from src.networks.mlp import MLP


def test_simple_gradient_flow():
    net = MLP({"input_dim": 10, "output_dim": 1, "hidden_dims": [16], "device": "cpu"})
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    x = torch.randn(32, 10)
    y_true = torch.randn(32, 1)
    y = net(x)
    loss = torch.nn.functional.mse_loss(y, y_true)
    opt.zero_grad()
    loss.backward()
    # Check that at least one parameter received non-zero grad
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in net.parameters())
    opt.step()

