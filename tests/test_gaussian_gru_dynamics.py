import numpy as np
import torch
import torch.nn.functional as F

from src.components.dynamics.gaussian_gru import GaussianGRUDynamics


def test_gaussian_gru_observe_single_step_matches_og_contract():
    model = GaussianGRUDynamics(
        latent_dim=8,
        action_dim=3,
        hidden_dim=16,
        num_gaussians=1,
        device="cpu",
    )
    model.reset_state(batch_size=2)

    outputs = model.observe(
        torch.randn(2, 8),
        torch.randn(2, 3),
        dones=np.zeros(2, dtype=bool),
        return_mixture=True,
        deterministic=True,
    )

    assert outputs["next_latent"].shape == (2, 8)
    assert outputs["pi_logits"].shape == (2, 1)
    assert outputs["mu"].shape == (2, 1, 8)
    assert outputs["logvar"].shape == (2, 1, 8)
    assert outputs["reward_pred"].shape == (2, 1)
    assert outputs["done_logit"].shape == (2, 1)


def test_gaussian_gru_observe_sequence_shapes_and_backward():
    latent_dim = 8
    action_dim = 3
    num_gaussians = 1
    model = GaussianGRUDynamics(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=16,
        num_gaussians=num_gaussians,
        device="cpu",
    )
    latents = torch.randn(2, 5, latent_dim)
    actions = torch.randn(2, 4, action_dim)
    dones = torch.zeros(2, 4, dtype=torch.bool)

    outputs = model.observe_sequence(latents[:, :-1], actions, dones)

    assert set(outputs) == {
        "next_latents",
        "pi_logits",
        "mu",
        "logvar",
        "reward_preds",
        "done_logits",
    }
    assert outputs["next_latents"].shape == (2, 4, latent_dim)
    assert outputs["pi_logits"].shape == (2, 4, num_gaussians)
    assert outputs["mu"].shape == (2, 4, num_gaussians, latent_dim)
    assert outputs["logvar"].shape == (2, 4, num_gaussians, latent_dim)
    assert outputs["reward_preds"].shape == (2, 4, 1)
    assert outputs["done_logits"].shape == (2, 4, 1)

    targets = latents[:, 1:]
    logvar = torch.clamp(outputs["logvar"], min=-10.0, max=10.0)
    log_prob = -0.5 * (
        torch.log(torch.tensor(2.0 * torch.pi))
        + logvar
        + (targets.unsqueeze(2) - outputs["mu"]).pow(2) / torch.exp(logvar)
    ).sum(dim=-1)
    latent_nll = -torch.logsumexp(
        log_prob + F.log_softmax(outputs["pi_logits"], dim=-1),
        dim=-1,
    ).mean()
    reward_loss = outputs["reward_preds"].pow(2).mean()
    done_loss = F.binary_cross_entropy_with_logits(
        outputs["done_logits"].squeeze(-1),
        dones.float(),
    )
    loss = latent_nll + reward_loss + done_loss

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())
