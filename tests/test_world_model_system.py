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

import torch.nn.functional as F

from src.components.dynamics.mdn_rnn import MDNRNNDynamics
from src.components.representation_learners.conv_vae import ConvVAERepresentationLearner


def _has_any_grad(module: torch.nn.Module) -> bool:
    return any(param.grad is not None for param in module.parameters())


def test_vae_forward_backward():
    vae = ConvVAERepresentationLearner(input_shape=(64, 64, 1), latent_dim=8)
    observations = torch.rand(2, 4, 64, 64, 1)

    features = vae.observe_sequence(observations)
    decoded = vae.decode(features["latent"])
    reconstruction_loss = F.binary_cross_entropy_with_logits(decoded, observations)
    kl = 0.5 * (
        features["mean"].pow(2) + features["logvar"].exp() - 1.0 - features["logvar"]
    ).mean()
    loss = reconstruction_loss + kl

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert _has_any_grad(vae)


def test_mdn_forward_backward():
    latent_dim = 8
    action_dim = 3
    mdn = MDNRNNDynamics(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=16,
        num_gaussians=3,
        device="cpu",
    )
    latents = torch.randn(2, 5, latent_dim)
    actions = torch.randn(2, 5, action_dim)
    dones = torch.zeros(2, 5, dtype=torch.bool)

    outputs = mdn.observe_sequence(latents[:, :-1], actions[:, :-1], dones[:, :-1])
    targets = latents[:, 1:]
    targets_expanded = targets.unsqueeze(2)
    logvar = torch.clamp(outputs["logvar"], min=-10.0, max=10.0)
    log_prob = -0.5 * (
        torch.log(torch.tensor(2.0 * torch.pi))
        + logvar
        + (targets_expanded - outputs["mu"]).pow(2) / torch.exp(logvar)
    )
    log_prob = log_prob.sum(dim=-1)
    log_pi = F.log_softmax(outputs["pi_logits"], dim=-1)
    latent_nll = -torch.logsumexp(log_prob + log_pi, dim=-1).mean()
    reward_loss = outputs["reward_preds"].pow(2).mean()
    done_loss = F.binary_cross_entropy_with_logits(
        outputs["done_logits"].squeeze(-1),
        torch.zeros_like(outputs["done_logits"].squeeze(-1)),
    )
    loss = latent_nll + reward_loss + done_loss

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert _has_any_grad(mdn)
