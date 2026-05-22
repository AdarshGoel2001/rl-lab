"""Simple Gaussian GRU latent dynamics model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class GaussianGRUDynamics(nn.Module):
    """GRU dynamics head with a single-Gaussian default output contract."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        merged = dict(config or {})
        if kwargs:
            merged.update(kwargs)
        self.config = merged

        self.latent_dim = int(self.config.get("latent_dim", 32))
        self.hidden_dim = int(self.config.get("hidden_dim", 256))
        self.action_dim = int(self.config.get("action_dim", 0))
        self.num_gaussians = int(self.config.get("num_gaussians", 1))
        self.dropout = float(self.config.get("dropout", 0.0))
        self.device = torch.device(self.config.get("device", "cpu"))

        self.gru = nn.GRUCell(self.latent_dim + self.action_dim, self.hidden_dim)
        self.pi_head = nn.Linear(self.hidden_dim, self.num_gaussians)
        self.mu_head = nn.Linear(self.hidden_dim, self.latent_dim * self.num_gaussians)
        self.logvar_head = nn.Linear(self.hidden_dim, self.latent_dim * self.num_gaussians)
        self.reward_head = nn.Linear(self.hidden_dim, 1)
        self.done_head = nn.Linear(self.hidden_dim, 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.hidden: Optional[torch.Tensor] = None

        self.to(self.device)

    def observe(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = False,
        return_mixture: bool = False,
        temperature: float = 1.0,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        latent = latent.to(self.device)
        action = action.to(self.device)
        batch_size = latent.shape[0]

        if self.hidden is None or self.hidden.shape[0] != batch_size:
            self.hidden = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        if dones is not None:
            if isinstance(dones, np.ndarray):
                dones = torch.from_numpy(dones)
            dones = dones.to(self.device).bool()
            continuation = (~dones).float().unsqueeze(-1)
            self.hidden = self.hidden * continuation

        gru_input = torch.cat([latent, action], dim=-1)
        self.hidden = self.gru(gru_input, self.hidden)
        head_input = self.dropout_layer(self.hidden)

        pi_logits = self.pi_head(head_input)
        mu = self.mu_head(head_input).view(batch_size, self.num_gaussians, self.latent_dim)
        logvar = self.logvar_head(head_input).view(batch_size, self.num_gaussians, self.latent_dim)
        next_latent = self._select_next_latent(
            pi_logits,
            mu,
            logvar,
            deterministic=deterministic,
            temperature=temperature,
        )

        if not return_mixture:
            return {
                "next_latent": next_latent,
                "hidden": self.hidden,
            }

        return {
            "next_latent": next_latent,
            "hidden": self.hidden,
            "pi_logits": pi_logits,
            "mu": mu,
            "logvar": logvar,
            "reward_pred": self.reward_head(head_input),
            "done_logit": self.done_head(head_input),
        }

    def observe_sequence(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        dones: torch.Tensor,
        *,
        deterministic: Optional[bool] = None,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, _ = latent.shape
        latent = latent.to(self.device)
        action = action.to(self.device)
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones)
        dones = dones.to(self.device)

        hidden = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        next_latents = []
        pi_logits = []
        mus = []
        logvars = []
        reward_preds = []
        done_logits = []

        for t in range(sequence_length):
            if t > 0:
                continuation = (~dones[:, t - 1].bool()).float().unsqueeze(-1)
                hidden = hidden * continuation

            gru_input = torch.cat([latent[:, t], action[:, t]], dim=-1)
            hidden = self.gru(gru_input, hidden)
            head_input = self.dropout_layer(hidden)

            logits = self.pi_head(head_input)
            mu = self.mu_head(head_input).view(
                batch_size,
                self.num_gaussians,
                self.latent_dim,
            )
            logvar = self.logvar_head(head_input).view(
                batch_size,
                self.num_gaussians,
                self.latent_dim,
            )

            next_latent = self._select_next_latent(
                logits,
                mu,
                logvar,
                deterministic=bool(deterministic),
                temperature=temperature,
            )

            next_latents.append(next_latent)
            pi_logits.append(logits)
            mus.append(mu)
            logvars.append(logvar)
            reward_preds.append(self.reward_head(head_input))
            done_logits.append(self.done_head(head_input))

        return {
            "next_latents": torch.stack(next_latents, dim=1),
            "pi_logits": torch.stack(pi_logits, dim=1),
            "mu": torch.stack(mus, dim=1),
            "logvar": torch.stack(logvars, dim=1),
            "reward_preds": torch.stack(reward_preds, dim=1),
            "done_logits": torch.stack(done_logits, dim=1),
        }

    def _select_next_latent(
        self,
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        deterministic: bool,
        temperature: float,
    ) -> torch.Tensor:
        if deterministic:
            component_idx = torch.argmax(pi_logits, dim=-1)
        else:
            pi_probs = torch.softmax(pi_logits / temperature, dim=-1)
            component_idx = torch.distributions.Categorical(pi_probs).sample()

        batch_idx = torch.arange(mu.shape[0], device=mu.device)
        selected_mu = mu[batch_idx, component_idx]
        if deterministic:
            return selected_mu

        selected_logvar = logvar[batch_idx, component_idx]
        return selected_mu + torch.exp(0.5 * selected_logvar) * torch.randn_like(selected_mu)

    def reset_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        self.hidden = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        return {"hidden": self.hidden}


__all__ = ["GaussianGRUDynamics"]
