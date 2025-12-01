"""MDN-RNN dynamics scaffold mirroring the World Models 'M' component.

Use this as homework: flesh out the recurrent core, mixture density heads,
training losses, and imagination helpers when you're ready.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MixtureSameFamily, Categorical, Normal

class MDNRNNDynamics(nn.Module):
    """Placeholder Mixture Density RNN that predicts the next latent distribution."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        merged = dict(config or {})
        if kwargs:
            merged.update(kwargs)
        self.config = merged

        self.latent_dim = int(self.config.get("latent_dim", 32))
        self.hidden_dim = int(self.config.get("hidden_dim", 256))
        self.action_dim = int(self.config.get("action_dim", 0))
        self.num_gaussians = int(self.config.get("num_gaussians", 5))
        self.dropout = float(self.config.get("dropout", 0.0))
        self.device = torch.device(self.config.get("device", "cpu"))

        self.rnn = nn.LSTM(self.latent_dim + self.action_dim, self.hidden_dim, batch_first=True)

        # Latent prediction heads (Mixture Density Network)
        self.pi_head = nn.Linear(self.hidden_dim, self.num_gaussians)
        self.mu_head = nn.Linear(self.hidden_dim, self.latent_dim*self.num_gaussians)
        self.logvar_head = nn.Linear(self.hidden_dim, self.latent_dim*self.num_gaussians)


        # Auxiliary prediction heads (from shared hidden state)
        self.reward_head = nn.Linear(self.hidden_dim, 1)  # Predict next reward
        self.done_head = nn.Linear(self.hidden_dim, 1)    # Predict next done (logit)
        

    def observe(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = False,
        return_mixture: Optional[bool] = False,
        temperature: float = 1.0,
        **_: Any,
    ) -> Dict[str, torch.Tensor]:

        if dones is not None:
            if isinstance(dones, np.ndarray):
                dones = torch.from_numpy(dones).to(self.device)
            
            #convert positive dones to 0.0 and false dones to 1.0

            dones_mask = ~dones
            mask = dones_mask.float()
            self.hidden = self.hidden * mask.view(1, -1, 1)
            self.cell = self.cell * mask.view(1, -1, 1)
            
        batch_size = latent.shape[0]
        latent_concat = (torch.cat([latent, action], dim=-1)).unsqueeze(1)
        rnn_output, (self.hidden, self.cell) = self.rnn(latent_concat, (self.hidden, self.cell))
        rnn_output = rnn_output.squeeze(1)

        # Predict next latent (mixture of Gaussians)
        pi_logits = self.pi_head(rnn_output)
        mu_flat = self.mu_head(rnn_output)
        logvar_flat= self.logvar_head(rnn_output)
        mu = mu_flat.view(batch_size, self.num_gaussians, self.latent_dim)
        logvar = logvar_flat.view(batch_size, self.num_gaussians, self.latent_dim)
        pi_probs = torch.softmax(pi_logits/temperature, dim=-1)

        # Predict next reward and done (from same hidden state)
        reward_pred = self.reward_head(rnn_output)  # (B, 1)
        done_logit = self.done_head(rnn_output)     # (B, 1) - logit for BCE
        if deterministic:
            most_likely = torch.argmax(pi_probs, dim=-1)
            next_latent = mu[torch.arange(batch_size), most_likely]
        else:
            component_dist = torch.distributions.Categorical(pi_probs)
            component_idx = component_dist.sample()
            mu_selected = mu[torch.arange(batch_size), component_idx]
            logvar_selected = logvar[torch.arange(batch_size), component_idx]
            next_latent = mu_selected + torch.exp(0.5*logvar_selected) * torch.randn_like(mu_selected)
        if return_mixture:
            return {
                    "next_latent": next_latent,
                    "hidden": (self.hidden, self.cell),
                    "pi_logits": pi_logits,
                    "mu": mu,
                    "logvar": logvar,
                    "reward_pred": reward_pred,
                    "done_logit": done_logit,
                   }
        else:
                return {
                    "next_latent": next_latent,
                    "hidden": (self.hidden, self.cell),
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
        batch_size = latent.shape[0]
        self.hidden = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        self.cell = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        
        next_latent_list = []
        pi_list = []
        mu_list = []
        logvar_list = []
        reward_list = []
        done_list = []

        for t in range(latent.shape[1]):
            latent_t = latent[:, t, :]
            action_t = action[:, t, :]
            dones_t = dones[:, t-1] if t > 0 else None
            result = self.observe(latent_t, action_t, dones_t, return_mixture=True, temperature=temperature)
            next_latent_list.append(result["next_latent"])
            pi_list.append(result["pi_logits"])
            mu_list.append(result["mu"])
            logvar_list.append(result["logvar"])
            reward_list.append(result["reward_pred"])
            done_list.append(result["done_logit"])

        return {
            "next_latents": torch.stack(next_latent_list, dim=1),
            "pi_logits": torch.stack(pi_list, dim=1),
            "mu": torch.stack(mu_list, dim=1),
            "logvar": torch.stack(logvar_list, dim=1),
            "reward_preds": torch.stack(reward_list, dim=1),
            "done_logits": torch.stack(done_list, dim=1),
        }

    def reset_state(self, batch_size: int) -> dict[str, torch.Tensor]:
        self.hidden = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        self.cell = torch.zeros(1, batch_size, self.hidden_dim, device=self.device)
        return {"hidden": (self.hidden, self.cell)}
        


__all__ = ["MDNRNNDynamics"]
