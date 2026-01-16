from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import numpy as np

from .base import BaseController


class NoiseSchedule:

    def __init__(self, num_steps: int, schedule_type: str = "cosine"):
        self.num_steps = num_steps
        self.schedule_type = schedule_type

        if self.schedule_type == "cosine":
           # f(t) = cos((t/T + s) / (1 + s) * π/2)²
           s = 0.008
           t = torch.linspace(0, num_steps , num_steps + 1 ) 
           f_t = torch.cos((t/num_steps + s) / (1 + s) * np.pi/2)**2
           alpha_bars = f_t / f_t[0]
           # shape of alpha bars is (num_steps,)
           betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
           betas = torch.clamp(betas, max=0.999)
           self.betas = betas
           self.alphas = 1 - betas
           self.alpha_bars = torch.cumprod(self.alphas, dim=0)
           self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
           self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
           # shape of betas is (num_steps,)
           # shape of alpha bars is (num_steps,)

    def add_noise(
        self,
        x_0: torch.Tensor, # shape is (B, horizon, action_dim)
        noise: torch.Tensor, # shape is (B, horizon, action_dim)
        timesteps: torch.Tensor, # shape is (B,)
    ) -> torch.Tensor:
        sqrt_alpha_bars = self.sqrt_alpha_bars[timesteps] #sha
        sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars[timesteps]
        sqrt_alpha_bars_t = sqrt_alpha_bars.view(-1, 1, 1)
        sqrt_one_minus_alpha_bars_t = sqrt_one_minus_alpha_bars.view(-1, 1, 1)
        x_t = sqrt_alpha_bars_t * x_0 + sqrt_one_minus_alpha_bars_t * noise
        return x_t # shape is (B, horizon, action_dim)

    def to(self, device: torch.device) -> "NoiseSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x

class DiffusionTransformer(nn.Module):
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        num_diffusion_steps: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):

        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.hidden_dim = hidden_dim
        self.time_embed = nn.Embedding(num_diffusion_steps, hidden_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, horizon, hidden_dim)*0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers) ])
        
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        
        

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        B, H, _ = noisy_actions.shape
        time_emb = self.time_embed(timestep)
        obs_feat = self.obs_encoder(observations)
        conditioning = time_emb + obs_feat
        x = self.action_proj(noisy_actions) + self.pos_encoding
        cond_expanded = conditioning.unsqueeze(1)
        x = x + cond_expanded
        for block in self.transformer_blocks:
            x = block(x)
        noise_pred = self.output_proj(x)
        return noise_pred


def denoise_step(
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    t: int,
    schedule: NoiseSchedule,
) -> torch.Tensor:
    alpha_t = schedule.alphas[t]
    beta_t = schedule.betas[t]
    alpha_bar_t = schedule.alpha_bars[t]
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
    mean = 1.0 / sqrt_alpha_t * (x_t - (beta_t / sqrt_one_minus_alpha_bar_t * noise_pred) )

    if t > 0:
        noise = torch.randn_like(x_t)
        x_t_minus_1 = mean + torch.sqrt(beta_t) * noise
    else:
        x_t_minus_1 = mean
    return x_t_minus_1



class DiffusionPolicyController(nn.Module):

    def __init__(self, config: Dict[str, Any] | None = None, **kwargs: Any):
        super().__init__()
        cfg = dict(config or {})
        cfg.update(kwargs)

        # Extract config
        self.obs_dim = cfg["obs_dim"]
        self.action_dim = cfg["action_dim"]
        self.horizon = cfg.get("horizon", 16)
        self.num_diffusion_steps = cfg.get("num_diffusion_steps", 100)
        self.device = torch.device(cfg.get("device", "cpu"))
        self.hidden_dim = cfg.get("hidden_dim", 256)
        self.num_layers = cfg.get("num_layers", 4)
        self.num_heads = cfg.get("num_heads", 4)
        self.dropout = cfg.get("dropout", 0.1)

        # Build network
        self.network = DiffusionTransformer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            horizon=self.horizon,
            num_diffusion_steps=self.num_diffusion_steps,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )

        # Build schedule
        self.schedule = NoiseSchedule(
            num_steps=self.num_diffusion_steps,
            schedule_type=cfg.get("schedule_type", "cosine"),
        )

        self.to(self.device)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through network (predicts noise).
        
        Args:
            noisy_actions: (B, H, action_dim)
            timesteps: (B,)
            observations: (B, obs_dim)
        
        Returns:
            predicted_noise: (B, H, action_dim)
        """
        return self.network(noisy_actions, timesteps, observations)

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        B = observations.shape[0]
        timesteps = torch.randint(0, self.num_diffusion_steps, (B,), device=self.device)
        noise = torch.randn_like(actions)
        noisy_actions = self.schedule.add_noise(actions, noise, timesteps)
        noise_pred = self.forward(noisy_actions, timesteps, observations)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss

        
    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        B = observations.shape[0]
        x = torch.randn (B, self.horizon, self.action_dim, device=self.device)
        for t in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.forward(x, timesteps, observations)
            x = denoise_step(x, noise_pred, t, self.schedule)
        return x    


    def parameters(self):
        """Return network parameters for optimizer."""
        return self.network.parameters()

    def to(self, device: torch.device):
        """Move controller to device."""
        super().to(device)
        self.device = device
        self.schedule = self.schedule.to(device)
        return self