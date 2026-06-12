"""Dreamer-style convolutional image decoder."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class DreamerImageDecoder(nn.Module):
    """Decode RSSM latent tensors into normalized image reconstructions."""

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        input_dim: int = 230,
        output_shape: tuple[int, int, int] | list[int] = (3, 64, 64),
        depth: int = 32,
        activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        merged = dict(config or {})
        merged.update(kwargs)
        input_dim = int(merged.get("input_dim", input_dim))
        if "output_shape" in merged:
            output_shape = merged["output_shape"]
        depth = int(merged.get("depth", depth))
        activation = str(merged.get("activation", activation))

        shape = tuple(int(value) for value in output_shape)
        if len(shape) != 3:
            raise ValueError(f"output_shape must be [C, H, W], got {shape}.")
        channels, height, width = shape
        if channels not in {1, 3}:
            raise ValueError(f"DreamerImageDecoder expects channel-first images, got shape {shape}.")
        if height != 64 or width != 64:
            raise ValueError("DreamerImageDecoder currently expects 64x64 images.")

        self.input_dim = input_dim
        self.output_shape = shape
        self.depth = depth
        self.fc = nn.Linear(input_dim, 8 * depth * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8 * depth, 4 * depth, kernel_size=4, stride=2, padding=1),
            _activation(activation),
            nn.ConvTranspose2d(4 * depth, 2 * depth, kernel_size=4, stride=2, padding=1),
            _activation(activation),
            nn.ConvTranspose2d(2 * depth, depth, kernel_size=4, stride=2, padding=1),
            _activation(activation),
            nn.ConvTranspose2d(depth, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        sequence = latent.dim() == 3
        if latent.dim() not in {2, 3}:
            raise ValueError(f"Expected latent tensor [B,D] or [B,T,D], got {tuple(latent.shape)}.")

        if sequence:
            batch, time, dim = latent.shape
            flat = latent.reshape(batch * time, dim)
        else:
            batch, dim = latent.shape
            time = None
            flat = latent
        if dim != self.input_dim:
            raise ValueError(f"Expected latent dim {self.input_dim}, got {dim}.")

        hidden = self.fc(flat).reshape(flat.shape[0], 8 * self.depth, 4, 4)
        decoded = self.decoder(hidden)
        if sequence:
            return decoded.reshape(batch, time, *self.output_shape)
        return decoded


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    return nn.ELU()


__all__ = ["DreamerImageDecoder"]
