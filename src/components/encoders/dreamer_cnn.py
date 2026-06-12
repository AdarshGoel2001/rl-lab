"""Dreamer-style convolutional image encoder."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class DreamerImageEncoder(nn.Module):
    """Encode image observations into RSSM feature vectors.

    Accepts either `[B, C, H, W]` or `[B, T, C, H, W]` tensors. Inputs may be
    uint8 `[0, 255]` pixels or normalized floats.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        *,
        input_shape: tuple[int, int, int] | list[int] = (3, 64, 64),
        output_dim: int = 1024,
        depth: int = 32,
        activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        merged = dict(config or {})
        merged.update(kwargs)
        if "input_shape" in merged:
            input_shape = merged["input_shape"]
        output_dim = int(merged.get("output_dim", output_dim))
        depth = int(merged.get("depth", depth))
        activation = str(merged.get("activation", activation))

        shape = tuple(int(value) for value in input_shape)
        if len(shape) != 3:
            raise ValueError(f"input_shape must be [C, H, W], got {shape}.")
        channels, height, width = shape
        if channels not in {1, 3}:
            raise ValueError(f"DreamerImageEncoder expects channel-first images, got shape {shape}.")
        if height != 64 or width != 64:
            raise ValueError("DreamerImageEncoder currently expects 64x64 images.")

        act = _activation(activation)
        self.input_shape = shape
        self.output_dim = output_dim
        self.depth = depth
        self.convs = nn.Sequential(
            nn.Conv2d(channels, depth, kernel_size=4, stride=2, padding=1),
            act,
            nn.Conv2d(depth, 2 * depth, kernel_size=4, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(2 * depth, 4 * depth, kernel_size=4, stride=2, padding=1),
            _activation(activation),
            nn.Conv2d(4 * depth, 8 * depth, kernel_size=4, stride=2, padding=1),
            _activation(activation),
        )
        self.proj = nn.Linear(8 * depth * 4 * 4, output_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sequence = observations.dim() == 5
        if observations.dim() not in {4, 5}:
            raise ValueError(f"Expected image tensor [B,C,H,W] or [B,T,C,H,W], got {tuple(observations.shape)}.")

        if sequence:
            batch, time, channels, height, width = observations.shape
            flat = observations.reshape(batch * time, channels, height, width)
        else:
            batch, channels, height, width = observations.shape
            time = None
            flat = observations

        flat = _normalize_pixels(flat)
        features = self.convs(flat).reshape(flat.shape[0], -1)
        embeddings = self.proj(features)
        if sequence:
            return embeddings.reshape(batch, time, self.output_dim)
        return embeddings


def _normalize_pixels(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(torch.float32)
    if tensor.numel() and float(tensor.detach().max().item()) > 1.5:
        tensor = tensor / 255.0
    return tensor


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    return nn.ELU()


__all__ = ["DreamerImageEncoder"]
