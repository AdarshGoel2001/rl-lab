"""Atari-focused observation decoders for 84x84 grayscale stacks."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .base import BaseObservationDecoder


class AtariConvObservationDecoder(BaseObservationDecoder):
    """Transposed-convolution decoder tuned for Atari 84x84 frame stacks."""

    def _build_decoder(self) -> None:
        activation_name = self.config.get("activation", "elu")
        activation_map = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
        }
        self.activation = activation_map.get(activation_name, nn.ELU())

        if self.output_shape is None or len(self.output_shape) != 3:
            raise ValueError("AtariConvObservationDecoder expects a 3D output_shape")

        channels, height, width = self._normalize_shape(self.output_shape)
        self.channels = channels
        self.height = height
        self.width = width

        # Base projection size determines upsampling factor (21 -> 42 -> 84 by default)
        self.initial_size = self.config.get("initial_size", 21)
        self.base_channels = self.config.get("base_channels", max(32, channels * 4))
        self.mid_channels = self.config.get("mid_channels", self.base_channels // 2)

        self.project = nn.Linear(
            self.representation_dim,
            self.base_channels * self.initial_size * self.initial_size,
        )

        final_activation_name = self.config.get("output_activation", "sigmoid")
        self.final_activation = activation_map.get(final_activation_name, nn.Sigmoid())

        # Three-stage decoder keeps the architecture shallow but expressive enough for 84x84
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                self.base_channels,
                self.mid_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.activation,
            nn.ConvTranspose2d(
                self.mid_channels,
                max(self.mid_channels // 2, self.channels),
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            self.activation,
            nn.Conv2d(
                max(self.mid_channels // 2, self.channels),
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        projected = self.project(latent)
        x = projected.view(batch_size, self.base_channels, self.initial_size, self.initial_size)
        reconstruction = self.decoder(x)
        if self.final_activation is not None:
            reconstruction = self.final_activation(reconstruction)

        if self._output_was_hw_last:
            reconstruction = reconstruction.permute(0, 2, 3, 1).contiguous()

        return reconstruction.view(batch_size, *self.output_shape)

    def _normalize_shape(self, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Infer channel placement; support CHW and HWC inputs."""
        if shape[0] <= 16:  # treat as CHW
            self._output_was_hw_last = False
            return int(shape[0]), int(shape[1]), int(shape[2])
        if shape[-1] <= 16:  # treat as HWC
            self._output_was_hw_last = True
            return int(shape[-1]), int(shape[0]), int(shape[1])
        raise ValueError(
            "Unable to infer channel dimension from output_shape; provide CHW or HWC ordering"
        )


__all__ = ["AtariConvObservationDecoder"]

