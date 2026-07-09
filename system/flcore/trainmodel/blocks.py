"""Reusable building blocks for UNet-style segmentation decoders.

Extension points:
    - Replace ConvBnRelu with attention-augmented variants.
    - Subclass DecoderBlock to inject disaster-aware attention or
      prototype-guided feature modulation between the skip concatenation
      and the output convolution.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU. The workhorse block of the decoder."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 padding: int = 1, bias: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Single UNet decoder stage: upsample -> concat skip -> double conv.

    Parameters
    ----------
    in_ch : int
        Channels of the feature map being upsampled (from deeper level).
    skip_ch : int
        Channels of the skip connection from the encoder.
    out_ch : int
        Output channels after the double convolution.

    Extension point: override ``forward`` to insert attention between
    the concatenation and the convolutions. The skip tensor is passed
    separately so subclasses can modulate it before concatenation.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                              align_corners=False)
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)

    def forward(self, x: torch.Tensor,
                skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_ch, H, W)
            Feature map from the deeper decoder stage.
        skip : (B, skip_ch, 2H, 2W) or None
            Encoder skip connection at the matching resolution.

        Returns
        -------
        (B, out_ch, 2H, 2W)
        """
        x = self.up(x)
        if skip is not None:
            # Handle size mismatch from non-power-of-2 inputs
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
