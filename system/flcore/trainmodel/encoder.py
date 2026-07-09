"""Shared ResNet50 encoder with multi-scale feature extraction.

Wraps ``torchvision.models.resnet50`` to expose intermediate feature maps
at five spatial scales (stem, layer1-layer4).  These are consumed by the
decoder as skip connections and by the fusion module for change detection.

In federated learning, this encoder's parameters are the SHARED component
that gets aggregated across clients (FedAvg, FedProx, SCAFFOLD, etc.).

Extension points:
    - ``pretrained`` flag to toggle ImageNet weights.
    - ``freeze()`` / ``unfreeze()`` helpers for controlling encoder training.
    - Swap ``torchvision.models.resnet50`` for resnet34/resnet101 via
      the ``backbone`` parameter.
    - Access ``out_channels`` for downstream module construction.
    - Access ``bottleneck_channels`` for prototype extraction modules.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """Multi-scale feature extractor based on torchvision ResNet.

    Parameters
    ----------
    backbone : str
        Torchvision model name. One of 'resnet18', 'resnet34', 'resnet50',
        'resnet101'. Default: 'resnet50'.
    pretrained : bool
        If True, load ImageNet-pretrained weights. Default: True.
    in_channels : int
        Number of input channels. If != 3, the first conv layer is replaced
        with a new Conv2d that accepts ``in_channels`` input channels.
        Default: 3.

    Attributes
    ----------
    out_channels : List[int]
        Channel counts for each feature scale [stem, layer1, layer2, layer3,
        layer4], e.g. [64, 256, 512, 1024, 2048] for ResNet50.
    bottleneck_channels : int
        Channel count at the deepest level (layer4 output).
    """

    # Channel counts per stage for each backbone family
    _CHANNEL_TABLE = {
        "resnet18":  [64, 64,  128, 256,  512],
        "resnet34":  [64, 64,  128, 256,  512],
        "resnet50":  [64, 256, 512, 1024, 2048],
        "resnet101": [64, 256, 512, 1024, 2048],
    }

    _BACKBONE_TABLE = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
    }

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 in_channels: int = 3):
        super().__init__()

        if backbone not in self._CHANNEL_TABLE:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from {list(self._CHANNEL_TABLE.keys())}."
            )

        # Load torchvision backbone
        factory, weights_enum = self._BACKBONE_TABLE[backbone]
        weights = weights_enum if pretrained else None
        resnet = factory(weights=weights)

        # --- Stem: conv1 + bn1 + relu (stride /2) ---
        if in_channels != 3:
            # Replace first conv to accept different channel count
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                          padding=3, bias=False),
                resnet.bn1,
                resnet.relu,
            )
        else:
            self.stem = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
            )

        # --- Pooling + residual stages ---
        self.pool = resnet.maxpool     # stride /4
        self.layer1 = resnet.layer1    # stride /4
        self.layer2 = resnet.layer2    # stride /8
        self.layer3 = resnet.layer3    # stride /16
        self.layer4 = resnet.layer4    # stride /32

        self.out_channels: List[int] = self._CHANNEL_TABLE[backbone]
        self.bottleneck_channels: int = self.out_channels[-1]

    def freeze(self) -> "ResNetEncoder":
        """Disable gradient updates for all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
        return self

    def unfreeze(self) -> "ResNetEncoder":
        """Enable gradient updates for all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.

        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        features : list of 5 tensors
            [s0, s1, s2, s3, s4] from shallowest to deepest.
            Shapes for ResNet50 with 128x128 input:
              s0: (B,   64, H/2,  W/2)   = (B,   64, 64, 64)
              s1: (B,  256, H/4,  W/4)   = (B,  256, 32, 32)
              s2: (B,  512, H/8,  W/8)   = (B,  512, 16, 16)
              s3: (B, 1024, H/16, W/16)  = (B, 1024,  8,  8)
              s4: (B, 2048, H/32, W/32)  = (B, 2048,  4,  4)
        """
        s0 = self.stem(x)                   # (B, 64, H/2, W/2)
        s1 = self.layer1(self.pool(s0))      # (B, 256, H/4, W/4)
        s2 = self.layer2(s1)                 # (B, 512, H/8, W/8)
        s3 = self.layer3(s2)                 # (B, 1024, H/16, W/16)
        s4 = self.layer4(s3)                 # (B, 2048, H/32, W/32)
        return [s0, s1, s2, s3, s4]
