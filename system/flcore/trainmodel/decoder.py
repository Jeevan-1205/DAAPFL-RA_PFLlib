"""UNet decoder and segmentation head for damage assessment.

Takes the multi-scale fused features (after Siamese difference fusion)
and progressively upsamples them back to the input spatial resolution
using skip connections from the encoder.

In personalized FL (FedPer), the decoder is the PRIVATE component that
stays local to each client while the encoder is aggregated.

Extension points:
    - Subclass ``UNetDecoder`` and override ``forward`` to insert
      disaster-aware attention between decoder stages.
    - Replace ``SegmentationHead`` with multi-task heads (localization +
      damage) without changing the decoder.
    - Pass ``return_features=True`` to expose intermediate decoder stage
      features for prototype computation at decoded feature levels.
"""
from __future__ import annotations
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.trainmodel.blocks import DecoderBlock


class UNetDecoder(nn.Module):
    """Progressive upsampling decoder with skip connections.

    Parameters
    ----------
    encoder_channels : list of int
        Channel counts from the encoder at each scale, ordered
        shallowest to deepest.  E.g. [64, 256, 512, 1024, 2048]
        for ResNet50 with diff fusion.
    decoder_channels : list of int
        Desired output channels for each decoder stage, from deepest
        to shallowest.  Length must be ``len(encoder_channels) - 1``.
        Default: (256, 128, 64, 32).

    Shapes (ResNet50, 128x128 input, diff fusion)
    -----------------------------------------------
    Input features (shallowest → deepest):
        [64, 64, 64]  [256, 32, 32]  [512, 16, 16]  [1024, 8, 8]  [2048, 4, 4]
         skip0          skip1          skip2           skip3         bottleneck

    Decoder stages (deepest → shallowest):
        d3 = DecoderBlock(2048, 1024, 256)  → (B, 256, 8, 8)
        d2 = DecoderBlock(256,   512, 128)  → (B, 128, 16, 16)
        d1 = DecoderBlock(128,   256,  64)  → (B,  64, 32, 32)
        d0 = DecoderBlock(64,     64,  32)  → (B,  32, 64, 64)

    Output: (B, 32, 64, 64)  ← fed to SegmentationHead
    """

    def __init__(self, encoder_channels: List[int],
                 decoder_channels: List[int] = (256, 128, 64, 32)):
        super().__init__()

        num_skips = len(encoder_channels) - 1
        if len(decoder_channels) != num_skips:
            raise ValueError(
                f"decoder_channels length ({len(decoder_channels)}) must equal "
                f"len(encoder_channels) - 1 ({num_skips})."
            )

        # Build decoder blocks from deepest to shallowest.
        # Each block takes:
        #   - in_ch: channels from the previous (deeper) stage
        #   - skip_ch: channels from the corresponding encoder skip
        #   - out_ch: desired output channels
        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[-1]  # bottleneck channels
        for i, out_ch in enumerate(decoder_channels):
            # Skip connections are indexed from second-deepest to shallowest
            skip_idx = num_skips - 1 - i
            skip_ch = encoder_channels[skip_idx]
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.out_channels = decoder_channels[-1]

    def forward(
        self,
        features: List[torch.Tensor],
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        features : list of tensors
            Encoder feature maps ordered shallowest to deepest.
            Length must match the encoder_channels used at construction.
        return_features : bool
            If True, also return the output of each decoder block ordered
            deepest to shallowest. Default: False.

        Returns
        -------
        x : (B, decoder_channels[-1], H/2, W/2)
            Decoded feature map at the shallowest decoder level.
        decoder_features : list of tensors, optional
            Intermediate decoder block outputs, returned only when
            ``return_features=True``.
        """
        # Separate bottleneck from skip connections
        skips = features[:-1]   # shallowest to second-deepest
        x = features[-1]        # deepest (bottleneck)
        decoder_features = []

        # Decode from deep to shallow, consuming skips in reverse
        for i, block in enumerate(self.blocks):
            skip_idx = len(skips) - 1 - i
            x = block(x, skips[skip_idx])
            decoder_features.append(x)

        if return_features:
            return x, decoder_features
        return x


class SegmentationHead(nn.Module):
    """Final 1x1 convolution mapping decoded features to class logits.

    This is the component that PFLlib's ``BaseHeadSplit`` treats as the
    ``head`` (personalized in FedPer) or the ``fc`` layer (replaced with
    ``nn.Identity()`` for head-splitting algorithms).

    Parameters
    ----------
    in_channels : int
        Number of input channels from the decoder output.
    num_classes : int
        Number of segmentation classes. Default: 5 (xBD damage levels).
    """

    def __init__(self, in_channels: int, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)

        Returns
        -------
        logits : (B, num_classes, H, W)
        """
        return self.conv(x)
