"""Siamese U-Net for building damage assessment on xBD.

Pipeline
--------
    Input:  (B, 6, H, W)  — 6 channels = [pre_RGB, post_RGB]
      ↓ split channels 0:3 and 3:6
    pre:  (B, 3, H, W)    post: (B, 3, H, W)
      ↓ shared ResNet50 encoder
    pre_feats:  [s0..s4]  post_feats: [s0..s4]
      ↓ diff fusion at each scale: post - pre
    fused: [f0..f4]
      ↓ UNet decoder with skip connections
    decoded: (B, 32, H/2, W/2)
      ↓ segmentation head (1x1 conv)
    logits: (B, 5, H/2, W/2)
      ↓ bilinear upsample to input resolution
    output: (B, 5, H, W)

PFLlib integration
------------------
    - ``self.fc`` is the segmentation head, following PFLlib's convention
      where ``main.py`` does ``args.head = model.fc`` for head-splitting
      algorithms (FedPer, FedRep, FedROD, etc.).
    - ``forward(x)`` takes a single 6-channel tensor, compatible with
      PFLlib's ``output = self.model(x)`` calling convention.
    - ``nn.CrossEntropyLoss(logits, mask)`` works directly since output
      shape is (B, C, H, W) and mask shape is (B, H, W).

Extension points
----------------
    - ``extract_bottleneck()`` — returns global-pooled deepest feature
      for prototype-based algorithms (FedProto, DAAPFL-RA).
    - ``fusion`` — replace with another module to change fusion strategy
      (absdiff, concat, attention-based).
    - ``fuse()`` — compatibility wrapper around ``self.fusion``.
    - ``encoder`` and ``decoder`` are separate ``nn.Module`` attributes,
      enabling independent parameter selection for aggregation vs.
      personalization in FL algorithms.
    - ``shared_encoder_state()`` / ``private_decoder_state()`` — explicit
      parameter separation helpers for custom FL strategies.
"""
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.trainmodel.encoder import ResNetEncoder
from flcore.trainmodel.decoder import UNetDecoder, SegmentationHead
from flcore.trainmodel.fusion import DifferenceFusion


class SiameseUNet(nn.Module):
    """Siamese encoder + diff fusion + UNet decoder for change detection.

    Parameters
    ----------
    backbone : str
        ResNet variant for the encoder. Default: 'resnet50'.
    pretrained : bool
        Load ImageNet-pretrained encoder weights. Default: True.
    in_channels : int
        Channels per image (pre or post). Default: 3 (RGB).
    num_classes : int
        Number of output segmentation classes. Default: 5.
    decoder_channels : tuple of int
        Channel counts for each decoder stage (deepest to shallowest).
        Must have length == len(encoder_out_channels) - 1. Default: (256, 128, 64, 32).
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 in_channels: int = 3, num_classes: int = 5,
                 decoder_channels: tuple = (256, 128, 64, 32)):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # ---- Shared encoder (aggregated in FL) ----
        self.encoder = ResNetEncoder(backbone, pretrained, in_channels)

        # ---- Feature fusion (pre/post change representation) ----
        self.fusion = DifferenceFusion()

        # ---- Private decoder (personalized in FL) ----
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=list(decoder_channels),
        )

        # ---- Segmentation head ----
        # Named ``fc`` to match PFLlib's BaseHeadSplit convention:
        #   args.head = copy.deepcopy(args.model.fc)
        #   args.model.fc = nn.Identity()
        #   args.model = BaseHeadSplit(args.model, args.head)
        self.fc = SegmentationHead(
            in_channels=self.decoder.out_channels,
            num_classes=num_classes,
        )

        # Cached for external use (prototype extraction, etc.)
        self._bottleneck_dim = self.encoder.bottleneck_channels

    # ------------------------------------------------------------------ #
    #  Fusion — extension point for custom change detection strategies
    # ------------------------------------------------------------------ #

    def fuse(self, feats_pre: List[torch.Tensor],
             feats_post: List[torch.Tensor]) -> List[torch.Tensor]:
        """Fuse pre/post features at each encoder scale.

        Override in a subclass for alternative strategies:
            - absolute difference:  |post - pre|
            - concatenation:        [pre, post]  (doubles channels)
            - attention-based:      attention(pre, post)

        Parameters
        ----------
        feats_pre, feats_post : lists of tensors
            Multi-scale encoder features, shallowest to deepest.

        Returns
        -------
        fused : list of tensors
            Same shapes as input (for diff/absdiff) or doubled channels
            (for concat).
        """
        return self.fusion(feats_pre, feats_post)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"input must be a torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 4:
            raise ValueError(f"input must have shape (B, C, H, W), got {tuple(x.shape)}.")

        expected_channels = 2 * self.in_channels
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"input channel count must be {expected_channels} "
                f"for stacked pre/post images, got {x.shape[1]}."
            )
        if x.shape[-2] <= 0 or x.shape[-1] <= 0:
            raise ValueError(f"input spatial dimensions must be positive, got {tuple(x.shape[-2:])}.")

    # ------------------------------------------------------------------ #
    #  Forward — single-tensor interface for PFLlib compatibility
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2*in_channels, H, W)
            Stacked pre/post image. Channels 0:in_channels are pre-disaster,
            channels in_channels:2*in_channels are post-disaster.

        Returns
        -------
        logits : (B, num_classes, H, W)
            Per-pixel class logits at the input spatial resolution.
        """
        self._validate_input(x)
        c = self.in_channels
        pre = x[:, :c, :, :]   # (B, 3, H, W)
        post = x[:, c:, :, :]  # (B, 3, H, W)
        input_size = x.shape[-2:]

        # Shared-weight encoding
        feats_pre = self.encoder(pre)
        feats_post = self.encoder(post)

        # Change detection fusion
        fused = self.fuse(feats_pre, feats_post)

        # Decode
        decoded = self.decoder(fused)

        # Classify
        logits = self.fc(decoded)

        # Restore to input resolution
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear",
                                   align_corners=False)

        return logits

    # ------------------------------------------------------------------ #
    #  Feature extraction — for prototype-based FL algorithms
    # ------------------------------------------------------------------ #

    def extract_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """Extract globally-pooled deepest fused feature.

        Useful for FedProto, DAAPFL-RA prototype computation.

        Parameters
        ----------
        x : (B, 2*in_channels, H, W)

        Returns
        -------
        features : (B, bottleneck_dim)
        """
        self._validate_input(x)
        c = self.in_channels
        feats_pre = self.encoder(x[:, :c])
        feats_post = self.encoder(x[:, c:])
        bottleneck = self.fuse(feats_pre, feats_post)[-1]  # deepest
        return F.adaptive_avg_pool2d(bottleneck, 1).flatten(1)

    @property
    def bottleneck_dim(self) -> int:
        """Dimensionality of the bottleneck feature vector."""
        return self._bottleneck_dim

    # ------------------------------------------------------------------ #
    #  Parameter separation — for custom FL aggregation strategies
    # ------------------------------------------------------------------ #

    def shared_encoder_state(self) -> Dict[str, torch.Tensor]:
        """Encoder parameters (aggregated across clients in FL)."""
        return {f"encoder.{k}": v for k, v in self.encoder.state_dict().items()}

    def private_decoder_state(self) -> Dict[str, torch.Tensor]:
        """Decoder + head parameters (kept local in personalized FL)."""
        state = {f"decoder.{k}": v
                 for k, v in self.decoder.state_dict().items()}
        state.update({f"fc.{k}": v
                      for k, v in self.fc.state_dict().items()})
        return state

    def load_shared_encoder(self, state: Dict[str, torch.Tensor]) -> None:
        """Load aggregated encoder weights without touching decoder/head."""
        clean = {k.replace("encoder.", "", 1): v for k, v in state.items()}
        self.encoder.load_state_dict(clean, strict=True)


# ====================================================================== #
#  Forward-pass test
# ====================================================================== #

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("SiameseUNet Forward-Pass Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- Build model ---
    model = SiameseUNet(
        backbone="resnet50",
        pretrained=False,   # skip download for CI
        in_channels=3,
        num_classes=5,
        decoder_channels=(256, 128, 64, 32),
    ).to(device)

    # --- Count parameters ---
    total = sum(p.numel() for p in model.parameters())
    encoder_p = sum(p.numel() for p in model.encoder.parameters())
    decoder_p = sum(p.numel() for p in model.decoder.parameters())
    head_p = sum(p.numel() for p in model.fc.parameters())
    print(f"Total parameters:   {total:>12,}")
    print(f"  Encoder (shared): {encoder_p:>12,}")
    print(f"  Decoder (private):{decoder_p:>12,}")
    print(f"  Head (fc):        {head_p:>12,}")
    print()

    # --- Forward pass ---
    B, H, W = 2, 128, 128
    x = torch.randn(B, 6, H, W, device=device)
    mask = torch.randint(0, 5, (B, H, W), device=device)

    t0 = time.time()
    logits = model(x)
    t1 = time.time()

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Forward time: {(t1-t0)*1000:.1f} ms")
    assert logits.shape == (B, 5, H, W), \
        f"Expected (B,5,H,W) but got {logits.shape}"
    print("✓ Output shape correct\n")

    # --- Loss compatibility ---
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, mask)
    loss.backward()
    print(f"CE Loss: {loss.item():.4f}")
    print("✓ Loss backward pass OK\n")

    # --- Bottleneck extraction ---
    feats = model.extract_bottleneck(x)
    print(f"Bottleneck feature: {tuple(feats.shape)}")
    assert feats.shape == (B, model.bottleneck_dim), \
        f"Expected (B, {model.bottleneck_dim}) but got {feats.shape}"
    print("✓ Bottleneck extraction OK\n")

    # --- Parameter separation ---
    enc_state = model.shared_encoder_state()
    dec_state = model.private_decoder_state()
    print(f"Encoder state keys: {len(enc_state)}")
    print(f"Decoder+head state keys: {len(dec_state)}")
    # Verify no overlap
    overlap = set(enc_state.keys()) & set(dec_state.keys())
    assert len(overlap) == 0, f"Parameter overlap: {overlap}"
    print("✓ No parameter overlap between encoder and decoder\n")

    # --- BaseHeadSplit compatibility ---
    print("Testing BaseHeadSplit compatibility...")
    import copy
    head = copy.deepcopy(model.fc)
    model.fc = nn.Identity()
    base_out = model(x)  # should return decoded features, not logits
    model.fc = head       # restore
    print(f"  Base output (fc=Identity): {tuple(base_out.shape)}")
    final = head(base_out)
    if final.shape[-2:] != (H, W):
        final = F.interpolate(final, size=(H, W), mode="bilinear",
                              align_corners=False)
    assert final.shape == (B, 5, H, W)
    print(f"  Head output: {tuple(final.shape)}")
    print("✓ BaseHeadSplit compatible\n")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
