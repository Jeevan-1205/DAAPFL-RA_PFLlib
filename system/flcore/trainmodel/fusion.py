"""Feature fusion modules for Siamese change detection models."""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn


class AbsoluteDifferenceFusion(nn.Module):
    """Fuse pre/post multi-scale features via ``post - pre``."""

    def forward(
        self,
        feats_pre: List[torch.Tensor],
        feats_post: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        self._validate_features(feats_pre, feats_post)
        return [torch.abs(post - pre) for pre, post in zip(feats_pre, feats_post)]

    @staticmethod
    def _validate_features(
        feats_pre: List[torch.Tensor],
        feats_post: List[torch.Tensor],
    ) -> None:
        if len(feats_pre) != len(feats_post):
            raise ValueError(
                "pre/post feature lists must have the same length, "
                f"got {len(feats_pre)} and {len(feats_post)}."
            )
        if len(feats_pre) == 0:
            raise ValueError("feature lists must not be empty.")

        for idx, (pre, post) in enumerate(zip(feats_pre, feats_post)):
            if not isinstance(pre, torch.Tensor) or not isinstance(post, torch.Tensor):
                raise TypeError(f"feature pair {idx} must contain tensors.")
            if pre.shape != post.shape:
                raise ValueError(
                    f"feature pair {idx} must have matching shapes, "
                    f"got {tuple(pre.shape)} and {tuple(post.shape)}."
                )
