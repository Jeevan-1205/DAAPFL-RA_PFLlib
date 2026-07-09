"""Compute per-class pixel frequencies + suggested focal-loss alpha weights
from the cached xBD tiles.

Usage (run from system/):
    python scripts/compute_class_weights.py \
        --cache_root /DATA/BU_Internship_Jeevan/Research/PFLlib/dataset/xBD/tile_cache \
        --num_classes 5

Prints raw pixel counts and two candidate weighting schemes:
  - inverse frequency (normalized to mean 1.0)
  - effective number of samples (Cui et al. 2019), beta=0.9999 -- usually the
    safer default for severe imbalance since it doesn't blow up as hard as
    raw inverse frequency when a class has very few pixels.

Paste whichever `alpha` list you want directly into config.yaml under
`class_weights:`.
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True,
                     help="Path to tile_cache/ (or a train-only subfolder within it)")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--include_substr", type=str, default="train",
                     help="Only count .npz files whose path contains this substring "
                          "(use '' to include everything, e.g. train+test combined). "
                          "Weights should come from the TRAIN split, since that's "
                          "what the loss actually trains on.")
    args = ap.parse_args()

    pattern = os.path.join(args.cache_root, "**", "*.npz")
    files = glob.glob(pattern, recursive=True)

    if args.include_substr:
        matched = [f for f in files if args.include_substr in f]
        if matched:
            files = matched
        else:
            print(f"WARNING: no files matched include_substr={args.include_substr!r}; "
                  f"falling back to all {len(files)} .npz files found under cache_root. "
                  f"If your cache doesn't split train/test by path/filename, this is fine -- "
                  f"just double check you're not mixing in test tiles unintentionally.")

    if not files:
        print(f"No .npz files found under {args.cache_root}. Check the path.")
        return

    print(f"Scanning {len(files)} cached tiles ...")

    counts = np.zeros(args.num_classes, dtype=np.int64)

    for i, f in enumerate(files):
        data = np.load(f)
        mask = data["mask"]
        binc = np.bincount(mask.ravel(), minlength=args.num_classes)
        counts += binc[:args.num_classes]

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(files)} tiles scanned")

    total = counts.sum()
    freqs = counts / total

    print("\n================ CLASS PIXEL COUNTS ================")
    for c in range(args.num_classes):
        print(f"Class {c}: {counts[c]:,} pixels  ({freqs[c] * 100:.4f}%)")
    print("======================================================\n")

    # Scheme 1: inverse frequency, normalized to mean 1.0
    # Aggressive -- can push the majority class weight toward ~0, which
    # occasionally destabilizes training (model over-predicts rare classes).
    inv_freq = 1.0 / (freqs + 1e-12)
    inv_freq = inv_freq / inv_freq.mean()

    # Scheme 2: sqrt inverse frequency, normalized to mean 1.0
    # A standard softened version of inverse frequency -- meaningfully
    # up-weights rare classes without nearly zeroing out the majority
    # class. Good default for severe pixel-level imbalance.
    sqrt_inv_freq = 1.0 / np.sqrt(freqs + 1e-12)
    sqrt_inv_freq = sqrt_inv_freq / sqrt_inv_freq.mean()

    print("Suggested alpha (inverse frequency, mean-normalized):")
    print("  [" + ", ".join(f"{w:.4f}" for w in inv_freq) + "]")
    print("  (aggressive -- majority class weight can collapse toward 0)")

    print("\nSuggested alpha (sqrt inverse frequency, mean-normalized):")
    print("  [" + ", ".join(f"{w:.4f}" for w in sqrt_inv_freq) + "]")
    print("  (RECOMMENDED default -- more moderate, safer starting point)")

    print("\nPaste one of these lists into config.yaml as:")
    print("  class_weights: [w0, w1, w2, w3, w4]")
    print("\nNote: the 'effective number of samples' scheme (Cui et al. 2019) is")
    print("intentionally not offered here -- it saturates and becomes useless")
    print("once class pixel counts exceed roughly 1/(1-beta), which for typical")
    print("beta values (0.999-0.9999) is far smaller than xBD's per-class pixel")
    print("counts (even the rarest class here has over 1M pixels). It's designed")
    print("for small sample-count imbalance (e.g. hundreds to low-thousands of")
    print("examples), not pixel-level segmentation imbalance at this scale.")


if __name__ == "__main__":
    main()