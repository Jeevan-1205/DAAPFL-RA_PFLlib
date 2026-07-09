"""One-time prep: convert raw xBD tif/mask paths into resized, cached .npz files.
Run this BEFORE starting FL training. Do not run it every round.
Usage: python scripts/prepare_xbd_cache.py --dataset xbd --image_size 224
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tifffile
from PIL import Image

from utils.data_utils import DATASET_ROOT, _pil_resampling, read_data


def load_and_resize_tif(tif_path, image_size):
    image = tifffile.imread(tif_path).astype(np.float32)
    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min + 1e-6)
    image = (image * 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] > 3:
        image = image[:, :, :3]

    image = Image.fromarray(image).resize(
        (image_size, image_size), _pil_resampling("bilinear")
    )
    return (np.asarray(image, dtype=np.float32) / 255.0)


def load_and_resize_mask(mask_path, image_size):
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = Image.fromarray(mask).resize(
        (image_size, image_size), _pil_resampling("nearest")
    )
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.min() < 0 or mask.max() > 4:
        raise ValueError(f"xBD mask labels must be in [0, 4], got {mask_path}")
    return mask


def cache_path_for(pre_path, post_path, cache_root):
    # deterministic name from the pre/post filenames
    stem = Path(pre_path).stem + "__" + Path(post_path).stem
    return cache_root / f"{stem}.npz"


def build_cache(dataset, split, image_size, cache_root):
    cache_root.mkdir(parents=True, exist_ok=True)
    is_train = split == "train"

    # walk every client id file already present under train/ or test/
    data_dir = DATASET_ROOT / dataset / split
    client_files = sorted(data_dir.glob("*.npz"))

    n_written, n_skipped = 0, 0
    for cf in client_files:
        idx = cf.stem
        data = read_data(dataset, idx, is_train)
        image_paths = data['x']   # list of [pre_path, post_path]
        mask_paths = data['y']    # list of mask paths

        for (pre_path, post_path), mask_path in zip(image_paths, mask_paths):
            out_path = cache_path_for(pre_path, post_path, cache_root)
            if out_path.exists():
                n_skipped += 1
                continue

            pre = load_and_resize_tif(pre_path, image_size)
            post = load_and_resize_tif(post_path, image_size)
            mask = load_and_resize_mask(mask_path, image_size)
            stacked = np.concatenate([pre, post], axis=2)  # (H, W, 6)

            np.savez_compressed(out_path, image=stacked, mask=mask)
            n_written += 1

        print(f"[{split}] client {idx}: done "
              f"(written={n_written}, skipped_existing={n_skipped})")

    print(f"Finished {split}. wrote={n_written} skipped={n_skipped} cache={cache_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xbd")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--cache_root", default=None)
    args = ap.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else (
        DATASET_ROOT / args.dataset / "tile_cache"
    )

    build_cache(args.dataset, "train", args.image_size, cache_root)
    build_cache(args.dataset, "test", args.image_size, cache_root)