#!/usr/bin/env python3
"""
Analyze xBD cached training dataset.

Produces:
1. outputs/dataset_analysis/train_tile_statistics.csv
2. Prints dataset statistics.

Run:

python scripts/analyze_xbd_dataset.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data_utils import read_client_data
from torch.utils.data import ConcatDataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET = "xBD"
NUM_CLIENTS = 7
OUTPUT_DIR = "outputs/dataset_analysis"

NUM_CLASSES = 5


# ---------------------------------------------------------
# Build pooled training dataset
# ---------------------------------------------------------

def build_dataset():
    datasets = []

    for cid in range(NUM_CLIENTS):
        ds = read_client_data(
            DATASET,
            cid,
            is_train=True,
        )
        datasets.append(ds)

    return ConcatDataset(datasets)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = build_dataset()

    print("=" * 70)
    print("Analyzing xBD cached dataset")
    print("=" * 70)
    print(f"Total tiles : {len(dataset):,}")
    print()

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    tile_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    background_only = 0
    damage_tiles = 0

    histogram_bins = {
        "0": 0,
        "1-50": 0,
        "51-100": 0,
        "101-500": 0,
        "501-1000": 0,
        "1000+": 0,
    }

    rows = []

    for idx in tqdm(range(len(dataset))):

        _, mask = dataset[idx]

        mask = mask.numpy()

        hist = np.bincount(
            mask.reshape(-1),
            minlength=NUM_CLASSES,
        )

        pixel_counts += hist
        tile_counts += (hist > 0)

        building_pixels = hist[1:].sum()

        has_building = building_pixels > 0

        damage_pixels = hist[2:].sum()

        has_severe_damage = damage_pixels > 0

        if damage_pixels == 0:
            background_only += 1
        else:
            damage_tiles += 1

        if damage_pixels == 0:
            histogram_bins["0"] += 1
        elif damage_pixels <= 50:
            histogram_bins["1-50"] += 1
        elif damage_pixels <= 100:
            histogram_bins["51-100"] += 1
        elif damage_pixels <= 500:
            histogram_bins["101-500"] += 1
        elif damage_pixels <= 1000:
            histogram_bins["501-1000"] += 1
        else:
            histogram_bins["1000+"] += 1

        # dominant damage class
        dominant_damage = -1

        if damage_pixels > 0:

            damage_hist = hist[2:]

            dominant_damage = np.argmax(damage_hist) + 2

        rows.append(
            {
                "tile_id": idx,

                "class0_pixels": hist[0],
                "class1_pixels": hist[1],
                "class2_pixels": hist[2],
                "class3_pixels": hist[3],
                "class4_pixels": hist[4],

                "total_pixels": hist.sum(),
                "damage_pixels": damage_pixels,

                "has_damage": int(damage_pixels > 0),

                "contains_class1": int(hist[1] > 0),
                "contains_class2": int(hist[2] > 0),
                "contains_class3": int(hist[3] > 0),
                "contains_class4": int(hist[4] > 0),

                "dominant_damage": dominant_damage,
            }
        )

    df = pd.DataFrame(rows)

    csv_path = Path(OUTPUT_DIR) / "train_tile_statistics.csv"

    df.to_csv(csv_path, index=False)

    print()
    print("=" * 70)
    print("Dataset Summary")
    print("=" * 70)

    print(f"Total tiles           : {len(dataset):,}")
    print(f"Background-only tiles : {background_only:,}")
    print(f"Damage tiles          : {damage_tiles:,}")

    print()

    print("=" * 70)
    print("Pixel Counts")
    print("=" * 70)

    total_pixels = pixel_counts.sum()

    for c in range(NUM_CLASSES):

        pct = pixel_counts[c] / total_pixels * 100

        print(
            f"Class {c}: "
            f"{pixel_counts[c]:12,d} "
            f"({pct:6.3f}%)"
        )

    print()

    print("=" * 70)
    print("Tiles Containing Each Class")
    print("=" * 70)

    for c in range(NUM_CLASSES):

        pct = tile_counts[c] / len(dataset) * 100

        print(
            f"Class {c}: "
            f"{tile_counts[c]:6,d} "
            f"({pct:6.2f}%)"
        )

    print()

    print("=" * 70)
    print("Damage Pixel Histogram")
    print("=" * 70)

    for k, v in histogram_bins.items():

        pct = v / len(dataset) * 100

        print(
            f"{k:>8}: "
            f"{v:6,d} "
            f"({pct:6.2f}%)"
        )

    print()

    print("=" * 70)
    print(f"CSV saved to:\n{csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()