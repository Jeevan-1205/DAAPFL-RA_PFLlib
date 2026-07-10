#!/usr/bin/env python3
"""
Analyze xBD client distributions.

Reports:
- number of tiles per client
- background-only tiles
- damage tiles
- class pixel counts
- tiles containing each class
- per-client CSV

Run:
python scripts/analyze_clients.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.data_utils import read_client_data

NUM_CLIENTS = 7
NUM_CLASSES = 5
DATASET = "xBD"

OUTPUT_DIR = "outputs/dataset_analysis"


def analyse_client(cid):

    ds = read_client_data(DATASET, cid, is_train=True)

    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    tile_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    background_only = 0
    damage_tiles = 0

    for _, mask in tqdm(ds,
                        desc=f"Client {cid}",
                        leave=False):

        mask = mask.numpy()

        hist = np.bincount(mask.reshape(-1), minlength=NUM_CLASSES)

        pixel_counts += hist
        tile_counts += (hist > 0)

        damage_pixels = hist[2:].sum()

        if damage_pixels == 0:
            background_only += 1
        else:
            damage_tiles += 1

    return {
        "client": cid,
        "tiles": len(ds),
        "background_only": background_only,
        "damage_tiles": damage_tiles,
        "damage_percent": 100.0 * damage_tiles / len(ds),

        "pixels_c0": pixel_counts[0],
        "pixels_c1": pixel_counts[1],
        "pixels_c2": pixel_counts[2],
        "pixels_c3": pixel_counts[3],
        "pixels_c4": pixel_counts[4],

        "tiles_c1": tile_counts[1],
        "tiles_c2": tile_counts[2],
        "tiles_c3": tile_counts[3],
        "tiles_c4": tile_counts[4],
    }


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []

    print("\nAnalyzing client distributions...\n")

    for cid in range(NUM_CLIENTS):
        rows.append(analyse_client(cid))

    df = pd.DataFrame(rows)

    csv_path = Path(OUTPUT_DIR) / "client_statistics.csv"

    df.to_csv(csv_path, index=False)

    print("\n")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    print("\nSaved:", csv_path)


if __name__ == "__main__":
    main()