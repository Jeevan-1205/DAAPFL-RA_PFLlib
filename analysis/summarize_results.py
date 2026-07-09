import os
import h5py
import numpy as np
import pandas as pd

RESULT_DIR = "results/FedAvg_xBD_LODO"

rows = []

for fold in range(7):

    path = os.path.join(RESULT_DIR, f"fold{fold}.h5")

    with h5py.File(path, "r") as f:

        rows.append({

            "Fold": fold,

            "Pixel Accuracy":
                float(f["rs_test_pixel_acc"][-1]),

            "Dice":
                float(f["rs_test_dice"][-1]),

            "mIoU":
                float(f["rs_test_miou"][-1]),
        })

df = pd.DataFrame(rows)

mean = df.mean(numeric_only=True)
std = df.std(numeric_only=True)

print(df)

print("\nMean")
print(mean)

print("\nStd")
print(std)

summary = pd.DataFrame([
    mean,
    std
], index=["Mean", "Std"])

summary.to_csv(
    os.path.join(RESULT_DIR, "summary.csv")
)