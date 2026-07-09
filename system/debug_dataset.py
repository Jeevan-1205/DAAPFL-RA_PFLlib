import torch
from torch.utils.data import ConcatDataset

from utils.data_utils import read_client_data

NUM_CLIENTS = 7
DATASET = "xBD"
NUM_CLASSES = 5

# Build pooled dataset exactly like train_centralized.py
datasets = [
    read_client_data(DATASET, cid, is_train=True)
    for cid in range(NUM_CLIENTS)
]

train_ds = ConcatDataset(datasets)

pixel_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
tile_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
classes_present = set()

for _, mask in train_ds:
    unique = torch.unique(mask)

    classes_present.update(unique.tolist())

    for c in unique:
        tile_counts[c] += 1

    pixel_counts += torch.bincount(mask.view(-1), minlength=NUM_CLASSES)

print("\nClasses present:", sorted(classes_present))

print("\nPixel counts:")
for i, c in enumerate(pixel_counts):
    print(f"Class {i}: {c.item():,}")

print("\nPixel percentages:")
total = pixel_counts.sum().item()
for i, c in enumerate(pixel_counts):
    print(f"Class {i}: {100*c.item()/total:.4f}%")

print("\nTiles containing each class:")
for i, c in enumerate(tile_counts):
    print(f"Class {i}: {c.item()}")