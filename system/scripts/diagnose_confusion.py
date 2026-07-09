"""Diagnose which classes a trained model confuses with each other.

Loads a saved model (.pt state_dict, e.g. results/centralized/<run_name>_model.pt
or a *_checkpoint.pt) and prints the full confusion matrix, plus row-normalized
(recall) and column-normalized (precision) breakdowns -- so you can see, e.g.,
whether true class 3 pixels are being predicted as class 2, class 4, or class 0.

Usage (run from system/):
    python scripts/diagnose_confusion.py \
        --model_path results/centralized/Centralized_xBD_gr200_..._model.pt \
        --config config.yaml

Works with a *_checkpoint.pt too (it'll pull out the model weights automatically).
"""
import argparse
import sys
from pathlib import Path

# scripts/diagnose_confusion.py lives in system/scripts/, but flcore and utils
# live directly under system/. Running `python scripts/diagnose_confusion.py`
# puts system/scripts/ on sys.path, not system/ -- so without this, imports
# below fail with ModuleNotFoundError: No module named 'flcore'.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from flcore.trainmodel.siamese_unet import SiameseUNet
from utils.data_utils import read_client_data
from utils.segmentation_metrics import segmentation_confusion_matrix


def build_pooled_dataset(dataset_name, num_clients, is_train, few_shot=0):
    per_client = []
    for cid in range(num_clients):
        ds = read_client_data(dataset_name, cid, is_train=is_train, few_shot=few_shot)
        per_client.append(ds)
    return ConcatDataset(per_client)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True,
                     help="Path to a *_model.pt or *_checkpoint.pt file")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    dataset = cfg.get("dataset", "xBD")
    num_clients = cfg.get("num_clients", 7)
    num_classes = cfg.get("num_classes", 5)
    batch_size = cfg.get("batch_size", 8)

    device = args.device if torch.cuda.is_available() else "cpu"

    model = SiameseUNet(num_classes=num_classes, backbone="resnet50", pretrained=False).to(device)

    loaded = torch.load(args.model_path, map_location=device)
    # Handle both a plain state_dict (*_model.pt) and a checkpoint dict (*_checkpoint.pt)
    state_dict = loaded.get("model_state", loaded) if isinstance(loaded, dict) and "model_state" in loaded else loaded
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded weights from: {args.model_path}")
    print(f"Evaluating on {dataset} test split, {num_clients} clients pooled ...")

    test_ds = build_pooled_dataset(dataset, num_clients, is_train=False)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
    )

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(model(x), dim=1)
            confusion += segmentation_confusion_matrix(pred, y, num_classes)

    cm = confusion.numpy()

    print("\nRaw confusion matrix (rows = true class, cols = predicted class):")
    header = "        " + "".join(f"pred={c:<10}" for c in range(num_classes))
    print(header)
    for c in range(num_classes):
        print(f"true={c}: " + "".join(f"{cm[c][p]:<15,}" for p in range(num_classes)))

    print("\nRow-normalized (recall) -- of all TRUE pixels of this class, where did the model predict them:")
    row_sums = cm.sum(axis=1, keepdims=True)
    row_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    for c in range(num_classes):
        breakdown = ", ".join(f"pred={p}: {row_norm[c][p] * 100:.2f}%" for p in range(num_classes))
        print(f"  True class {c} ({int(row_sums[c][0]):,} pixels): {breakdown}")

    print("\nColumn-normalized (precision) -- of all pixels PREDICTED as this class, what were they actually:")
    col_sums = cm.sum(axis=0, keepdims=True)
    col_norm = np.divide(cm, col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums != 0)
    for c in range(num_classes):
        breakdown = ", ".join(f"true={t}: {col_norm[t][c] * 100:.2f}%" for t in range(num_classes))
        print(f"  Pred class {c} ({int(col_sums[0][c]):,} pixels): {breakdown}")

    print("\nInterpretation tip: if a rare class's row shows most of its mass landing")
    print("in a neighboring class's column (not class 0), that's confusion between")
    print("adjacent damage severities, not pure class-imbalance drowning. If its mass")
    print("lands almost entirely in class 0's column, the model is still just ignoring")
    print("it in favor of background, which is more of a weighting problem.")


if __name__ == "__main__":
    main()