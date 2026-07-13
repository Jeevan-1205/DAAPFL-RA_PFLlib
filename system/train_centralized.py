"""Centralized (non-federated) baseline training for xBD segmentation.

Pools every client's cached tiles into one dataset and trains a single
SiameseUNet model normally -- no client sampling, no aggregation.
Uses the same cache (XBDCachedDataset via utils.data_utils) that the
FL clients already use, so a run of this script is directly comparable
to a FedAvg/FedProx run on the same data.

Usage (run from /PFLlib/system) -- same shared config.yaml as the FL runs:
    python train_centralized.py --config config.yaml

CLI flags still work and override whatever is in config.yaml, e.g.:
    python train_centralized.py --config config.yaml --epochs 10

Resume from a checkpoint:
    python train_centralized.py --config config.yaml \\
        --resume results/centralized/checkpoints/epoch_100.pt
"""
import argparse
import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from utils.data_utils import read_client_data
from utils.segmentation_metrics import segmentation_confusion_matrix, segmentation_metrics
from utils.csv_logger import CSVLogger, make_run_id
from flcore.trainmodel.siamese_unet import SiameseUNet

from utils.visualization import save_prediction_visualization
from torch.utils.data import WeightedRandomSampler
# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Maps this script's arg name -> key name used in the shared config.yaml
YAML_KEY_MAP = {
    "dataset": "dataset",
    "num_clients": "num_clients",
    "num_classes": "num_classes",
    "batch_size": "batch_size",
    "epochs": "global_rounds",       # shared config calls this global_rounds
    "lr": "local_learning_rate",     # shared config calls this local_learning_rate
    "output_dir": "output_dir",
}

DEFAULTS = {
    "dataset": "xBD",
    "num_clients": 7,
    "num_classes": 5,
    "batch_size": 8,
    "epochs": 50,
    "lr": 0.005,
    "output_dir": "centralized",
}

NUM_WORKERS = 8
BANNER_WIDTH = 58
DISASTER_NAMES = [
    "Earthquake",
    "Flood",
    "Hurricane",
    "Tornado",
    "Tsunami",
    "Volcano",
    "Wildfire",
]

# ──────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Convert seconds to a human-readable HH:MM:SS or MM:SS string."""
    if seconds < 0:
        return "--:--"
    td = timedelta(seconds=int(seconds))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def print_banner(dataset_name: str) -> None:
    """Print a prominent banner at the start of training."""
    border = "═" * BANNER_WIDTH
    title = f"  Centralized Training — {dataset_name} Segmentation"
    print(f"\n{border}")
    print(title)
    print(f"{border}\n")


def print_section(title: str) -> None:
    """Print a section header with bordered lines."""
    line = "─" * BANNER_WIDTH
    print(f"\n{line}")
    print(f"  {title}")
    print(line)


def print_dataset_info(train_ds, test_ds, train_loader, test_loader, args) -> None:
    """Print the Dataset info section."""
    print_section("Dataset")
    print(f"  Train tiles      : {len(train_ds):,}")
    print(f"  Validation tiles : {len(test_ds):,}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Workers          : {NUM_WORKERS}")


def print_model_info(model) -> None:
    """Print the Model info section."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_section("Model")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")


def print_training_config(args, device, run_name, result_path) -> None:
    """Print the Training Configuration section."""
    print_section("Training Configuration")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Learning rate  : {args.lr:.6f}")
    print(f"  Momentum       : {args.momentum}")
    print(f"  Eval gap       : {args.eval_gap}")
    print(f"  Checkpoint gap : {args.checkpoint_gap}")
    print(f"  Loss balance   : dice={args.dice_weight}, focal={args.focal_weight}")
    print(f"  LR schedule    : {args.lr_schedule}")
    if args.lr_schedule == "step":
        print(f"    step_size={args.lr_step_size}, gamma={args.lr_gamma}")
    elif args.lr_schedule == "cosine_warm_restarts":
        print(f"    T_0={args.lr_t0}, T_mult={args.lr_tmult}")
    elif args.lr_schedule == "plateau":
        print(f"    metric={args.lr_plateau_metric}, patience={args.lr_plateau_patience}, factor={args.lr_plateau_factor}")
    print(f"  Device         : {device}")
    print(f"  Output         : {result_path}/")
    print(f"  Run name       : {run_name}")
    if args.class_weights:
        weights_str = "[" + ", ".join(f"{w:.4f}" for w in args.class_weights) + "]"
        print(f"  Class weights  : {weights_str}")
    else:
        print("  Class weights  : None (no per-class weighting)")


def print_resume_info(checkpoint_path: str, start_epoch: int, best_miou: float,
                      best_pixel_acc: float, best_dice: float, best_f1_dam: float = 0.0) -> None:
    """Print resume information when loading from a checkpoint."""
    print_section("Resuming from Checkpoint")
    print(f"  Checkpoint     : {checkpoint_path}")
    print(f"  Resume epoch   : {start_epoch}")
    print(f"  Best mIoU      : {best_miou:.4f}")
    print(f"  Best Pixel Acc : {best_pixel_acc:.4f}")
    print(f"  Best Dice      : {best_dice:.4f}")
    print(f"  Best F1-dam    : {best_f1_dam:.4f}")


def print_epoch_summary(epoch: int, total_epochs: int, avg_loss: float,
                        metrics: dict | None, best_miou: float,
                        epoch_time: float, eta_seconds: float) -> None:
    """Print an aligned per-epoch summary block."""
    print_section(f"Epoch {epoch + 1}/{total_epochs}")
    print(f"  Train Loss   : {avg_loss:.4f}")

    if metrics is not None:
        pixel_acc = metrics["pixel_accuracy"]
        dice = metrics["dice"]
        miou = metrics["mean_iou"]
        f1_dam = metrics.get("f1_dam")
        print(f"  Pixel Acc    : {pixel_acc:.4f}")
        print(f"  Dice         : {dice:.4f}")
        print(f"  mIoU         : {miou:.4f}")
        if f1_dam is not None:
            print(f"  F1-dam       : {f1_dam:.4f}")
        star = " ★" if miou >= best_miou else ""
        print(f"  Best mIoU    : {best_miou:.4f}{star}")
    else:
        print(f"  Best mIoU    : {best_miou:.4f}")

    print(f"  Epoch Time   : {format_time(epoch_time)}")
    print(f"  ETA          : {format_time(eta_seconds)}")


# ──────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler,
                    best_miou: float, best_pixel_acc: float, best_dice: float,
                    args, best_f1_dam: float = 0.0) -> None:
    """Save a full training checkpoint via torch.save().

    Parameters
    ----------
    path : str
        Destination file path (e.g. ``checkpoints/epoch_010.pt``).
    epoch : int
        Current epoch number (0-indexed).
    model : nn.Module
        The model to save.
    optimizer : torch.optim.Optimizer
        Optimizer whose state dict is saved.
    scheduler : optional
        Learning rate scheduler (may be ``None``).
    best_miou, best_pixel_acc, best_dice, best_f1_dam : float
        Best metrics observed so far.
    args : argparse.Namespace
        Training configuration.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_miou": best_miou,
        "best_pixel_acc": best_pixel_acc,
        "best_dice": best_dice,
        "best_f1_dam": best_f1_dam,
        "config": {
            "dataset": args.dataset,
            "num_clients": args.num_clients,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "eval_gap": args.eval_gap,
            "class_weights": args.class_weights,
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str, model, optimizer, scheduler, device: str,
                     current_lr: float = None, current_momentum: float = None) -> dict:
    """Load a checkpoint and restore model/optimizer/scheduler state.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.
    model : nn.Module
        Model to restore weights into.
    optimizer : torch.optim.Optimizer
        Optimizer to restore state into.
    scheduler : optional
        Learning rate scheduler to restore (may be ``None``).
    device : str
        Device to map tensors to (``"cuda"`` or ``"cpu"``).
    current_lr, current_momentum : float, optional
        If given, force these values onto every optimizer param_group AFTER
        loading. This matters because optimizer.load_state_dict() restores
        the FULL param_groups dict from the checkpoint -- including lr and
        momentum -- not just the per-parameter state buffers. Without this,
        resuming with a different --lr or --momentum than the original run
        silently reverts to whatever was saved in the checkpoint.

    Returns
    -------
    dict
        The full checkpoint dictionary so the caller can extract
        ``epoch``, ``best_miou``, etc.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if current_lr is not None or current_momentum is not None:
        for group in optimizer.param_groups:
            if current_lr is not None:
                group["lr"] = current_lr
            if current_momentum is not None:
                group["momentum"] = current_momentum
        print(f"  Overriding resumed optimizer: lr={current_lr}, momentum={current_momentum} "
              f"(otherwise these would silently revert to the checkpoint's original values)")

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


# ──────────────────────────────────────────────────────────────────────
# Model / dataset builders
# ──────────────────────────────────────────────────────────────────────

def build_model(num_classes, device):
    model = SiameseUNet(
        num_classes=num_classes,
        backbone="resnet50",
        pretrained=True,
    ).to(device)
    return model


def build_pooled_dataset(
    dataset_name,
    num_clients,
    is_train,
    few_shot=0,
    held_out_idx=None,
):
    """
    Build a pooled dataset.

    held_out_idx=None:
        Original behavior (all clients pooled)

    held_out_idx=k:
        LODO
        - Train: all clients except k
        - Test : only client k
    """

    per_client = []

    # Original centralized behaviour
    if held_out_idx is None:
        for cid in range(num_clients):
            ds = read_client_data(
                dataset_name,
                cid,
                is_train=is_train,
                few_shot=few_shot,
            )
            per_client.append(ds)

    # LODO behaviour
    else:

        if is_train:

            print(f"LODO Train: excluding client {held_out_idx}")

            for cid in range(num_clients):

                if cid == held_out_idx:
                    continue

                ds = read_client_data(
                    dataset_name,
                    cid,
                    is_train=True,
                    few_shot=few_shot,
                )

                per_client.append(ds)

        else:

            print(f"LODO Test: client {held_out_idx}")

            ds = read_client_data(
                dataset_name,
                held_out_idx,
                is_train=False,
                few_shot=few_shot,
            )

            per_client.append(ds)

    return ConcatDataset(per_client)

# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

def compute_f1_dam(confusion, damage_classes, eps=1e-8):
    """F1-dam: harmonic mean of per-class F1 across the damage severity
    classes (i.e. every class except background/no-damage), following the
    xBD/xView2 challenge convention for damage-classification F1. Harmonic
    mean (rather than a plain average) means F1-dam is dragged down hard by
    any single damage class the model is failing on -- it can't be
    inflated by one strong class hiding a dead one.

    Computed directly from the confusion matrix (rows=true, cols=pred),
    independent of whatever utils.segmentation_metrics.segmentation_metrics
    returns, so it doesn't depend on that module's internals.
    """
    cm = confusion.float()
    per_class_f1 = []
    for c in damage_classes:
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        per_class_f1.append(f1.item())

    # harmonic mean, guarding against any exact-zero class killing the whole
    # metric via division by zero (clamped to eps instead)
    safe_f1s = [max(f, eps) for f in per_class_f1]
    f1_dam = len(safe_f1s) / sum(1.0 / f for f in safe_f1s)
    return f1_dam, per_class_f1


def evaluate(model, loader, num_classes, device,
             save_predictions=False,
             prediction_dir=None):

    model.eval()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    saved = 0                      # <-- ADD THIS

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            output = model(x)
            

            pred = torch.argmax(output, dim=1)

            # ===========================================
            # SAVE VISUALIZATIONS HERE
            # ===========================================
            if save_predictions and saved < 10:

                batch_size = x.size(0)

                for i in range(batch_size):

                    if saved >= 10:
                        break

                    pre = x[i, :3]
                    post = x[i, 3:]

                    save_prediction_visualization(
                        pre,
                        post,
                        y[i],
                        pred[i],
                        prediction_dir,
                        saved,
                    )

                    saved += 1
            # ===========================================

            confusion += segmentation_confusion_matrix(
                pred,
                y,
                num_classes,
            )

    metrics = segmentation_metrics(confusion)



    damage_classes = tuple(range(1, num_classes))  # every class except 0 (background)
    f1_dam, f1_dam_per_class = compute_f1_dam(confusion, damage_classes)
    metrics["f1_dam"] = f1_dam
    metrics["f1_dam_per_class"] = f1_dam_per_class

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Argument resolution
# ──────────────────────────────────────────────────────────────────────

def resolve_args():
    ap = argparse.ArgumentParser(
        description="Centralized (non-federated) baseline training for xBD segmentation.",
    )
    ap.add_argument("--config", type=str, default=None,
                     help="Path to shared config.yaml (same file used for FL runs)")

    # All of these default to None here so we can tell whether the user
    # explicitly passed them on the CLI. Real defaults are applied below,
    # in priority order: CLI flag > config.yaml > DEFAULTS.
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--num_clients", type=int, default=None,
                     help="How many client shards to pool together")
    ap.add_argument(
        "--held_out_idx",
        type=int,
        default=None,
        help="LODO: hold out this client/disaster for testing. "
            "Default=None pools all clients (original centralized behaviour)."
    )
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval_gap", type=int, default=1,
                     help="Evaluate every N epochs")
    ap.add_argument("--device", type=str, default="cuda",
                     choices=["cuda", "cpu"])
    ap.add_argument("--output_dir", type=str, default=None,
                     help="Results land in results/<output_dir>/, same convention as FL runs")
    ap.add_argument("--class_weights", type=str, default=None,
                     help="Comma-separated per-class alpha weights, e.g. "
                          "'0.21,1.85,3.40,4.95,4.80'. Overrides config.yaml's "
                          "class_weights if given.")
    # New arguments
    ap.add_argument("--resume", type=str, default=None,
                     help="Path to a .pt checkpoint to resume training from")
    ap.add_argument("--checkpoint_gap", type=int, default=10,
                     help="Save a checkpoint every N epochs (default: 10)")
    ap.add_argument("--dice_weight", type=float, default=None,
                     help="Weight of the dice term in the hybrid loss (default 0.5, "
                          "or config.yaml's dice_weight if set)")
    ap.add_argument("--focal_weight", type=float, default=None,
                     help="Weight of the focal term in the hybrid loss (default 0.5, "
                          "or config.yaml's focal_weight if set)")
    ap.add_argument("--lr_schedule", type=str, default=None,
                     choices=["none", "cosine", "step", "cosine_warm_restarts", "plateau"],
                     help="Learning rate schedule. 'cosine' decays lr to ~0 over --epochs. "
                          "'step' drops lr by --lr_gamma every --lr_step_size epochs. "
                          "'cosine_warm_restarts' (SGDR) cosine-decays then periodically "
                          "resets lr back up every --lr_t0 epochs (x--lr_tmult each cycle) "
                          "-- useful if plateaus are local optima rather than genuine "
                          "convergence. 'plateau' (ReduceLROnPlateau) only decays lr when "
                          "--lr_plateau_metric stops improving for --lr_plateau_patience "
                          "evaluated epochs, reacting to the actual training curve instead "
                          "of a fixed schedule.")
    ap.add_argument("--lr_step_size", type=int, default=None,
                     help="Epochs between decays for --lr_schedule step (default 30)")
    ap.add_argument("--lr_gamma", type=float, default=None,
                     help="Decay factor for --lr_schedule step (default 0.1)")
    ap.add_argument("--lr_t0", type=int, default=None,
                     help="Epochs until first restart for --lr_schedule cosine_warm_restarts (default 20)")
    ap.add_argument("--lr_tmult", type=int, default=None,
                     help="Multiplier on restart period after each restart, cosine_warm_restarts (default 2)")
    ap.add_argument("--lr_plateau_patience", type=int, default=None,
                     help="Evaluated epochs with no improvement before decaying, --lr_schedule plateau (default 10)")
    ap.add_argument("--lr_plateau_factor", type=float, default=None,
                     help="Decay factor for --lr_schedule plateau (default 0.5)")
    ap.add_argument("--lr_plateau_metric", type=str, default=None,
                     choices=["miou", "dice", "f1_dam"],
                     help="Metric ReduceLROnPlateau monitors (default: miou, matching "
                          "the best_miou improvement tracker used elsewhere)")
    ap.add_argument("--momentum", type=float, default=None,
                     help="SGD momentum (default 0.9, or config.yaml's momentum if set). "
                          "Plain SGD defaults to momentum=0, which makes it easy for a rare "
                          "output channel's bias to get pushed very negative early on and "
                          "never recover -- momentum helps the optimizer escape that.")
    args = ap.parse_args()
    if args.held_out_idx is None:
        print("\n" + "=" * 60)
        print("Evaluation Protocol : Centralized (All Disaster Types)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Evaluation Protocol : Leave-One-Disaster-Type-Out (LODO)")
        print(f"Held-out Disaster : {DISASTER_NAMES[args.held_out_idx]}")
        print("=" * 60)

    yaml_config = {}
    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f) or {}

    for arg_key, yaml_key in YAML_KEY_MAP.items():
        cli_val = getattr(args, arg_key)
        if cli_val is not None:
            continue  # explicit CLI flag always wins
        yaml_val = yaml_config.get(yaml_key)
        setattr(args, arg_key, yaml_val if yaml_val is not None else DEFAULTS[arg_key])

    if args.dice_weight is None:
        args.dice_weight = yaml_config.get("dice_weight", 0.5)
    if args.focal_weight is None:
        args.focal_weight = yaml_config.get("focal_weight", 0.5)
    if args.lr_schedule is None:
        args.lr_schedule = yaml_config.get("lr_schedule", "none")
    if args.lr_step_size is None:
        args.lr_step_size = yaml_config.get("lr_step_size", 30)
    if args.lr_gamma is None:
        args.lr_gamma = yaml_config.get("lr_gamma", 0.1)
    if args.lr_t0 is None:
        args.lr_t0 = yaml_config.get("lr_t0", 20)
    if args.lr_tmult is None:
        args.lr_tmult = yaml_config.get("lr_tmult", 2)
    if args.lr_plateau_patience is None:
        args.lr_plateau_patience = yaml_config.get("lr_plateau_patience", 10)
    if args.lr_plateau_factor is None:
        args.lr_plateau_factor = yaml_config.get("lr_plateau_factor", 0.5)
    if args.lr_plateau_metric is None:
        args.lr_plateau_metric = yaml_config.get("lr_plateau_metric", "miou")
    if args.momentum is None:
        args.momentum = yaml_config.get("momentum", 0.9)

    if args.class_weights is not None:
        # explicit CLI override, comma-separated string -> list of floats
        args.class_weights = [float(x) for x in args.class_weights.split(",")]
    else:
        # falls back to config.yaml's class_weights (may be None/null, which
        # is fine -- build_loss just gets alpha=None, same as before)
        args.class_weights = yaml_config.get("class_weights")

    return args


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Train for one epoch and return the average loss.

    Uses a tqdm progress bar for per-batch feedback.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    current_lr = optimizer.param_groups[0]["lr"]

    batch_bar = tqdm(
        loader,
        desc="  Train",
        leave=False,
        dynamic_ncols=True,
        bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for batch_idx, (x, y) in enumerate(batch_bar):
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1
        avg_loss = running_loss / n_batches

        batch_bar.set_postfix_str(
            f"loss={loss.item():.4f} | avg={avg_loss:.4f} | lr={current_lr:.6f}"
        )

    batch_bar.close()
    return running_loss / max(n_batches, 1)


def main():
    args = resolve_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    run_id = make_run_id()

    if args.held_out_idx is None:
        run_name = f"Centralized_{args.dataset}_ALL_gr{args.epochs}_{run_id}"
    else:
        disaster = DISASTER_NAMES[args.held_out_idx].replace(" ", "_")
        run_name = (
            f"Centralized_LODO_{disaster}_"
            f"gr{args.epochs}_{run_id}"
        )

        # Automatically create a separate folder for each LODO fold
        args.output_dir = os.path.join(
            "centralized_lodo",
            f"fold{args.held_out_idx}_{disaster.lower()}"
        )

    result_path = os.path.join("results", args.output_dir)
    checkpoint_dir = os.path.join(result_path, "checkpoints")

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────
    print_banner(args.dataset)

    # ── Dataset ───────────────────────────────────────────────────────
    train_ds = build_pooled_dataset(
        args.dataset,
        args.num_clients,
        is_train=True,
        held_out_idx=args.held_out_idx,

    )

    print(f"Original training tiles: {len(train_ds)}")
    test_ds = build_pooled_dataset(args.dataset, args.num_clients, is_train=False,  held_out_idx=args.held_out_idx,)
    print("Building weighted sampler...")

    weights = []
    
    damage_tiles = 0

    background_tiles = 0
    damage_weight = getattr(args, "damage_sampling_weight", 5.0)

    for client_ds in train_ds.datasets:
        for i in range(len(client_ds)):
            if client_ds.has_damage(i):
                weights.append(damage_weight)
                damage_tiles += 1
            else:
                weights.append(1.0)
                background_tiles += 1
    print(f"Damage tiles     : {damage_tiles}")
    print(f"Background tiles : {background_tiles}")

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True,
    )

    print_dataset_info(train_ds, test_ds, train_loader, test_loader, args)

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(args.num_classes, device)
    print_model_info(model)

    # ── Loss & Optimizer ──────────────────────────────────────────────
    from losses import build_loss

    loss_fn = build_loss(
        name="dice_focal",
        num_classes=args.num_classes,
        alpha=args.class_weights,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # ── Resume (part 1: model + optimizer) ──────────────────────────────
    # Done BEFORE the scheduler is constructed so we know the real
    # start_epoch first -- otherwise a cosine schedule restarts its decay
    # curve from epoch 0 on every resume instead of picking up where it
    # left off, under-annealing the LR by the true final epoch.
    start_epoch = 0
    best_miou = 0.0
    best_pixel_acc = 0.0
    best_dice = 0.0
    best_f1_dam = 0.0
    checkpoint = None

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, scheduler=None, device=device,
            current_lr=args.lr, current_momentum=args.momentum,
        )
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint.get("best_miou", 0.0)
        best_pixel_acc = checkpoint.get("best_pixel_acc", 0.0)
        best_dice = checkpoint.get("best_dice", 0.0)
        best_f1_dam = checkpoint.get("best_f1_dam", 0.0)  # 0.0 for old checkpoints saved before this metric existed

    # ── LR Scheduler ─────────────────────────────────────────────────
    # last_epoch=start_epoch-1 syncs a fresh curve to the real epoch count
    # on resume, for every schedule type that supports last_epoch. If the
    # checkpoint has its own saved scheduler state, that's authoritative
    # and overrides this guess -- except for 'plateau', which has no
    # last_epoch concept at all (see note below).
    scheduler = None
    is_plateau_scheduler = False
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, last_epoch=start_epoch - 1,
        )
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma,
            last_epoch=start_epoch - 1,
        )
    elif args.lr_schedule == "cosine_warm_restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.lr_t0, T_mult=args.lr_tmult,
            last_epoch=start_epoch - 1,
        )
    elif args.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.lr_plateau_factor,
            patience=args.lr_plateau_patience,
        )
        is_plateau_scheduler = True
        # ReduceLROnPlateau has no last_epoch / step-count concept -- its
        # state is just "how many evaluated epochs since last improvement".
        # On resume without saved scheduler state, it starts monitoring for
        # a fresh plateau from here rather than replaying history. That's a
        # real (if minor) behavior gap vs the other schedules -- flagged in
        # print_resume_info below rather than silently accepted.

    if scheduler is not None and checkpoint is not None and checkpoint.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            # Most likely args.lr_schedule differs from what the checkpoint
            # was trained with (e.g. resuming a 'cosine' run as 'plateau').
            # Keep the freshly constructed scheduler rather than crashing.
            print(f"  WARNING: could not load saved scheduler state (schedule type "
                  f"may have changed since this checkpoint was saved) -- "
                  f"starting '{args.lr_schedule}' fresh from epoch {start_epoch}.")

    if args.resume:
        print_resume_info(args.resume, start_epoch, best_miou, best_pixel_acc, best_dice, best_f1_dam)
        if args.lr_schedule in ("cosine", "step", "cosine_warm_restarts"):
            print(f"  {args.lr_schedule} schedule synced to epoch {start_epoch}/{args.epochs} "
                  f"(lr={optimizer.param_groups[0]['lr']:.6f})")
        elif args.lr_schedule == "plateau":
            print(f"  plateau schedule resumed -- monitoring '{args.lr_plateau_metric}' "
                  f"fresh from epoch {start_epoch} (no plateau history carried over)")

    # ── Training Config ───────────────────────────────────────────────
    print_training_config(args, device, run_name, result_path)

    # ── CSV Logger Setup ──────────────────────────────────────────────
    class_fieldnames = []
    for c in range(args.num_classes):
        class_fieldnames.append(f"dice_class_{c}")
    for c in range(args.num_classes):
        class_fieldnames.append(f"iou_class_{c}")
    for c in range(1, args.num_classes):
        class_fieldnames.append(f"f1_dam_class_{c}")

    epoch_csv_path = os.path.join(result_path, f"{run_name}_epochs.csv")
    epoch_logger = CSVLogger(
        epoch_csv_path,
        fieldnames=[
            "epoch", "evaluated", "train_loss",
            "pixel_acc", "dice", "miou", "f1_dam",
        ] + class_fieldnames + [
            "epoch_time_sec",
        ],
    )

    # ── Training Loop ─────────────────────────────────────────────────
    print_section("Training")
    print()  # blank line before progress bars

    epoch_times = []
    epochs_completed = start_epoch
    training_start = time.time()

    epoch_bar = tqdm(
        range(start_epoch, args.epochs),
        desc="Epochs",
        initial=start_epoch,
        total=args.epochs,
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for epoch in epoch_bar:
        epoch_start = time.time()

        # ── Train one epoch ───────────────────────────────────────────
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        if scheduler is not None and not is_plateau_scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # ── Evaluate ──────────────────────────────────────────────────
        epoch_metrics = None
        row = {
            "epoch": epoch,
            "evaluated": 0,
            "train_loss": avg_loss,
            "epoch_time_sec": epoch_time,
        }

        if epoch % args.eval_gap == 0:

            save_preds = (
                epoch == 0 or
                (epoch + 1) % 5 == 0
            )

            prediction_dir = os.path.join(
                result_path,
                "predictions",
                f"epoch_{epoch+1:03d}",
            )

            epoch_metrics = evaluate(
                model,
                test_loader,
                args.num_classes,
                device,
                save_predictions=save_preds,
                prediction_dir=prediction_dir,
            )
            pixel_acc = epoch_metrics["pixel_accuracy"]
            dice = epoch_metrics["dice"]
            miou = epoch_metrics["mean_iou"]
            dice_per_class = epoch_metrics["dice_per_class"]
            iou_per_class = epoch_metrics["iou"]
            f1_dam = epoch_metrics["f1_dam"]
            f1_dam_per_class = epoch_metrics["f1_dam_per_class"]

            if scheduler is not None and is_plateau_scheduler:
                monitor_value = {"miou": miou, "dice": dice, "f1_dam": f1_dam}[args.lr_plateau_metric]
                scheduler.step(monitor_value)

            improved = miou > best_miou
            best_miou = max(best_miou, miou)
            best_pixel_acc = max(best_pixel_acc, pixel_acc)
            best_dice = max(best_dice, dice)
            best_f1_dam = max(best_f1_dam, f1_dam)

            row.update({
                "evaluated": 1,
                "pixel_acc": pixel_acc,
                "dice": dice,
                "miou": miou,
                "f1_dam": f1_dam,
            })
            for c in range(args.num_classes):
                row[f"dice_class_{c}"] = float(dice_per_class[c])
                row[f"iou_class_{c}"] = float(iou_per_class[c])
            for i, c in enumerate(range(1, args.num_classes)):
                row[f"f1_dam_class_{c}"] = float(f1_dam_per_class[i])

            # ── Save best model ───────────────────────────────────────
            if improved:
                best_model_path = os.path.join(result_path, "best_model.pt")
                save_checkpoint(
                    best_model_path, epoch, model, optimizer, scheduler,
                    best_miou, best_pixel_acc, best_dice, args, best_f1_dam,
                )

        # ── CSV logging ────────────────────────────────────────────────
        epoch_logger.log(row)

        # ── Save last model ───────────────────────────────────────────
        last_model_path = os.path.join(result_path, "last_model.pt")
        save_checkpoint(
            last_model_path, epoch, model, optimizer, scheduler,
            best_miou, best_pixel_acc, best_dice, args, best_f1_dam,
        )

        # ── Periodic checkpoint ───────────────────────────────────────
        # Save on checkpoint_gap boundaries (1-indexed epochs for naming)
        epoch_1indexed = epoch + 1
        if epoch_1indexed % args.checkpoint_gap == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch_1indexed:03d}.pt")
            save_checkpoint(
                ckpt_path, epoch, model, optimizer, scheduler,
                best_miou, best_pixel_acc, best_dice, args, best_f1_dam,
            )

        # ── ETA computation ───────────────────────────────────────────
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = args.epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs

        # ── Update outer progress bar ─────────────────────────────────
        epoch_bar.set_postfix_str(f"best_mIoU={best_miou:.4f}")

        # ── Epoch summary ─────────────────────────────────────────────
        print_epoch_summary(
            epoch, args.epochs, avg_loss,
            epoch_metrics, best_miou,
            epoch_time, eta_seconds,
        )

        epochs_completed = epoch + 1

    epoch_bar.close()

    # ── Final model save (backward compat) ────────────────────────────
    print_section("Training Complete")
    print(f"  Best mIoU  : {best_miou:.4f}")
    print(f"  Best F1-dam: {best_f1_dam:.4f}")
    print(f"  Total time : {format_time(time.time() - training_start)}")

    model_path = os.path.join(result_path, f"{run_name}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Saved final weights   : {model_path}")
    print(f"  Best model checkpoint : {os.path.join(result_path, 'best_model.pt')}")
    print(f"  Last model checkpoint : {os.path.join(result_path, 'last_model.pt')}")
    print(f"  Periodic checkpoints  : {checkpoint_dir}/")
    print(f"  Epoch metrics CSV     : {epoch_csv_path}")

    # --------------------------------------------------
    # Append a row to the same shared summary.csv used by FL runs,
    # so centralized and FedAvg/FedProx results sit side by side.
    # --------------------------------------------------
    summary_path = os.path.join(result_path, "summary.csv")
    summary_logger = CSVLogger(
        summary_path,
        fieldnames=[
            "run_name", "run_id", "timestamp", "dataset", "algorithm",
            "protocol", "held_out_idx", "num_clients", "num_classes",
            "global_rounds_configured", "global_rounds_completed",
            "best_pixel_acc", "best_dice", "best_miou", "best_f1_dam",
            "avg_round_time_sec", "total_time_sec",
        ],
    )
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    summary_logger.log({
        "run_name": run_name,
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "algorithm": "Centralized",
        "protocol": "centralized",
        "held_out_idx": "",
        "num_clients": args.num_clients,
        "num_classes": args.num_classes,
        "global_rounds_configured": args.epochs,
        "global_rounds_completed": epochs_completed,
        "best_pixel_acc": best_pixel_acc,
        "best_dice": best_dice,
        "best_miou": best_miou,
        "best_f1_dam": best_f1_dam,
        "avg_round_time_sec": avg_epoch_time,
        "total_time_sec": sum(epoch_times),
    })
    print(f"  Summary appended to   : {summary_path}")


if __name__ == "__main__":
    main()
