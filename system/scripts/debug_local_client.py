import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import argparse
import copy

from utils.config import load_yaml_config
from utils.segmentation_metrics import segmentation_metrics
from flcore.clients.clientsegmentation import clientSegmentation
from flcore.trainmodel.siamese_unet import SiameseUNet

def run_local_ablation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    
    # Base args to ensure config loading works
    parser.add_argument('-dev', "--device", type=str, default="cuda")
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-ncl', "--num_classes", type=int, default=5)
    parser.add_argument("--class_weights", type=float, nargs="+", default=[0.0787, 0.4698, 1.3944, 1.3535, 1.7037])
    parser.add_argument("--dice_weight", type=float, default=0.7)
    parser.add_argument("--focal_weight", type=float, default=0.3)
    parser.add_argument('-data', "--dataset", type=str, default="xBD")
    parser.add_argument('-lbs', "--batch_size", type=int, default=8)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-gr', "--global_rounds", type=int, default=1)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument("--lr_schedule", type=str, default="cosine_warm_restarts")
    parser.add_argument("--lr_t0", type=int, default=6)
    parser.add_argument("--lr_tmult", type=int, default=2)
    parser.add_argument("--lr_step_size", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--lr_plateau_patience", type=int, default=10)
    parser.add_argument("--lr_plateau_factor", type=float, default=0.5)
    parser.add_argument('-mo', "--momentum", type=float, default=0.9)
    parser.add_argument('-mu', "--mu", type=float, default=0.01)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default="test")
    parser.add_argument('-m', "--model", type=str, default="SiameseUNet")
    parser.add_argument('-nc', "--num_clients", type=int, default=1)
    
    args, _ = parser.parse_known_args()
    default_args = parser.parse_args([])
    args = load_yaml_config(args, default_args, parser)
    
    # We want to train for 5 epochs locally
    args.local_epochs = 5
    
    args.model = SiameseUNet(num_classes=args.num_classes, backbone="resnet50", pretrained=True).to(args.device)
    
    print(f"=== INITIALIZING CLIENT 0 (Earthquake) LOCAL TRAINING ===")
    from utils.data_utils import read_client_data
    train_data = read_client_data("xBD", 0, is_train=True, few_shot=0)
    test_data = read_client_data("xBD", 0, is_train=False, few_shot=0)
    
    client = clientSegmentation(args, 0, len(train_data), len(test_data), train_slow=False, send_slow=False)
    
    # Evaluate at start
    confusion, _ = client.test_metrics()
    metrics = segmentation_metrics(confusion)
    print("\n--- Epoch 0 (Init) ---")
    print(f"Minor F1: {metrics['f1_per_class'][1]:.4f}")
    print(f"Major F1: {metrics['f1_per_class'][2]:.4f}")
    
    # Train 5 epochs
    for epoch in range(5):
        print(f"\nTraining Epoch {epoch+1}...")
        
        # client.train() trains for local_epochs (which is 5), but wait, client.train() trains for args.local_epochs!
        # If I call it once with args.local_epochs = 1, it trains 1 epoch.
        args.local_epochs = 1
        client.train()
        
        confusion, _ = client.test_metrics()
        metrics = segmentation_metrics(confusion)
        print(f"--- Epoch {epoch+1} Evaluation ---")
        print(f"Minor F1: {metrics['f1_per_class'][1]:.4f} (Dice: {metrics['dice_per_class'][2]:.4f})")
        print(f"Major F1: {metrics['f1_per_class'][2]:.4f} (Dice: {metrics['dice_per_class'][3]:.4f})")
        print(f"Destroyed F1: {metrics['f1_per_class'][3]:.4f} (Dice: {metrics['dice_per_class'][4]:.4f})")
        
        # Check actual positive pixels in test set for this client
        gt = confusion.sum(dim=1)
        print(f"Ground Truth Pixels in Test Set: Minor={gt[2]}, Major={gt[3]}, Destroyed={gt[4]}")
        pred = confusion.sum(dim=0)
        print(f"Predicted Pixels in Test Set:    Minor={pred[2]}, Major={pred[3]}, Destroyed={pred[4]}")

if __name__ == "__main__":
    run_local_ablation()
