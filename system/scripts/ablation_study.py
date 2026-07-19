import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import os

from utils.config import load_yaml_config
from utils.segmentation_metrics import segmentation_metrics
from utils.data_utils import read_client_data
from flcore.clients.clientsegmentation import clientSegmentation
from flcore.trainmodel.siamese_unet import SiameseUNet
from losses.dice import DiceLoss
from losses.focal import FocalLoss

class DiceLossNoMask(DiceLoss):
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        valid = target != self.ignore_index
        tgt = target.clone()
        tgt[~valid] = 0
        onehot = F.one_hot(tgt, num_classes).permute(0,3,1,2).float()
        mask = valid.unsqueeze(1).float()
        probs = probs * mask
        onehot = onehot * mask
        dims = (0,2,3)
        intersection = (probs * onehot).sum(dims)
        denominator = (probs.sum(dims) + onehot.sum(dims))
        dice = (2*intersection + self.smooth) / (denominator + self.smooth)
        
        # Disable present-class masking entirely. We just exclude background.
        if not self.include_background:
            dice = dice[1:]
        
        if dice.numel() == 0:
            return logits.new_tensor(0.)
        return 1 - dice.mean()

class AblationLoss(nn.Module):
    def __init__(self, mode, alpha):
        super().__init__()
        self.mode = mode
        self.focal = FocalLoss(alpha=torch.tensor(alpha), gamma=2.0)
        self.dice_masked = DiceLoss(include_background=False)
        self.dice_unmasked = DiceLossNoMask(include_background=False)
        
    def forward(self, logits, target):
        if self.mode == 'A':
            return 0.3 * self.focal(logits, target) + 0.7 * self.dice_masked(logits, target)
        elif self.mode == 'B':
            return 1.0 * self.dice_masked(logits, target)
        elif self.mode == 'C':
            return 1.0 * self.focal(logits, target)
        elif self.mode == 'D':
            return 0.3 * self.focal(logits, target) + 0.7 * self.dice_unmasked(logits, target)

def run_ablation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    parser.add_argument('-dev', "--device", type=str, default="cuda")
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-algo', "--algorithm", type=str, default="Local")
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
    parser.add_argument('-nc', "--num_clients", type=int, default=1)
    
    args, _ = parser.parse_known_args()
    default_args = parser.parse_args([])
    args = load_yaml_config(args, default_args, parser)
    
    # Preload client 0 dataset
    train_data = read_client_data("xBD", 0, is_train=True, few_shot=0)
    test_data = read_client_data("xBD", 0, is_train=False, few_shot=0)
    
    modes = ['A', 'B', 'C', 'D']
    all_metrics = []
    
    os.makedirs("ablation_results", exist_ok=True)
    
    for mode in modes:
        print(f"\n{'='*50}\nStarting Experiment {mode}\n{'='*50}")
        
        # Ensure identical initialization
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.manual_seed_all(42)
        
        args.model = SiameseUNet(num_classes=args.num_classes, backbone="resnet50", pretrained=True).to(args.device)
        client = clientSegmentation(args, 0, len(train_data), len(test_data), train_slow=False, send_slow=False)
        
        # Replace the loss function with the specific ablation variant
        client.loss = AblationLoss(mode, args.class_weights)
        
        for epoch in range(1, 6):
            print(f"Training Epoch {epoch}...")
            args.local_epochs = 1
            
            # Setup forward/backward hooks to monitor logit drift and gradient norms
            logits_stats = []
            grad_norms = []
            
            def hook_forward(module, input, output):
                stats = []
                for c in range(args.num_classes):
                    c_logits = output[:, c, :, :]
                    stats.append(c_logits.mean().item())
                logits_stats.append(stats)
            
            # Hook the last layer to get logits
            hook_handle = client.model.fc.register_forward_hook(hook_forward)
            
            client.train()
            hook_handle.remove()
            
            # Evaluate after epoch
            confusion, _ = client.test_metrics()
            metrics = segmentation_metrics(confusion)
            
            gt = confusion.sum(dim=1)
            pred = confusion.sum(dim=0)
            
            # Precision and Recall
            tp = torch.diag(confusion)
            fp = pred - tp
            fn = gt - tp
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            
            epoch_data = {
                'Mode': mode,
                'Epoch': epoch,
                'Minor_Dice': metrics['dice_per_class'][2],
                'Major_Dice': metrics['dice_per_class'][3],
                'Destroyed_Dice': metrics['dice_per_class'][4],
                'Minor_Recall': recall[2].item(),
                'Major_Recall': recall[3].item(),
                'Destroyed_Recall': recall[4].item(),
                'Minor_Precision': precision[2].item(),
                'Major_Precision': precision[3].item(),
                'Destroyed_Precision': precision[4].item(),
                'Minor_GT': gt[2].item(),
                'Major_GT': gt[3].item(),
                'Destroyed_GT': gt[4].item(),
                'Minor_Pred': pred[2].item(),
                'Major_Pred': pred[3].item(),
                'Destroyed_Pred': pred[4].item(),
                'Logits_Minor': np.mean([s[2] for s in logits_stats]),
                'Logits_Major': np.mean([s[3] for s in logits_stats]),
                'Logits_Destroyed': np.mean([s[4] for s in logits_stats])
            }
            
            all_metrics.append(epoch_data)
            
            print(f"Epoch {epoch} Results:")
            print(f"Minor     GT: {gt[2].item()} | Pred: {pred[2].item()} | Dice: {metrics['dice_per_class'][2]:.4f}")
            print(f"Major     GT: {gt[3].item()} | Pred: {pred[3].item()} | Dice: {metrics['dice_per_class'][3]:.4f}")
            print(f"Destroyed GT: {gt[4].item()} | Pred: {pred[4].item()} | Dice: {metrics['dice_per_class'][4]:.4f}")
            print(f"Minor Mean Logit: {epoch_data['Logits_Minor']:.4f}")

    df = pd.DataFrame(all_metrics)
    df.to_csv("ablation_results/metrics.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for mode in modes:
        data = df[df['Mode'] == mode]
        plt.plot(data['Epoch'], data['Minor_Dice'], marker='o', label=f'Exp {mode}')
    plt.title('Minor Damage Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    for mode in modes:
        data = df[df['Mode'] == mode]
        plt.plot(data['Epoch'], data['Major_Dice'], marker='o', label=f'Exp {mode}')
    plt.title('Major Damage Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    for mode in modes:
        data = df[df['Mode'] == mode]
        plt.plot(data['Epoch'], data['Destroyed_Dice'], marker='o', label=f'Exp {mode}')
    plt.title('Destroyed Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("ablation_results/dice_curves.png")

if __name__ == "__main__":
    run_ablation()
