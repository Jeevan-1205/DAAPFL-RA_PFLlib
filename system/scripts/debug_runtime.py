import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import argparse
import copy

from utils.config import load_yaml_config
from utils.segmentation_metrics import segmentation_confusion_matrix, compute_f1_dam
from flcore.trainmodel.siamese_unet import SiameseUNet
from flcore.clients.clientsegmentation import clientSegmentation
from flcore.servers.serversegmentation import ServerSegmentation
from losses.builder import build_loss

class DebugRuntime:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # Override to ensure we just run a very small test
        self.args.global_rounds = 1
        self.args.local_epochs = 1
        self.args.num_clients = 1
        self.args.batch_size = 4
        self.args.model = SiameseUNet(
            num_classes=self.args.num_classes,
            backbone="resnet50",
            pretrained=True,
        ).to(self.device)
        
        print(f"--- Stage 1 & 2: Initializing Debug Runtime ---")
        
        self.client = clientSegmentation(
            args=self.args, 
            id=0, 
            train_samples=100, 
            test_samples=100,
            train_slow=False,
            send_slow=False
        )
        
        self.model = self.client.model
        self.optimizer = self.client.optimizer
        self.loss_fn = self.client.loss
        self.trainloader = self.client.load_train_data(batch_size=self.args.batch_size)
        
        # Hooks state
        self.logits = None
        self.softmax_probs = None
        
    def hook_forward(self, module, input, output):
        self.logits = output.clone().detach()
        self.softmax_probs = F.softmax(self.logits, dim=1)
        
    def run_investigation(self):
        # Register hooks
        hook_handle = self.model.fc.register_forward_hook(self.hook_forward)
        
        self.model.train()
        
        print("\n--- Starting Training Loop Investigation ---")
        
        for batch_idx, (x, y) in enumerate(self.trainloader):
            if batch_idx >= 5: # Only run 5 batches for quick evidence gathering
                break
                
            x = x.to(self.device)
            y = y.to(self.device)
            
            # --- Stage 15: Runtime Assertions ---
            assert not torch.isnan(x).any(), "NaN in inputs!"
            assert not torch.isinf(x).any(), "Inf in inputs!"
            assert not torch.isnan(y).any(), "NaN in labels!"
            assert y.min() >= 0 and y.max() < self.args.num_classes, f"Labels out of range! Min: {y.min()}, Max: {y.max()}"
            
            print(f"\n[Batch {batch_idx+1}] Label Distribution:")
            unique, counts = torch.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"  Class {u.item()}: {c.item()} pixels")
                
            # Forward pass
            output = self.model(x)
            
            # --- Stage 3 & 8: Model Output & Logit Analysis ---
            assert not torch.isnan(output).any(), "NaN in output logits!"
            assert not torch.isinf(output).any(), "Inf in output logits!"
            
            print(f"\n[Batch {batch_idx+1}] Logit Statistics (Shape: {output.shape}, dtype: {output.dtype}):")
            for c in range(self.args.num_classes):
                c_logits = output[:, c, :, :]
                print(f"  Class {c}: Mean {c_logits.mean().item():.4f}, Min {c_logits.min().item():.4f}, Max {c_logits.max().item():.4f}, Std {c_logits.std().item():.4f}")
                
            # --- Stage 7: Softmax Analysis ---
            print(f"\n[Batch {batch_idx+1}] Softmax Probabilities:")
            probs = F.softmax(output, dim=1)
            for c in range(self.args.num_classes):
                c_probs = probs[:, c, :, :]
                print(f"  Class {c}: Mean {c_probs.mean().item():.4f}, Min {c_probs.min().item():.4f}, Max {c_probs.max().item():.4f}")
                
            # --- Stage 11: Loss Breakdown ---
            # Manually compute Dice and Focal to see contribution
            focal_loss = self.loss_fn.focal(output, y)
            dice_loss = self.loss_fn.dice(output, y)
            hybrid_loss = self.loss_fn.fw * focal_loss + self.loss_fn.dw * dice_loss
            
            print(f"\n[Batch {batch_idx+1}] Loss Breakdown:")
            print(f"  Focal Loss: {focal_loss.item():.6f} (Weight {self.loss_fn.fw})")
            print(f"  Dice Loss:  {dice_loss.item():.6f} (Weight {self.loss_fn.dw})")
            print(f"  Total Loss: {hybrid_loss.item():.6f}")
            
            # Backward
            self.optimizer.zero_grad()
            hybrid_loss.backward()
            
            # --- Stage 10 & 9: Gradient Investigation ---
            print(f"\n[Batch {batch_idx+1}] Gradient Norms:")
            
            def get_norm(module):
                norms = []
                for p in module.parameters():
                    if p.grad is not None:
                        norms.append(p.grad.norm(2).item()**2)
                return sum(norms)**0.5
                
            print(f"  Encoder: {get_norm(self.model.encoder):.6f}")
            print(f"  Decoder: {get_norm(self.model.decoder):.6f}")
            print(f"  Seg Head: {get_norm(self.model.fc):.6f}")
            
            # Check Seg Head gradients per class output
            if self.model.fc.conv.weight.grad is not None:
                grad_w = self.model.fc.conv.weight.grad
                print("  Seg Head Weight Gradient Norms per channel:")
                for c in range(self.args.num_classes):
                    print(f"    Class {c}: {grad_w[c].norm(2).item():.6f}")
                    
            self.optimizer.step()
            
            # --- Stage 4 & 5 & 14: Confusion Matrix & Metrics ---
            pred = torch.argmax(output, dim=1)
            cm = segmentation_confusion_matrix(pred, y, self.args.num_classes)
            print(f"\n[Batch {batch_idx+1}] Confusion Matrix:")
            print(cm.cpu().numpy())
            
            print("\n[Batch {batch_idx+1}] Class Prediction Distribution (Pixels):")
            pred_unique, pred_counts = torch.unique(pred, return_counts=True)
            for u, c in zip(pred_unique, pred_counts):
                print(f"  Class {u.item()}: {c.item()} pixels")
                
        hook_handle.remove()
        print("\n--- Investigation Complete ---")

if __name__ == "__main__":
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
    parser.add_argument('-lbs', "--batch_size", type=int, default=4)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005)
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-gr', "--global_rounds", type=int, default=1)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument("--lr_schedule", type=str, default="none")
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
    
    args = parser.parse_args()
    default_args = parser.parse_args([])
    
    # Load YAML
    args = load_yaml_config(args, default_args, parser)
    
    # Run
    investigator = DebugRuntime(args)
    investigator.run_investigation()
