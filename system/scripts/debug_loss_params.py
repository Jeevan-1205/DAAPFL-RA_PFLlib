import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import argparse

from utils.config import load_yaml_config
from flcore.clients.clientsegmentation import clientSegmentation
from losses.builder import build_loss

class CrossEntropyTracer:
    def __init__(self):
        self.original_ce = F.cross_entropy
        self.captured_kwargs = None
        
    def hook(self, input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
        # We hook cross_entropy to capture its exact parameters at runtime
        self.captured_kwargs = {
            'weight': weight,
            'ignore_index': ignore_index,
            'reduction': reduction,
            'label_smoothing': label_smoothing
        }
        return self.original_ce(input, target, weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce, reduction=reduction, label_smoothing=label_smoothing)

def run_trace():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    
    # Base args to ensure config loading works
    parser.add_argument('-dev', "--device", type=str, default="cpu")
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-ncl', "--num_classes", type=int, default=5)
    parser.add_argument("--class_weights", type=float, nargs="+", default=[0.0787, 0.4698, 1.3944, 1.3535, 1.7037])
    parser.add_argument("--dice_weight", type=float, default=0.7)
    parser.add_argument("--focal_weight", type=float, default=0.3)
    parser.add_argument('-data', "--dataset", type=str, default="xBD")
    parser.add_argument('-lbs', "--batch_size", type=int, default=2)
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
    
    args, _ = parser.parse_known_args()
    default_args = parser.parse_args([])
    
    # Load YAML
    args = load_yaml_config(args, default_args, parser)
    
    print(f"=== YAML CONFIG VALUES ===")
    print(f"  class_weights: {args.class_weights}")
    print(f"  dice_weight: {args.dice_weight}")
    print(f"  focal_weight: {args.focal_weight}")
    print(f"==========================\n")
    
    print(f"=== INITIALIZING FedAvg CLIENT ===")
    from flcore.trainmodel.siamese_unet import SiameseUNet
    args.model = SiameseUNet(num_classes=args.num_classes, backbone="resnet50", pretrained=False).to(args.device)
    
    client = clientSegmentation(args, 0, 10, 10, train_slow=False, send_slow=False)
    loss_fn = client.loss
    print(f"Client Loss Type: {type(loss_fn).__name__}")
    print(f"  fw (Focal Weight): {loss_fn.fw}")
    print(f"  dw (Dice Weight): {loss_fn.dw}")
    print(f"  Focal alpha: {loss_fn.focal.alpha}")
    print(f"  Focal ignore_index: {loss_fn.focal.ignore_index}")
    print(f"  Dice include_background: {loss_fn.dice.include_background}")
    print(f"  Dice ignore_index: {loss_fn.dice.ignore_index}")
    
    print(f"\n=== TRACING FIRST ITERATION ===")
    tracer = CrossEntropyTracer()
    F.cross_entropy = tracer.hook
    
    # Dummy data
    logits = torch.randn(2, 5, 128, 128)
    target = torch.randint(0, 5, (2, 128, 128))
    
    loss = loss_fn(logits, target)
    
    print(f"F.cross_entropy() called with:")
    for k, v in tracer.captured_kwargs.items():
        print(f"  {k}: {v}")
    
    # Restore
    F.cross_entropy = tracer.original_ce

if __name__ == "__main__":
    run_trace()
