import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from utils.config import load_yaml_config
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader

disaster_names = ['Earthquake', 'Flood', 'Hurricane', 'Tornado', 'Tsunami', 'Volcano', 'Wildfire']
classes = ['Background', 'No Damage', 'Minor', 'Major', 'Destroyed']

def run_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../config.yaml")
    parser.add_argument('-dev', "--device", type=str, default="cpu")
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
    parser.add_argument('-nc', "--num_clients", type=int, default=7)
    
    args, _ = parser.parse_known_args()
    default_args = parser.parse_args([])
    args = load_yaml_config(args, default_args, parser)

    out_dir = "dataset_exposure"
    os.makedirs(out_dir, exist_ok=True)

    client_reports = []
    
    # Store global data for cross-client plots
    all_patch_pixels = {c: [] for c in range(5)}
    
    for cid in range(args.num_clients):
        dname = disaster_names[cid] if cid < len(disaster_names) else f"Client_{cid}"
        print(f"Analyzing {dname} (Client {cid})...")
        train_data = read_client_data("xBD", cid, is_train=True, few_shot=0)
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        
        c_stats = {
            'cid': cid,
            'name': dname,
            'total_patches': 0,
            'total_pixels': 0,
            'total_batches': 0,
            'class_pixels': np.zeros(5),
            'patch_counts': np.zeros(5),
            'patch_pixels': {i: [] for i in range(5)},
            
            'only_bg_patches': 0,
            'no_damage_classes_patches': 0,
            'at_least_one_damage_patches': 0,
            'multiple_damage_patches': 0,
            
            'positive_batches': np.zeros(5),
            'pixels_in_positive_batches': np.zeros(5)
        }
        
        for x, y in loader:
            c_stats['total_batches'] += 1
            batch_size = y.shape[0]
            
            # Batch level analysis
            for cls in range(5):
                cls_pixels_in_batch = (y == cls).sum().item()
                if cls_pixels_in_batch > 0:
                    c_stats['positive_batches'][cls] += 1
                    c_stats['pixels_in_positive_batches'][cls] += cls_pixels_in_batch
            
            # Patch level analysis
            for b in range(batch_size):
                patch = y[b]
                c_stats['total_patches'] += 1
                c_stats['total_pixels'] += patch.numel()
                
                unique_classes, counts = torch.unique(patch, return_counts=True)
                patch_cls_counts = {cls: 0 for cls in range(5)}
                for u, c in zip(unique_classes, counts):
                    patch_cls_counts[u.item()] = c.item()
                
                damage_present = sum(1 for c in range(2, 5) if patch_cls_counts[c] > 0)
                
                if patch_cls_counts[0] == patch.numel():
                    c_stats['only_bg_patches'] += 1
                if damage_present == 0:
                    c_stats['no_damage_classes_patches'] += 1
                if damage_present >= 1:
                    c_stats['at_least_one_damage_patches'] += 1
                if damage_present > 1:
                    c_stats['multiple_damage_patches'] += 1
                    
                for cls in range(5):
                    cnt = patch_cls_counts[cls]
                    c_stats['class_pixels'][cls] += cnt
                    c_stats['patch_pixels'][cls].append(cnt)
                    all_patch_pixels[cls].append(cnt)
                    if cnt > 0:
                        c_stats['patch_counts'][cls] += 1
                        
        
        # Aggregate Client Report
        report = {
            'Client': dname,
            'Num Patches': c_stats['total_patches'],
            'Num Batches': c_stats['total_batches'],
            'Total Pixels': c_stats['total_pixels'],
        }
        
        for cls in range(5):
            cls_name = classes[cls]
            pixels = c_stats['patch_pixels'][cls]
            pixels_pos = [p for p in pixels if p > 0]
            
            occupancy = {
                '0': len([p for p in pixels if p == 0]),
                '1-10': len([p for p in pixels if 1 <= p <= 10]),
                '11-100': len([p for p in pixels if 11 <= p <= 100]),
                '101-1000': len([p for p in pixels if 101 <= p <= 1000]),
                '>1000': len([p for p in pixels if p > 1000]),
            }
            
            report[cls_name] = {
                'Total Pixels': int(c_stats['class_pixels'][cls]),
                'Percent Pixels': (c_stats['class_pixels'][cls] / c_stats['total_pixels']) * 100,
                'Patches Containing': int(c_stats['patch_counts'][cls]),
                
                'Mean Px/Patch': float(np.mean(pixels)) if pixels else 0.0,
                'Median Px/Patch': float(np.median(pixels)) if pixels else 0.0,
                'Min Px/Patch': float(np.min(pixels)) if pixels else 0.0,
                'Max Px/Patch': float(np.max(pixels)) if pixels else 0.0,
                'Std Px/Patch': float(np.std(pixels)) if pixels else 0.0,
                
                'Mean Px/(Pos Patch)': float(np.mean(pixels_pos)) if pixels_pos else 0.0,
                
                'Occupancy': occupancy,
                
                'Positive Batches': int(c_stats['positive_batches'][cls]),
                'Percent Pos Batches': float((c_stats['positive_batches'][cls] / c_stats['total_batches']) * 100),
                'Avg Px/Pos Batch': float(c_stats['pixels_in_positive_batches'][cls] / c_stats['positive_batches'][cls]) if c_stats['positive_batches'][cls] > 0 else 0.0
            }
            
        report['Background Analysis'] = {
            'Only Background Patches': c_stats['only_bg_patches'],
            'No Damage Patches': c_stats['no_damage_classes_patches'],
            'At Least One Damage Patch': c_stats['at_least_one_damage_patches'],
            'Multiple Damage Patches': c_stats['multiple_damage_patches']
        }
        
        client_reports.append(report)

    # Save reports
    with open(os.path.join(out_dir, "exposure_report.json"), "w") as f:
        json.dump(client_reports, f, indent=4)
        
    # Generate Visualizations
    sns.set_theme(style="whitegrid")
    
    # 1. Positive Batches per Class per Client (Heatmap)
    data = []
    for r in client_reports:
        row = {'Client': r['Client']}
        for cls in ['Minor', 'Major', 'Destroyed']:
            row[cls] = r[cls]['Percent Pos Batches']
        data.append(row)
    df = pd.DataFrame(data).set_index('Client')
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlOrRd", fmt=".1f", cbar_kws={'label': '% Batches Containing Class'})
    plt.title('Percentage of Batches Containing Rare Damage Classes')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "batch_presence_heatmap.png"))
    plt.close()
    
    # 2. Cumulative Distribution of Damage Pixels
    plt.figure(figsize=(10, 6))
    for c_idx, cls in enumerate(['Minor', 'Major', 'Destroyed']):
        cls_idx = c_idx + 2
        pixels = all_patch_pixels[cls_idx]
        pos_pixels = [p for p in pixels if p > 0]
        if pos_pixels:
            sns.ecdfplot(pos_pixels, label=cls)
    plt.xscale('log')
    plt.xlabel('Pixels per Patch (Log Scale)')
    plt.ylabel('Cumulative Distribution')
    plt.title('CDF of Damage Pixels in Patches (Where Present)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "damage_pixels_cdf.png"))
    plt.close()
    
    # 3. Patch Occupancy for Damage Classes (Bar Chart)
    occ_data = []
    for cls_idx, cls in enumerate(['Minor', 'Major', 'Destroyed']):
        for r in client_reports:
            occ = r[cls]['Occupancy']
            for k, v in occ.items():
                if k != '0':
                    occ_data.append({'Client': r['Client'], 'Class': cls, 'Bin': k, 'Count': v})
    
    occ_df = pd.DataFrame(occ_data)
    g = sns.catplot(data=occ_df, kind="bar", x="Bin", y="Count", hue="Class", col="Client", col_wrap=4, height=3, aspect=1.2, order=['1-10', '11-100', '101-1000', '>1000'])
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Patch Occupancy: Number of Patches by Pixel Count Bin')
    plt.savefig(os.path.join(out_dir, "patch_occupancy.png"))
    plt.close()
    
    print("Done. Results saved to", out_dir)

if __name__ == "__main__":
    run_analysis()
