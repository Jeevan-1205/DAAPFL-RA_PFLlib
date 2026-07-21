import csv
import os

base_dir = "/DATA/BU_Internship_Jeevan/Research/PFLlib/system/results"
files = {
    'Centralized': 'centralized/Centralized_xBD_gr200_20260713_145119_epochs.csv',
    'Local': 'Local_xBD_standard_gr20_20260718_160008_rounds.csv',
    'FedAvg': 'FedAvg_xBD_standard_gr200_20260716_004914_rounds.csv',
    'FedProx': 'FedProx_xBD_standard_gr50_20260716_192859_rounds.csv',
    'FedPer': 'FedPer_xBD_standard_gr20_20260718_172621_rounds.csv',
    'FedRep': 'FedRep_xBD_standard_gr50_20260719_110303_rounds.csv',
    'Ditto': 'Ditto_xBD_standard_gr50_20260719_141409_rounds.csv',
    'pFedMe': 'pFedMe_xBD_standard_gr50_20260719_195743_rounds.csv',
    'SCAFFOLD': 'SCAFFOLD_xBD_standard_gr2_20260719_173927_rounds.csv',
    'FedLC': 'FedLC_xBD_standard_gr1_20260719_210440_rounds.csv'
}

for algo, rel_path in files.items():
    path = os.path.join(base_dir, rel_path)
    if not os.path.exists(path):
        print(f"{algo}: File not found ({rel_path})")
        continue
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if len(rows) == 0:
            print(f"{algo}: Empty file")
            continue
            
        print(f"\n{'='*50}\n{algo}\n{'='*50}")
        print(f"Total rounds/epochs completed: {len(rows)}")
        
        # Centralized has different col names
        if algo == 'Centralized':
            best_row = max(rows, key=lambda x: float(x.get('f1_dam', 0) if x.get('f1_dam') else 0))
            print("BEST METRICS:")
            print(f"Epoch: {best_row['epoch']}")
            print(f"Pixel Acc: {best_row['pixel_acc']}")
            print(f"Dice: {best_row['dice']}")
            print(f"mIoU: {best_row['miou']}")
            print(f"F1-dam: {best_row['f1_dam']}")
            print(f"Per-class Dice: BG={best_row['dice_class_0']}, ND={best_row['dice_class_1']}, Min={best_row['dice_class_2']}, Maj={best_row['dice_class_3']}, Des={best_row['dice_class_4']}")
            print(f"Per-class IoU: BG={best_row['iou_class_0']}, ND={best_row['iou_class_1']}, Min={best_row['iou_class_2']}, Maj={best_row['iou_class_3']}, Des={best_row['iou_class_4']}")
        else:
            best_row = max(rows, key=lambda x: float(x.get('f1_dam', 0) if x.get('f1_dam') else 0))
            print("BEST METRICS:")
            print(f"Round: {best_row['round']}")
            print(f"Pixel Acc: {best_row['pixel_acc']}")
            print(f"Dice: {best_row['dice']}")
            print(f"mIoU: {best_row['miou']}")
            print(f"F1-dam: {best_row.get('f1_dam', 'N/A')}")
            print(f"Per-class Dice: BG={best_row.get('dice_bg', 'N/A')}, ND={best_row.get('dice_no_damage', 'N/A')}, Min={best_row.get('dice_minor', 'N/A')}, Maj={best_row.get('dice_major', 'N/A')}, Des={best_row.get('dice_destroyed', 'N/A')}")
            print(f"Per-class IoU: BG={best_row.get('iou_bg', 'N/A')}, ND={best_row.get('iou_no_damage', 'N/A')}, Min={best_row.get('iou_minor', 'N/A')}, Maj={best_row.get('iou_major', 'N/A')}, Des={best_row.get('iou_destroyed', 'N/A')}")
            
        print("\nLEARNING TREND (F1-dam over rounds):")
        # Print first few, middle, and last few to see convergence
        if len(rows) > 10:
            trend = [r.get('f1_dam', 0) for r in rows]
            print(f"Start: {trend[:3]}")
            print(f"Mid:   {trend[len(trend)//2-1:len(trend)//2+2]}")
            print(f"End:   {trend[-3:]}")
        else:
            print([r.get('f1_dam', 0) for r in rows])
