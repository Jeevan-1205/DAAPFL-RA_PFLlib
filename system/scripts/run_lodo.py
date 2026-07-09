import subprocess
import sys

NUM_FOLDS = 7
CONFIG = "configs/fedavg_xbd.yaml"

CLIENTS = {
    0: ("Earthquake", ["Mexico"]),
    1: ("Flooding", ["Midwest", "Nepal"]),
    2: ("Hurricane", ["Florence", "Harvey", "Matthew", "Michael"]),
    3: ("Tornado", ["Moore", "Tuscaloosa", "Joplin"]),
    4: ("Tsunami", ["Sunda", "Palu"]),
    5: ("Volcano", ["Lower Puna", "Guatemala"]),
    6: ("Wildfire", ["SoCal", "Portugal", "Woolsey", "Pinery", "Santa Rosa"]),
}

for fold in range(NUM_FOLDS):

    dtype, events = CLIENTS[fold]

    print("\n" + "=" * 80)
    print(f"LODO Fold {fold + 1}/{NUM_FOLDS}")
    print(f"Held-out disaster type : {dtype}")
    print("Held-out events:")
    for e in events:
        print(f"   • {e}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "system/main.py",
        "--config",
        CONFIG,
        "--held_out_idx",
        str(fold),
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Fold {fold} failed.")
        sys.exit(1)

    print(f"\n✅ Fold {fold} completed successfully.")

print("\n🎉 All 7 LODO folds completed.")