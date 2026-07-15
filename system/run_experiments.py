import argparse
import subprocess
import time
import os
import sys

DISASTERS = [
    "Earthquake",
    "Flood",
    "Hurricane",
    "Tornado",
    "Tsunami",
    "Volcano",
    "Wildfire",
]

def run_command(cmd):
    print("\nRunning:")
    print(" ".join(cmd))
    print("-" * 80)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n❌ Experiment failed.")
        sys.exit(result.returncode)


def main():

    parser = argparse.ArgumentParser(
        description="Run all LODO folds for a given algorithm."
    )

    parser.add_argument(
        "--algorithm",
        default="Centralized",
        choices=[
            "Centralized",
            "FedAvg",
            "FedProx",
            "SCAFFOLD",
            "FedPer",
            "FedRep",
            "Ditto",
            "FedALA",
            "pFedMe",
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    args = parser.parse_args()

    overall_start = time.time()

    print("=" * 80)
    print(f"Running {args.algorithm} LODO Experiments")
    print("=" * 80)

    for fold, disaster in enumerate(DISASTERS):

        print("\n" + "=" * 80)
        print(f"Fold {fold}/6")
        print(f"Held-out Disaster : {disaster}")
        print("=" * 80)

        summary_file = os.path.join(
            "results",
            "centralized_lodo",
            f"fold{fold}_{disaster.lower()}",
            "summary.csv",
        )

        if args.algorithm == "Centralized" and os.path.exists(summary_file):
            print(f"✓ Fold {fold} ({disaster}) already completed. Skipping.")
            continue

        start = time.time()

        if args.algorithm == "Centralized":

            cmd = [
                "python",
                "train_centralized.py",
                "--config",
                args.config,
                "--held_out_idx",
                str(fold),
                "--epochs",
                str(args.epochs),
            ]

        else:

            cmd = [
                "python",
                "main.py",
                "--config",
                args.config,
                "--algorithm",
                args.algorithm,
                "--dataset",
                "xBD",
                "--held_out_idx",
                str(fold),
            ]

        run_command(cmd)

        elapsed = time.time() - start

        print(f"\n✅ Fold completed in {elapsed/3600:.2f} hours")

    total = time.time() - overall_start

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"Total runtime: {total/3600:.2f} hours")


if __name__ == "__main__":
    main()