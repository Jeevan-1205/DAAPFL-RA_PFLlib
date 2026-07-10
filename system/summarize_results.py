"""Turn hp_search.py's results/hp_search/trials.csv into a paper-ready summary:
a Markdown table of per-lr_schedule best/mean performance, the overall winning
config, and a suggested Methods-section paragraph.

Usage:
    python summarize_results.py --trials_csv results/hp_search/trials.csv --metric f1_dam

Output:
    Printed to stdout, and written to <trials_csv's directory>/paper_summary.md
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials_csv", type=str, default="results/hp_search/trials.csv")
    ap.add_argument("--metric", type=str, default="f1_dam",
                     help="Must match the column name used in trials.csv (f1_dam, dice, or miou)")
    ap.add_argument("--method_label", type=str, default="TPE (Optuna)",
                     help="How to describe the search method in the generated paragraph. "
                          "Use 'Grid search' if trials.csv came from --method grid.")
    return ap.parse_args()


def load_trials(path, metric):
    rows = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "completed":
                continue
            try:
                row[metric] = float(row[metric])
            except (KeyError, ValueError):
                continue
            row["dice_weight"] = float(row["dice_weight"])
            row["focal_weight"] = float(row["focal_weight"])
            rows.append(row)
    return rows


def per_schedule_table(rows, metric):
    by_schedule = defaultdict(list)
    for r in rows:
        by_schedule[r["lr_schedule"]].append(r[metric])

    lines = [
        "| lr_schedule | n trials | best {0} | mean {0} | std {0} |".format(metric),
        "|---|---|---|---|---|",
    ]
    for schedule in sorted(by_schedule, key=lambda s: -max(by_schedule[s])):
        vals = by_schedule[schedule]
        best = max(vals)
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lines.append(f"| {schedule} | {len(vals)} | {best:.4f} | {mean:.4f} | {std:.4f} |")
    return "\n".join(lines)


def build_summary(rows, metric, method_label, n_total_including_incomplete):
    completed = len(rows)
    best = max(rows, key=lambda r: r[metric])

    schedule_table = per_schedule_table(rows, metric)

    paragraph = (
        f"Hyperparameters (loss-term mixing weight and learning-rate schedule) were tuned via "
        f"{method_label} over {n_total_including_incomplete} trials ({completed} completed) using "
        f"short proxy training runs, optimizing {metric.replace('_', '-')} on the held-out validation "
        f"split. The best configuration found was dice_weight={best['dice_weight']:.3f}, "
        f"focal_weight={best['focal_weight']:.3f}, lr_schedule={best['lr_schedule']} "
        f"({metric.replace('_', '-')}={best[metric]:.4f} in the proxy search; see Section X for the "
        f"full-length confirmation run)."
    )

    md = (
        f"## Hyperparameter search summary\n\n"
        f"{paragraph}\n\n"
        f"### Performance by learning-rate schedule\n\n"
        f"{schedule_table}\n\n"
        f"### Best configuration\n\n"
        f"| dice_weight | focal_weight | lr_schedule | best {metric} |\n"
        f"|---|---|---|---|\n"
        f"| {best['dice_weight']:.4f} | {best['focal_weight']:.4f} | {best['lr_schedule']} | {best[metric]:.4f} |\n"
    )
    return md


def main():
    args = parse_args()
    if not os.path.exists(args.trials_csv):
        print(f"ERROR: {args.trials_csv} not found. Run hp_search.py first.")
        return

    with open(args.trials_csv, "r", newline="") as f:
        n_total = sum(1 for _ in csv.DictReader(f))

    rows = load_trials(args.trials_csv, args.metric)
    if not rows:
        print(f"No completed trials with metric '{args.metric}' found in {args.trials_csv}.")
        return

    md = build_summary(rows, args.metric, args.method_label, n_total)
    print(md)

    out_path = os.path.join(os.path.dirname(args.trials_csv) or ".", "paper_summary.md")
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\n(also written to {out_path})")


if __name__ == "__main__":
    main()
