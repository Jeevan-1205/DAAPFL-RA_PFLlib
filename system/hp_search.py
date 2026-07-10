"""Hyperparameter search for train_centralized.py.

Replaces hand-picked values for dice_weight / focal_weight / lr_schedule
with a proper search. Two methods are supported:

  optuna  (default) - TPE sampling + median pruning. Each trial is a short
                       proxy run (e.g. 25 epochs instead of your full 200),
                       and Optuna kills clearly-losing trials partway
                       through by polling the trial's live epoch CSV.
  grid              - exhaustive grid over --dice_ratios x --lr_schedules.
                       No pruning (nothing adaptive to prune against), but
                       fully interpretable / reproducible.

Every trial runs train_centralized.py UNCHANGED, as a subprocess, with its
own isolated results/hp_search/<trial_name>/ output_dir so trials never
collide with each other or with your real runs' summary.csv.

Usage (run from the same directory as train_centralized.py, e.g. system/):
    pip install optuna --break-system-packages   # one-time

    # TPE search, 15 trials, 25-epoch proxy runs, optimizing f1_dam
    python hp_search.py --config config.yaml --n_trials 15 --epochs_per_trial 25

    # Exhaustive grid instead (5 schedules x 4 dice_ratios = 20 combos)
    python hp_search.py --config config.yaml --method grid \\
        --dice_ratios 0.3,0.5,0.7,0.9 \\
        --lr_schedules none,cosine,step,cosine_warm_restarts,plateau \\
        --epochs_per_trial 25

    # Optimize a different metric
    python hp_search.py --config config.yaml --metric miou

Output:
    results/hp_search/trials.csv        - every trial's params + achieved metric
    results/hp_search/best_config.yaml  - snippet ready to paste into config.yaml

Time budget (rule of thumb at ~80s/epoch):
    n_trials=15, epochs_per_trial=25  ->  ~8.5 hours worst case if every
    trial ran to completion, but MedianPruner typically cuts weak trials
    off well before that, so real runtime is usually noticeably lower.
    Increase --n_startup_trials if pruning feels too aggressive early on
    (it needs a few completed trials before it has a baseline to prune against).
"""
import argparse
import csv
import glob
import itertools
import os
import subprocess
import sys
import time
import shutil

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, required=True,
                     help="Same shared config.yaml used for train_centralized.py")
    ap.add_argument("--method", type=str, default="optuna", choices=["optuna", "grid"])
    ap.add_argument("--metric", type=str, default="f1_dam", choices=["f1_dam", "dice", "miou"],
                     help="Metric to optimize (default: f1_dam, the domain-standard "
                          "xView2 damage-classification metric)")
    ap.add_argument("--epochs_per_trial", type=int, default=25,
                     help="Short proxy-run length per trial (default: 25). "
                          "Class collapse/recovery signals show up well before your "
                          "full 200-epoch runs, so this is usually enough to rank configs.")
    ap.add_argument("--output_root", type=str, default="results/hp_search",
                     help="Each trial gets its own subdirectory under here")
    ap.add_argument("--python", type=str, default=sys.executable,
                     help="Python executable to launch trials with")
    ap.add_argument("--train_script", type=str, default="train_centralized.py")
    ap.add_argument("--poll_interval", type=float, default=5.0,
                     help="Seconds between polls of a trial's epoch CSV")
    ap.add_argument("--epoch_time_estimate", type=float, default=80.0,
                     help="Assumed seconds/epoch, used only for the upfront rough ETA print "
                          "(default 80.0, matching observed timing on this project so far)")

    # Optuna-specific
    ap.add_argument("--n_trials", type=int, default=15)
    ap.add_argument("--n_startup_trials", type=int, default=5,
                     help="Trials to run to completion before pruning kicks in")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tune_momentum", action="store_true",
                     help="Also search SGD momentum in [0.5, 0.99]. Off by default -- "
                          "you asked specifically about dice/focal/lr_schedule, and "
                          "momentum=0 is already known to fail catastrophically so it's "
                          "excluded from the search range rather than left for TPE to rediscover.")

    # Schedule-specific hyperparameters. Left at None by default, in which case
    # they're auto-scaled to --epochs_per_trial rather than using
    # train_centralized.py's own defaults (which are sized for full 200-epoch
    # runs -- e.g. step_size=30 would never fire within a 25-epoch proxy trial,
    # making 'step' indistinguishable from 'none' in the search). Override any
    # of these explicitly if you want a specific value tested instead.
    ap.add_argument("--lr_step_size", type=int, default=None)
    ap.add_argument("--lr_gamma", type=float, default=None)
    ap.add_argument("--lr_t0", type=int, default=None)
    ap.add_argument("--lr_tmult", type=int, default=None)
    ap.add_argument("--lr_plateau_patience", type=int, default=None)
    ap.add_argument("--lr_plateau_factor", type=float, default=None)

    # Grid-specific
    ap.add_argument("--dice_ratios", type=str, default="0.3,0.5,0.7,0.9",
                     help="Comma-separated dice_weight values to try (grid method only). "
                          "focal_weight = 1 - dice_weight for each.")
    ap.add_argument("--lr_schedules", type=str, default="cosine,step,cosine_warm_restarts,plateau",
                     help="Comma-separated lr_schedule values to try (grid method only)")

    return ap.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Trial execution: launch train_centralized.py, poll its epoch CSV,
# optionally report to an Optuna trial for pruning.
# ──────────────────────────────────────────────────────────────────────

METRIC_COLUMN = {"f1_dam": "f1_dam", "dice": "dice", "miou": "miou"}


def format_duration(seconds):
    """Human-readable duration, e.g. 1h23m, 4m05s, 12s."""
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def find_epoch_csv(trial_dir, timeout=120):
    """Wait for train_centralized.py to create its *_epochs.csv in trial_dir."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        matches = glob.glob(os.path.join(trial_dir, "*_epochs.csv"))
        if matches:
            return matches[0]
        time.sleep(1.0)
    return None


def run_trial(args, params, trial_name, optuna_trial=None,
              trial_index=None, total_trials=None, search_start_time=None):
    """Launch one train_centralized.py subprocess, poll its live epoch CSV,
    report intermediate values to optuna_trial (if given) for pruning, and
    return the best value of the target metric observed.

    trial_index/total_trials/search_start_time are optional -- if given,
    live progress prints include overall search position/ETA, not just
    this trial's own progress.

    Returns
    -------
    (best_value, status) where status is "completed", "pruned", or "failed"
    """
    trial_dir = os.path.join(args.output_root, trial_name)

    if os.path.exists(trial_dir):
        shutil.rmtree(trial_dir)

    os.makedirs(trial_dir)
    log_path = os.path.join(trial_dir, "train.log")
    trial_start = time.time()
    

    cmd = [
        args.python, args.train_script,
        "--config", args.config,
        "--epochs", str(args.epochs_per_trial),
        "--output_dir", os.path.join(os.path.basename(args.output_root), trial_name),
        "--dice_weight", str(params["dice_weight"]),
        "--focal_weight", str(params["focal_weight"]),
        "--lr_schedule", params["lr_schedule"],
        "--checkpoint_gap", str(args.epochs_per_trial + 1),  # skip periodic checkpoints during search
        "--eval_gap", "1",
    ]
    if "momentum" in params:
        cmd += ["--momentum", str(params["momentum"])]

    lr_schedule = params["lr_schedule"]
    if lr_schedule == "step":
        step_size = args.lr_step_size or max(5, args.epochs_per_trial // 3)
        gamma = args.lr_gamma or 0.3  # steeper than train_centralized.py's 0.1 default, since
                                       # a short proxy run only gets 2-3 decay events at most
        cmd += ["--lr_step_size", str(step_size), "--lr_gamma", str(gamma)]
    elif lr_schedule == "cosine_warm_restarts":
        t0 = args.lr_t0 or max(5, args.epochs_per_trial // 3)
        tmult = args.lr_tmult or 2
        cmd += ["--lr_t0", str(t0), "--lr_tmult", str(tmult)]
    elif lr_schedule == "plateau":
        patience = args.lr_plateau_patience or max(3, args.epochs_per_trial // 5)
        factor = args.lr_plateau_factor or 0.5
        cmd += ["--lr_plateau_patience", str(patience), "--lr_plateau_factor", str(factor),
                "--lr_plateau_metric", args.metric]  # monitor whatever the search is optimizing

    print(
            f"\n[{trial_name}] launching",
            flush=True,
            )

    print(
            f"Log: {log_path}",
            flush=True,
        )

    print(
            " ".join(cmd),
            flush=True,
        )

    log_file = open(log_path, "w")

    try:
        
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
    )

        epoch_csv = find_epoch_csv(trial_dir)
        if epoch_csv is None:
            proc.kill()
            proc.wait()
            print(f"[{trial_name}] FAILED: no epoch CSV appeared within timeout", flush=True)
            return float("-inf"), "failed"

        metric_col = METRIC_COLUMN[args.metric]
        best_value = float("-inf")
        rows_seen = 0
        last_progress = time.time()

        while True:
            retcode = proc.poll()
            log_file.flush()
            os.fsync(log_file.fileno())
        

            # read any new rows since last poll
            try:
                with open(epoch_csv, "r", newline="") as f:
                    reader = list(csv.DictReader(f))
            except (FileNotFoundError, csv.Error):
                reader = []

            new_rows = reader[rows_seen:]
            rows_seen = len(reader)
            if new_rows:
                last_progress = time.time()

            for row in new_rows:
                if row.get("evaluated") != "1":
                    continue
                try:
                    value = float(row[metric_col])
                except (KeyError, ValueError):
                    continue
                best_value = max(best_value, value)
                
                if optuna_trial is not None:
                    epoch = int(row["epoch"])
                    optuna_trial.report(value, step=epoch)
                    if optuna_trial.should_prune():
                        proc.kill()
                        proc.wait()
                        print(f"[{trial_name}] PRUNED at epoch {epoch} "
                            f"({args.metric}={value:.4f}, best_so_far={best_value:.4f})", flush=True)
                        import optuna
                        raise optuna.TrialPruned()

                epoch = int(row["epoch"])
                trial_elapsed = time.time() - trial_start
                avg_epoch_time = trial_elapsed / (epoch + 1)
                epochs_remaining = args.epochs_per_trial - (epoch + 1)
                trial_eta = avg_epoch_time * epochs_remaining

                line = (f"[{trial_name}] epoch {epoch + 1}/{args.epochs_per_trial} | "
                        f"{metric_col}={value:.4f} (best={best_value:.4f}) | "
                        f"~{avg_epoch_time:.0f}s/epoch | this trial ETA: {format_duration(trial_eta)}")

                if trial_index is not None and total_trials is not None and search_start_time is not None:
                    overall_elapsed = time.time() - search_start_time
                    line += (f" | trial {trial_index}/{total_trials} | "
                            f"search elapsed: {format_duration(overall_elapsed)}")

                print(line, flush=True)

            if retcode is not None:
                break
            if time.time() - last_progress > 900:
                proc.kill()
                proc.wait()

                print(
                    f"[{trial_name}] STALLED "
                    "(no new epochs for 5 minutes).",
                    flush=True,
                )

                return float("-inf"), "failed"
            time.sleep(args.poll_interval)
    finally:
        log_file.close()

    if retcode != 0:
        print(
            f"[{trial_name}] FAILED (exit {retcode})\n"
            f"See log: {log_path}",
            flush=True,
        )
        return float("-inf"), "failed"

    print(
        f"[{trial_name}] completed, best {args.metric}={best_value:.4f}",
        flush=True,
    )

    return best_value, "completed"


# ──────────────────────────────────────────────────────────────────────
# Results logging
# ──────────────────────────────────────────────────────────────────────

class TrialLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._write_header = not os.path.exists(path)

    def log(self, row):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow(row)


def write_best_config_yaml(path, best_params, metric_name, metric_value):
    with open(path, "w") as f:
        f.write(f"# Selected by hp_search.py, optimizing {metric_name}={metric_value:.4f}\n")
        f.write(f"dice_weight: {best_params['dice_weight']:.4f}\n")
        f.write(f"focal_weight: {best_params['focal_weight']:.4f}\n")
        f.write(f"lr_schedule: {best_params['lr_schedule']}\n")
        if "momentum" in best_params:
            f.write(f"momentum: {best_params['momentum']:.4f}\n")
    print(f"\nBest config written to {path} -- paste these lines into config.yaml, "
          f"then run the real 200-epoch training as usual.")


# ──────────────────────────────────────────────────────────────────────
# Optuna method
# ──────────────────────────────────────────────────────────────────────

def run_optuna(args):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    logger = TrialLogger(
        os.path.join(args.output_root, "trials.csv"),
        fieldnames=["trial_number", "dice_weight", "focal_weight", "lr_schedule",
                    "momentum", args.metric, "status"],
    )

    print(f"Optuna search: up to {args.n_trials} trials x {args.epochs_per_trial} epochs each "
          f"(pruning may cut many of these short)")
    est_hours = args.n_trials * args.epochs_per_trial * args.epoch_time_estimate / 3600
    print(f"Rough upper bound at ~{args.epoch_time_estimate:.0f}s/epoch, zero pruning benefit: "
          f"~{est_hours:.1f} hours. Real runtime is usually noticeably lower once pruning kicks in.\n")

    search_start = time.time()
    trial_durations = []

    def objective(trial):
        dice_ratio = trial.suggest_float("dice_ratio", 0.3, 0.9)
        lr_schedule = trial.suggest_categorical(
            "lr_schedule", ["cosine", "step", "cosine_warm_restarts", "plateau"]
        )
        params = {
            "dice_weight": dice_ratio,
            "focal_weight": 1.0 - dice_ratio,
            "lr_schedule": lr_schedule,
        }
        if args.tune_momentum:
            params["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)

        trial_name = f"trial_{trial.number:03d}"
        trial_t0 = time.time()
        value, status = run_trial(
            args, params, trial_name, optuna_trial=trial,
            trial_index=trial.number + 1, total_trials=args.n_trials, search_start_time=search_start,
        )
        trial_durations.append(time.time() - trial_t0)

        avg_trial_duration = sum(trial_durations) / len(trial_durations)
        remaining = max(args.n_trials - (trial.number + 1), 0)
        eta_remaining = avg_trial_duration * remaining
        print(f"=== {trial.number + 1}/{args.n_trials} trials attempted | "
              f"avg {format_duration(avg_trial_duration)}/trial so far "
              f"(mix of full + pruned trials, so this keeps refining) | "
              f"est. remaining: {format_duration(eta_remaining)} ===\n", flush=True)

        logger.log({
            "trial_number": trial.number,
            "dice_weight": params["dice_weight"],
            "focal_weight": params["focal_weight"],
            "lr_schedule": params["lr_schedule"],
            "momentum": params.get("momentum", ""),
            args.metric: value,
            "status": status,
        })

        if status == "failed":
            raise optuna.TrialPruned()  # don't let a crashed trial poison the search
        return value

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=args.n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("\nNo trials completed successfully -- check results/hp_search/trials.csv and trial stderr above.")
        return

    best = study.best_trial
    best_params = {
        "dice_weight": best.params["dice_ratio"],
        "focal_weight": 1.0 - best.params["dice_ratio"],
        "lr_schedule": best.params["lr_schedule"],
    }
    if args.tune_momentum:
        best_params["momentum"] = best.params["momentum"]

    print(f"\nBest trial: #{best.number}  {args.metric}={best.value:.4f}")
    print(f"  dice_weight  = {best_params['dice_weight']:.4f}")
    print(f"  focal_weight = {best_params['focal_weight']:.4f}")
    print(f"  lr_schedule  = {best_params['lr_schedule']}")
    if args.tune_momentum:
        print(f"  momentum     = {best_params['momentum']:.4f}")

    write_best_config_yaml(
        os.path.join(args.output_root, "best_config.yaml"),
        best_params, args.metric, best.value,
    )


# ──────────────────────────────────────────────────────────────────────
# Grid method
# ──────────────────────────────────────────────────────────────────────

def run_grid(args):
    dice_ratios = [float(x) for x in args.dice_ratios.split(",")]
    lr_schedules = [x.strip() for x in args.lr_schedules.split(",")]

    logger = TrialLogger(
        os.path.join(args.output_root, "trials.csv"),
        fieldnames=["trial_number", "dice_weight", "focal_weight", "lr_schedule", args.metric, "status"],
    )

    results = []
    combos = list(itertools.product(dice_ratios, lr_schedules))
    print(f"Grid search: {len(combos)} combinations x {args.epochs_per_trial} epochs each")
    est_hours = len(combos) * args.epochs_per_trial * args.epoch_time_estimate / 3600
    print(f"Rough upper bound at ~{args.epoch_time_estimate:.0f}s/epoch, no pruning in grid mode: "
          f"~{est_hours:.1f} hours\n")

    search_start = time.time()
    trial_durations = []

    for i, (dice_ratio, lr_schedule) in enumerate(combos):
        params = {
            "dice_weight": dice_ratio,
            "focal_weight": 1.0 - dice_ratio,
            "lr_schedule": lr_schedule,
        }
        trial_name = f"grid_{i:03d}"
        trial_t0 = time.time()
        value, status = run_trial(
            args, params, trial_name,
            trial_index=i + 1, total_trials=len(combos), search_start_time=search_start,
        )
        trial_durations.append(time.time() - trial_t0)

        avg_trial_duration = sum(trial_durations) / len(trial_durations)
        remaining_trials = len(combos) - (i + 1)
        eta_remaining = avg_trial_duration * remaining_trials
        print(f"=== {i + 1}/{len(combos)} trials done | "
              f"avg {format_duration(avg_trial_duration)}/trial | "
              f"est. remaining: {format_duration(eta_remaining)} ===\n", flush=True)

        logger.log({
            "trial_number": i,
            "dice_weight": params["dice_weight"],
            "focal_weight": params["focal_weight"],
            "lr_schedule": params["lr_schedule"],
            args.metric: value,
            "status": status,
        })
        if status == "completed":
            results.append((value, params))

    if not results:
        print("\nNo trials completed successfully -- check results/hp_search/trials.csv and trial stderr above.")
        return

    best_value, best_params = max(results, key=lambda r: r[0])
    print(f"\nBest combo: {args.metric}={best_value:.4f}")
    print(f"  dice_weight  = {best_params['dice_weight']:.4f}")
    print(f"  focal_weight = {best_params['focal_weight']:.4f}")
    print(f"  lr_schedule  = {best_params['lr_schedule']}")

    write_best_config_yaml(
        os.path.join(args.output_root, "best_config.yaml"),
        best_params, args.metric, best_value,
    )


# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    if not os.path.exists(args.train_script):
        print(f"ERROR: {args.train_script} not found in the current directory. "
              f"Run hp_search.py from the same directory as train_centralized.py "
              f"(e.g. system/), or pass --train_script with a path.")
        sys.exit(1)

    if args.method == "optuna":
        run_optuna(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()