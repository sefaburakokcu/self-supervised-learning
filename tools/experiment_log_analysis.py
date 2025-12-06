import re
import os
import glob
import json
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from collections import defaultdict


# -----------------------------------------------------------
#               PARSE A SINGLE LOG FILE
# -----------------------------------------------------------
def parse_experiment_log(log_path):
    # Dynamic metric storage
    metrics = defaultdict(list)

    exp_name = None
    batch_size = None
    start_time = None
    end_time = None

    # Patterns
    ts_pattern = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
    exp_pattern = r"Experiment requested:\s*(.*)"
    batch_pattern = r"batch_size=(\d+)"

    # Generic epoch metric pattern (auto-detect keys!)
    epoch_metrics_pattern = (
        r"Epoch\s+(\d+)\/\d+\s*\|\s*(.*)"  # captures the whole metric section
    )

    # Key-value pairs inside metrics
    kv_pattern = r"([\w_]+):\s*([\d.]+)"

    with open(log_path, "r") as f:
        for line in f:
            # timestamp
            ts = re.search(ts_pattern, line)
            if ts:
                t = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
                if start_time is None:
                    start_time = t
                end_time = t

            # experiment name
            exp = re.search(exp_pattern, line)
            if exp:
                exp_name = exp.group(1).strip()

            # batch size
            bs = re.search(batch_pattern, line)
            if bs:
                batch_size = int(bs.group(1))

            # epoch-level metrics (auto-detect)
            em = re.search(epoch_metrics_pattern, line)
            if em:
                epoch_id = int(em.group(1))
                metrics["epoch"].append(epoch_id)

                metric_blob = em.group(2)
                for k, v in re.findall(kv_pattern, metric_blob):
                    metrics[k.lower().replace(" ", "_")].append(float(v))

    runtime = (end_time - start_time).total_seconds() if end_time else None

    return {
        "experiment": exp_name,
        "batch_size": batch_size,
        "runtime_sec": runtime,
        "metrics": dict(metrics),
        "path": log_path
    }


# -----------------------------------------------------------
#               SAVE SUMMARY TABLE + JSON
# -----------------------------------------------------------
def save_summary_single(data, save_json=True):
    metrics = data["metrics"]
    last_row = {k: (v[-1] if len(v) > 0 else None) for k, v in metrics.items()}

    summary = {
        "experiment": data["experiment"],
        "batch_size": data["batch_size"],
        "runtime_sec": data["runtime_sec"],
        **{("final_" + k): v for k, v in last_row.items()}
    }

    df = pd.DataFrame([summary])

    if save_json:
        with open("summary.json", "w") as f:
            json.dump(summary, f, indent=4)

    # Full column print
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)

    return df


# -----------------------------------------------------------
#                       PLOTTER
# -----------------------------------------------------------
def plot_metrics(data, save_png=True):
    metrics = data["metrics"]
    epochs = metrics["epoch"]

    # LOSS
    if "train_loss" in metrics and "val_loss" in metrics:
        plt.figure(figsize=(12, 6), dpi=150)
        plt.plot(epochs, metrics["train_loss"], label="Train Loss", linewidth=2)
        plt.plot(epochs, metrics["val_loss"], label="Val Loss", linewidth=2)
        plt.title("Loss Curve", fontsize=18)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        if save_png:
            plt.savefig("loss_curve.png")
        plt.show()

    # ACCURACY
    if "train_acc" in metrics and "val_acc" in metrics:
        plt.figure(figsize=(12, 6), dpi=150)
        plt.plot(epochs, metrics["train_acc"], label="Train Acc", linewidth=2)
        plt.plot(epochs, metrics["val_acc"], label="Val Acc", linewidth=2)
        plt.title("Accuracy Curve", fontsize=18)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        if save_png:
            plt.savefig("acc_curve.png")
        plt.show()


# -----------------------------------------------------------
#             MULTI-EXPERIMENT COMPARISON
# -----------------------------------------------------------
def build_experiment_summary(exp_data):
    metrics = exp_data["metrics"]

    # final values (last recorded value for each metric)
    final_vals = {
        f"final_{k}": (v[-1] if len(v) > 0 else None)
        for k, v in metrics.items()
        if k != "epoch"
    }

    return {
        "experiment": exp_data["experiment"],
        "clean_name": exp_data["experiment"].split("/")[-1] if exp_data["experiment"] else "unknown",
        "epoch": max(metrics["epoch"]) if "epoch" in metrics else 0,
        "batch_size": exp_data["batch_size"],
        "runtime_sec": exp_data["runtime_sec"],
        "runtime_hour": exp_data["runtime_sec"] / 3600 if exp_data["runtime_sec"] else 0,
        **final_vals
    }

def save_comparison_results(experiments, json_path="comparison_results.json"):
    summaries = [build_experiment_summary(exp) for exp in experiments]
    df = pd.DataFrame(summaries)

    # Pretty-print full table
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n=== COMPARISON SUMMARY TABLE ===")
        print(df)

    # Save JSON file
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=4)

    print(f"\nSaved comparison JSON → {json_path}\n")

    return df


def compare_experiments(log_paths, save_png=True):
    experiments = [parse_experiment_log(p) for p in log_paths]
    experiments = sorted(
        experiments,
        key=lambda e: e["experiment"].split("/")[-1] if e["experiment"] else ""
    )

    save_comparison_results(experiments)

    # Collect all metric keys across experiments
    all_metric_keys = set()
    for exp in experiments:
        all_metric_keys.update(exp["metrics"].keys())
    all_metric_keys.discard("epoch")  # x-axis only

    for metric_name in sorted(all_metric_keys):
        plt.figure(figsize=(12, 6), dpi=150)

        for exp in experiments:
            metrics = exp["metrics"]
            if metric_name not in metrics:
                continue

            epochs = metrics["epoch"]
            values = metrics[metric_name]

            # Align lengths if different epoch counts
            if len(values) != len(epochs):
                min_len = min(len(values), len(epochs))
                epochs = epochs[:min_len]
                values = values[:min_len]

            # --- NEW CLEAN NAME EXTRACTION ---
            exp_name_clean = exp["experiment"].split("/")[-1] if exp["experiment"] else "unknown"

            plt.plot(
                epochs,
                values,
                linewidth=2,
                label=f"{exp_name_clean}"
            )

        plt.title(f"Comparison: {metric_name}", fontsize=18)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()

        if save_png:
            plt.savefig(f"compare_{metric_name}.png")

        plt.show()


def load_all_logs_from_folder(folder_path, extensions=(".log", ".txt")):
    """
    Automatically finds all experiment logs inside a folder.
    Returns a list of file paths sorted by timestamp in filename (if present).
    """
    logs = []
    for ext in extensions:
        logs.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))

    if not logs:
        print("No log files found in:", folder_path)
        return []

    # Sort naturally (good for timestamped file names)
    logs = sorted(logs, key=lambda x: os.path.basename(x))
    print(f"Found {len(logs)} logs:")
    for l in logs:
        print(" -", l)

    return logs


# -----------------------------------------------------------
#                      USAGE
# -----------------------------------------------------------

# Single log:
# data = parse_experiment_log("log1.log")
# save_summary_single(data)
# plot_metrics(data)

# Multi-log comparison:
# compare_experiments(["log1.log", "log2.log", "log3.log"])

folder_path = "../experiment_results"
log_files = load_all_logs_from_folder(folder_path)
compare_experiments(log_files)
