"""Parse autoresearch git log and plot improvement over time."""

import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_git_log():
    """Extract experiment data from structured commit messages."""
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%H%n%aI%n%s%n%b%n---END---", "autoresearch-ukiyo", "--not", "main"],
        capture_output=True, text=True, cwd=".."
    )

    experiments = []
    for block in result.stdout.split("---END---"):
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        if len(lines) < 3:
            continue

        sha = lines[0][:8]
        timestamp = lines[1]
        subject = lines[2]

        if not subject.startswith("experiment:"):
            continue

        body = "\n".join(lines[3:])

        # Parse fields
        mean_rank_m = re.search(r"mean_rank:\s*([\d.]+)\s*±\s*([\d.]+)", body)
        delta_m = re.search(r"delta:\s*(-?[\d.]+)", body)
        folds_m = re.search(r"folds:\s*\[([\d., ]+)\]", body)
        phase_m = re.search(r"phase:\s*(\d+)", body)
        peak_mem_m = re.search(r"peak_memory_gb:\s*([\d.]+)", body)

        if not mean_rank_m:
            continue

        exp = {
            "sha": sha,
            "timestamp": timestamp,
            "description": subject.replace("experiment:", "").strip(),
            "mean_rank": float(mean_rank_m.group(1)),
            "std": float(mean_rank_m.group(2)),
            "delta": float(delta_m.group(1)) if delta_m else None,
            "phase": int(phase_m.group(1)) if phase_m else None,
            "peak_mem": float(peak_mem_m.group(1)) if peak_mem_m else None,
            "folds": [float(x.strip()) for x in folds_m.group(1).split(",")] if folds_m else None,
        }
        experiments.append(exp)

    return experiments


def print_table(experiments):
    """Print results as a markdown table."""
    print("| # | Description | mean_rank | delta | peak_mem | phase |")
    print("|---|-------------|-----------|-------|----------|-------|")

    # Baselines
    print("| 0 | CLIP baseline (no expert) | 345.91 ± 10.41 | - | 6.3 | - |")
    print("| 1 | eCLIP baseline (EXPERT_PROB=0.3) | 344.68 ± 12.66 | - | 11.3 | - |")

    for i, exp in enumerate(experiments, start=2):
        delta_str = f"{exp['delta']:+.2f}" if exp['delta'] is not None else "-"
        mem_str = f"{exp['peak_mem']:.1f}" if exp['peak_mem'] is not None else "-"
        phase_str = str(exp['phase']) if exp['phase'] is not None else "-"
        print(f"| {i} | {exp['description']} | {exp['mean_rank']:.2f} ± {exp['std']:.2f} | {delta_str} | {mem_str} | {phase_str} |")


def plot(experiments):
    """Plot mean_rank improvement over experiments."""
    if not experiments:
        print("No experiment commits found yet.")
        return

    # Include baselines
    names = ["CLIP\nbaseline", "eCLIP\nbaseline"] + [f"#{i+2}" for i in range(len(experiments))]
    ranks = [345.91, 344.68] + [e["mean_rank"] for e in experiments]
    stds = [10.41, 12.66] + [e["std"] for e in experiments]
    phases = [0, 0] + [e["phase"] or 0 for e in experiments]

    phase_colors = {0: "#888888", 1: "#4CAF50", 2: "#2196F3", 3: "#FF9800", 4: "#9C27B0"}
    colors = [phase_colors.get(p, "#888888") for p in phases]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    x = np.arange(len(names))
    bars = ax.bar(x, ranks, yerr=stds, capsize=3, color=colors, alpha=0.8, edgecolor="white")

    ax.set_ylabel("mean_rank (lower is better)")
    ax.set_title("eCLIP Autoresearch — Experiment Progress")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=345.91, color="#888888", linestyle="--", alpha=0.5, label="CLIP baseline")

    # Annotate best
    best_idx = np.argmin(ranks)
    ax.annotate(f"{ranks[best_idx]:.1f}", (best_idx, ranks[best_idx]),
                textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, fontweight="bold")

    # Legend for phases
    from matplotlib.patches import Patch
    legend_items = [Patch(color=c, label=f"Phase {p}") for p, c in sorted(phase_colors.items()) if p in phases and p > 0]
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("logs/progress.png", dpi=150)
    print("Saved: autoresearch/logs/progress.png")
    plt.show()


if __name__ == "__main__":
    experiments = parse_git_log()

    if "--plot" in sys.argv:
        plot(experiments)
    else:
        print_table(experiments)
        if experiments:
            best = min(experiments, key=lambda e: e["mean_rank"])
            print(f"\nBest so far: {best['description']} — {best['mean_rank']:.2f} ± {best['std']:.2f}")
