"""Parse autoresearch experiment history and plot progress.

Parses both git log (committed experiments) and scratchpad.md (all experiments
including reverted ones) to create a comprehensive visualization.

Usage:
    uv run python autoresearch/plot_results.py          # print table
    uv run python autoresearch/plot_results.py --plot    # save plot
"""

import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------------------------------------------------
# Parse experiment data from scratchpad (has ALL experiments, including reverted)
# ---------------------------------------------------------------------------
def parse_scratchpad():
    """Parse the experiment log table from scratchpad.md."""
    scratchpad = Path(__file__).parent / "scratchpad.md"
    if not scratchpad.exists():
        return []

    text = scratchpad.read_text()

    # Find table rows: | # | Change | mean_rank | delta | peak_mem | Status |
    rows = re.findall(
        r"\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)\s*\|\s*([+\-\d.—]+)\s*\|\s*([\d.]+)\s*\|\s*(\w+.*?)\s*\|",
        text,
    )

    experiments = []
    for num, desc, mean_rank, std, delta, peak_mem, status in rows:
        delta_val = None
        if delta not in ("—", "-", ""):
            try:
                delta_val = float(delta)
            except ValueError:
                pass

        experiments.append({
            "num": int(num),
            "description": desc.strip(),
            "mean_rank": float(mean_rank),
            "std": float(std),
            "delta": delta_val,
            "peak_mem": float(peak_mem),
            "status": status.strip().lower(),
            "committed": "committed" in status.lower() or "reference" in status.lower(),
        })

    return experiments


# ---------------------------------------------------------------------------
# Parse git log (committed experiments only, with extra metadata)
# ---------------------------------------------------------------------------
def parse_git_log():
    """Extract experiment data from structured commit messages."""
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%H%n%aI%n%s%n%b%n---END---",
         "autoresearch-ukiyo", "--not", "main"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent.parent,
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

        mean_rank_m = re.search(r"mean_rank:\s*([\d.]+)\s*±\s*([\d.]+)", body)
        delta_m = re.search(r"delta:\s*(-?[\d.]+)", body)
        phase_m = re.search(r"phase:\s*(\d+)", body)
        peak_mem_m = re.search(r"peak_memory_gb:\s*([\d.]+)", body)
        model_m = re.search(r"Co-Authored-By: Claude (\w+)", body)

        if not mean_rank_m:
            continue

        experiments.append({
            "sha": sha,
            "timestamp": timestamp,
            "description": subject.replace("experiment:", "").strip(),
            "mean_rank": float(mean_rank_m.group(1)),
            "std": float(mean_rank_m.group(2)),
            "delta": float(delta_m.group(1)) if delta_m else None,
            "phase": int(phase_m.group(1)) if phase_m else None,
            "peak_mem": float(peak_mem_m.group(1)) if peak_mem_m else None,
            "model": model_m.group(1).lower() if model_m else "unknown",
        })

    return experiments


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------
def print_table(experiments):
    """Print results as a markdown table."""
    print("| # | Description | mean_rank | delta | Status |")
    print("|---|-------------|-----------|-------|--------|")

    for exp in experiments:
        delta_str = f"{exp['delta']:+.2f}" if exp.get("delta") is not None else "—"
        status = "committed" if exp.get("committed", True) else "REVERTED"
        print(f"| {exp.get('num', '?')} | {exp['description'][:45]} | "
              f"{exp['mean_rank']:.2f} ± {exp['std']:.2f} | {delta_str} | {status} |")


# ---------------------------------------------------------------------------
# Karpathy-style plot
# ---------------------------------------------------------------------------
def plot(all_experiments, committed_experiments, dark=False):
    """Create a publication-quality autoresearch progress plot.

    Only shows committed experiments in the main plot.
    Reverted experiments summarized in a side table.
    """

    if dark:
        bg = "#1a1a2e"
        fg = "#e0e0e0"
        fg_dim = "#888"
        grid_color = "#2a2a4a"
        line_color = "#aaa"
        band_color = "#E8590C"
        dot_edge = "#1a1a2e"
        baseline_color = "#666"
        table_bg = "#1a1a2e"
        table_alt = "#222244"
        table_header = "#2a2a4a"
        table_edge = "#333366"
        delta_green = "#4ADE80"
        delta_red = "#F87171"
        test_color = "#F87171"
        facecolor = bg
    else:
        bg = "white"
        fg = "#333"
        fg_dim = "#888"
        grid_color = "#f0f0f0"
        line_color = "#333"
        band_color = "#E8590C"
        dot_edge = "white"
        baseline_color = "#999"
        table_bg = "white"
        table_alt = "#fafafa"
        table_header = "#f0f0f0"
        table_edge = "#e0e0e0"
        delta_green = "#16A34A"
        delta_red = "#DC2626"
        test_color = "#DC2626"
        facecolor = "white"

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Noto Serif", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.titlesize": 15,
        "text.color": fg,
        "axes.labelcolor": fg,
        "axes.edgecolor": fg_dim,
        "xtick.color": fg,
        "ytick.color": fg,
    })

    # Split committed vs reverted
    kept = [e for e in all_experiments if e.get("committed")]
    reverted = [e for e in all_experiments if not e.get("committed")]

    n = len(kept)
    x = np.arange(n)

    ranks = [e["mean_rank"] for e in kept]
    stds = [e["std"] for e in kept]

    # Match models from git log
    def get_model(exp):
        for ce in committed_experiments:
            if abs(ce["mean_rank"] - exp["mean_rank"]) < 0.01:
                return ce.get("model", "unknown")
        return "unknown"

    models = [get_model(e) for e in kept]
    model_colors = {"opus": "#E8590C", "sonnet": "#2563EB", "haiku": "#16A34A", "unknown": "#E8590C"}

    # Layout: main plot left, reverted table right
    fig = plt.figure(figsize=(16, 7), facecolor=facecolor)
    if reverted:
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)
        ax = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
    else:
        ax = fig.add_subplot(111)
    ax.set_facecolor(facecolor)

    # --- Main plot ---

    # Shaded std band
    ax.fill_between(x, [r - s for r, s in zip(ranks, stds)],
                    [r + s for r, s in zip(ranks, stds)],
                    alpha=0.15 if dark else 0.1, color=band_color)

    # Connect with line
    ax.plot(x, ranks, color=line_color, linewidth=1.5, alpha=0.4, zorder=1)

    # Dots colored by model
    for i, (r, m) in enumerate(zip(ranks, models)):
        ax.scatter(i, r, color=model_colors.get(m, fg), s=120, zorder=4,
                   edgecolors=dot_edge, linewidths=1.5)

    # Baseline reference
    ax.axhline(y=344.68, color=baseline_color, linestyle=":", alpha=0.5, linewidth=1)
    ax.text(n - 0.5, 348, "baseline (344.7)", ha="right", va="bottom",
            fontsize=9, color=baseline_color, style="italic")

    # Annotate each point: delta and description, alternating above/below
    for i, exp in enumerate(kept):
        delta = exp.get("delta")

        # Clean up description for display
        desc = exp["description"]
        for prefix in ["increase ", "reduce ", "fix "]:
            if desc.lower().startswith(prefix):
                desc = desc[len(prefix):]
                break
        # Make code-style names human-readable
        desc = (desc
                .replace("PROJ_DIM", "proj dim")
                .replace("MAX_TEMPERATURE", "max temp")
                .replace("MAX_STEPS", "steps")
                .replace("WARMUP_STEPS", "warmup")
                .replace("EXPERT_PROB", "expert prob")
                .replace("LEARNING_RATE", "LR")
                .replace("BATCH_SIZE", "batch size")
                .replace("MIXUP_ALPHA", "mixup α")
                .replace("AUX_LOSS_WEIGHT", "aux weight")
                .replace("WEIGHT_DECAY", "weight decay")
                .replace("_", " "))
        desc = desc[:24]

        if i % 2 == 0:
            # Above the point: delta, then description
            delta_y, desc_y = 14, 26
        else:
            # Below the point: description, then delta
            delta_y, desc_y = -16, -28

        if delta is not None and exp.get("num", 0) > 1:
            color = delta_green if delta < 0 else delta_red
            ax.annotate(f"{delta:+.1f}", (i, ranks[i]),
                        textcoords="offset points", xytext=(0, delta_y),
                        ha="center", fontsize=7.5, color=color, fontweight="bold")

        ax.annotate(desc, (i, ranks[i]),
                    textcoords="offset points", xytext=(0, desc_y),
                    ha="center", fontsize=6, color=fg_dim)

    # Annotate best
    best_idx = np.argmin(ranks)
    ax.annotate(
        f"best: {ranks[best_idx]:.1f}",
        (best_idx, ranks[best_idx]),
        textcoords="offset points", xytext=(20, 18),
        fontsize=12, fontweight="bold", color=band_color,
        arrowprops=dict(arrowstyle="->", color=band_color, lw=1.2,
                        shrinkB=8),
    )

    # X-axis: experiment numbers
    ax.set_xticks(x)
    ax.set_xticklabels([str(e.get("num", i)) for i, e in enumerate(kept)], fontsize=9)

    ax.set_ylabel("Val Mean Rank (lower is better)", fontsize=12)
    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_title("eCLIP Autoresearch Progress",
                 fontsize=14, fontweight="bold", pad=15)

    # Final test result: star marker at the end
    # Scale test mean_rank to val pool size for comparability
    # Test: 34.30 on 1,100 candidates → scaled to 3,297 candidates
    test_rank_raw = 34.30
    test_pool = 1100
    val_pool = 3297
    test_rank_scaled = test_rank_raw * (val_pool / test_pool)  # ~102.8

    test_x = n
    ax.scatter(test_x, test_rank_scaled, color=test_color, s=200, zorder=5,
               marker="*", edgecolors=dot_edge, linewidths=0.8)
    ax.annotate(
        f"Test (scaled): {test_rank_scaled:.0f}\n"
        f"R@1=26.8%  R@5=53.0%\n"
        f"(raw: {test_rank_raw:.1f} on {test_pool} samples)",
        (test_x, test_rank_scaled),
        textcoords="offset points", xytext=(12, 8),
        fontsize=7.5, fontweight="bold", color=test_color,
        arrowprops=dict(arrowstyle="->", color=test_color, lw=1, shrinkB=8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=facecolor, edgecolor=test_color, alpha=0.9),
    )
    # Extend x-axis to fit test point
    ax.set_xlim(-0.5, test_x + 1.5)

    # Add summary stats
    total = len(all_experiments)
    n_committed = len(kept)
    n_reverted = len(reverted)
    ax.text(0.02, 0.02,
            f"{total} experiments  ·  {n_committed} committed  ·  {n_reverted} reverted  ·  1 Saturday  ·  1 GPU",
            transform=ax.transAxes, fontsize=8, color=fg_dim, va="bottom")

    # Legend for models
    from matplotlib.lines import Line2D
    seen_models = set(models)
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[m],
               markersize=9, markeredgecolor="white", label=m.capitalize())
        for m in ["opus", "sonnet", "haiku"] if m in seen_models
    ]
    ax.legend(handles=legend_items, loc="center right", fontsize=9,
              framealpha=0.9, edgecolor=table_edge,
              facecolor=facecolor, labelcolor=fg)

    # --- Reverted experiments table (right panel) ---
    if reverted:
        ax_table.set_facecolor(facecolor)
        ax_table.axis("off")
        ax_table.set_title("Reverted experiments", fontsize=10, fontweight="bold",
                           color=fg_dim, pad=10)

        table_data = []
        for e in reverted:
            desc = (e["description"][:30]
                    .replace("PROJ_DIM", "proj dim")
                    .replace("MAX_TEMPERATURE", "max temp")
                    .replace("MAX_STEPS", "steps")
                    .replace("WARMUP_STEPS", "warmup")
                    .replace("EXPERT_PROB", "expert prob")
                    .replace("LEARNING_RATE", "LR")
                    .replace("BATCH_SIZE", "batch size")
                    .replace("MIXUP_ALPHA", "mixup α")
                    .replace("AUX_LOSS_WEIGHT", "aux weight")
                    .replace("WEIGHT_DECAY", "weight decay")
                    .replace("_", " "))
            delta = e.get("delta")
            delta_str = f"{delta:+.1f}" if delta is not None else "—"
            table_data.append([f"#{e.get('num', '?')}", desc, delta_str])

        table = ax_table.table(
            cellText=table_data,
            colLabels=["#", "Change", "\u0394"],
            colWidths=[0.10, 0.70, 0.20],
            loc="upper center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.3)

        # Style the table
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(table_edge)
            cell.set_text_props(family="serif", color=fg)
            if row == 0:
                cell.set_facecolor(table_header)
                cell.set_text_props(fontweight="bold", fontsize=8, family="serif", color=fg)
            else:
                cell.set_facecolor(table_bg if row % 2 == 1 else table_alt)
                # Color deltas
                if col == 2:
                    text = cell.get_text().get_text()
                    if text.startswith("+"):
                        cell.get_text().set_color(delta_red)
                    elif text.startswith("-"):
                        cell.get_text().set_color(delta_green)

    plt.tight_layout()

    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    suffix = "_dark" if dark else ""
    outfile = out / f"progress{suffix}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor=facecolor)
    print(f"Saved: {outfile}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_experiments = parse_scratchpad()
    committed_experiments = parse_git_log()

    if not all_experiments:
        # Fall back to git log only
        all_experiments = committed_experiments

    if "--plot" in sys.argv:
        dark = "--dark" in sys.argv
        plot(all_experiments, committed_experiments, dark=dark)
        if "--both" in sys.argv:
            plot(all_experiments, committed_experiments, dark=not dark)
    else:
        print_table(all_experiments)
        if all_experiments:
            best = min(all_experiments, key=lambda e: e["mean_rank"])
            print(f"\nBest: {best['description']} — {best['mean_rank']:.2f} ± {best['std']:.2f}")
            print(f"Total experiments: {len(all_experiments)} "
                  f"({sum(1 for e in all_experiments if e.get('committed'))} committed, "
                  f"{sum(1 for e in all_experiments if not e.get('committed'))} reverted)")
