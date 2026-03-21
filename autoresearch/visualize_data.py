"""Visualize Ukiyo-eVG samples with heatmap overlays.

Usage: uv run python autoresearch/visualize_data.py
Outputs: data/ukiyoe/viz/ with sample images
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_annotations, regions_to_heatmap, bbox_to_heatmap, IMAGES_DIR, TRAIN_JSONL, TEST_JSONL

VIZ_DIR = Path(__file__).resolve().parent.parent / "data" / "ukiyoe" / "viz"


def visualize_sample(sample, out_path, show_individual_regions=True):
    """Create a visualization of one sample: original image, bboxes, heatmap overlay."""
    img_path = IMAGES_DIR / sample.filename

    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        has_image = True
    except (FileNotFoundError, OSError):
        # Use gray placeholder if image not available
        img_np = np.full((sample.height, sample.width, 3), 200, dtype=np.uint8)
        has_image = False

    h, w = img_np.shape[:2]

    # Generate heatmap at original resolution
    heatmap = regions_to_heatmap(sample.regions, sample.width, sample.height, size=max(h, w))
    heatmap_np = heatmap.squeeze().numpy()
    # Crop to actual image dimensions
    heatmap_np = heatmap_np[:h, :w] if heatmap_np.shape[0] >= h else np.pad(
        heatmap_np, ((0, h - heatmap_np.shape[0]), (0, w - heatmap_np.shape[1]))
    )

    n_regions = len(sample.regions)
    n_cols = 2 + (min(n_regions, 4) if show_individual_regions else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # 1) Original image with bounding boxes
    ax = axes[0]
    ax.imshow(img_np)
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_regions, 1)))
    for i, region in enumerate(sample.regions):
        x1, y1, x2, y2 = region["bbox"]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=colors[i % len(colors)], facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5, region["phrase"],
            fontsize=8, color="white", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i % len(colors)], alpha=0.8),
        )
    ax.set_title(f"Bounding Boxes\n{sample.filename}", fontsize=9)
    ax.axis("off")

    # 2) Heatmap overlay
    ax = axes[1]
    ax.imshow(img_np)
    ax.imshow(heatmap_np, alpha=0.5, cmap="jet")
    ax.set_title(f"Merged Heatmap\n\"{sample.caption[:60]}...\"" if len(sample.caption) > 60
                 else f"Merged Heatmap\n\"{sample.caption}\"", fontsize=9)
    ax.axis("off")

    # 3) Individual region heatmaps
    if show_individual_regions:
        for i, region in enumerate(sample.regions[:4]):
            ax = axes[2 + i]
            single_hm = bbox_to_heatmap(region["bbox"], sample.width, sample.height, size=max(h, w))
            single_hm_np = single_hm.squeeze().numpy()[:h, :w]
            ax.imshow(img_np)
            ax.imshow(single_hm_np, alpha=0.5, cmap="jet")
            ax.set_title(f"\"{region['phrase']}\"", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return True


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Check if images exist
    if not IMAGES_DIR.exists() or not any(IMAGES_DIR.iterdir()):
        print(f"No images in {IMAGES_DIR} — download still in progress?")
        print("Will visualize heatmaps only (no image background).\n")

    # Load samples from both train and test
    print("Loading annotations...")
    train_samples = load_annotations(TRAIN_JSONL)
    test_samples = load_annotations(TEST_JSONL)

    # Pick diverse samples: different region counts
    viz_samples = []

    # From train (pseudo bboxes)
    for s in train_samples:
        if len(s.regions) == 1 and len(viz_samples) < 2:
            viz_samples.append(("train", s))
        elif len(s.regions) == 2 and len(viz_samples) < 4:
            viz_samples.append(("train", s))
        elif len(s.regions) >= 3 and len(viz_samples) < 6:
            viz_samples.append(("train", s))

    # From test (manual bboxes)
    for s in test_samples:
        if len(s.regions) == 1 and len(viz_samples) < 8:
            viz_samples.append(("test", s))
        elif len(s.regions) >= 2 and len(viz_samples) < 10:
            viz_samples.append(("test", s))

    print(f"Visualizing {len(viz_samples)} samples...\n")
    created = 0
    for i, (split, sample) in enumerate(viz_samples):
        out_path = VIZ_DIR / f"sample_{i:02d}_{split}_{sample.filename.replace('.jpg', '.png')}"
        print(f"  [{split}] {sample.filename}: \"{sample.caption}\" ({len(sample.regions)} regions)")
        if visualize_sample(sample, out_path):
            print(f"    → {out_path.name}")
            created += 1

    print(f"\nDone! Created {created} visualizations in {VIZ_DIR}")
    if created == 0:
        print("No images found — run extract_images.py after download completes.")


if __name__ == "__main__":
    main()
