"""
prepare.py — Fixed data preparation for autoresearch.
DO NOT MODIFY THIS FILE. The agent should only modify train.py.

Handles:
1. Loading Ukiyo-eVG annotations (ODVG JSONL)
2. Converting bounding boxes to gaussian heatmaps
3. Building PyTorch datasets with 3-fold CV
4. Evaluation metrics
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "ukiyoe"
DATA_ROOT = Path(os.environ.get("ECLIP_DATA_ROOT", _DEFAULT_DATA))
ANNOTATIONS_DIR = DATA_ROOT / "annotations" / "Ukiyo-eVG"
IMAGES_DIR = DATA_ROOT / "images"

TRAIN_JSONL = ANNOTATIONS_DIR / "refined" / "ukiyoe_vg_train_pgt_odvg.jsonl"
VAL_JSONL = ANNOTATIONS_DIR / "refined" / "ukiyoe_vg_val_pgt_odvg.jsonl"
TEST_JSONL = ANNOTATIONS_DIR / "ukiyoe_vg_test_annotated.jsonl"

N_FOLDS = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    filename: str
    caption: str
    regions: list  # [{"bbox": [x1,y1,x2,y2], "phrase": str}, ...]
    width: int
    height: int


def load_annotations(jsonl_path: Path) -> list[Sample]:
    """Load ODVG JSONL annotations."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            grounding = d.get("grounding", {})
            if not grounding.get("caption") or not grounding.get("regions"):
                continue
            samples.append(
                Sample(
                    filename=d["filename"],
                    caption=grounding["caption"],
                    regions=grounding["regions"],
                    width=d["width"],
                    height=d["height"],
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Heatmap generation
# ---------------------------------------------------------------------------
def bbox_to_heatmap(bbox: list[float], img_w: int, img_h: int, size: int = 224) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] bounding box to a gaussian heatmap tensor.

    Returns a (1, size, size) tensor with values in [0, 1].
    """
    x1, y1, x2, y2 = bbox
    # Normalize to [0, 1]
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    sx = ((x2 - x1) / img_w) / 2  # half-width as sigma
    sy = ((y2 - y1) / img_h) / 2  # half-height as sigma

    # Clamp sigma to avoid degenerate gaussians
    sx = max(sx, 0.02)
    sy = max(sy, 0.02)

    # Build gaussian
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, size),
        torch.linspace(0, 1, size),
        indexing="ij",
    )
    heatmap = torch.exp(-((xx - cx) ** 2 / (2 * sx**2) + (yy - cy) ** 2 / (2 * sy**2)))
    return heatmap.unsqueeze(0)  # (1, H, W)


def regions_to_heatmap(regions: list[dict], img_w: int, img_h: int, size: int = 224) -> torch.Tensor:
    """Merge all region bounding boxes into a single heatmap (max over gaussians)."""
    heatmap = torch.zeros(1, size, size)
    for region in regions:
        h = bbox_to_heatmap(region["bbox"], img_w, img_h, size)
        heatmap = torch.max(heatmap, h)
    return heatmap


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class UkiyoeCLIPDataset(Dataset):
    """Standard image-text dataset for CLIP training."""

    def __init__(self, samples: list[Sample], img_size: int = 224):
        self.samples = samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = IMAGES_DIR / sample.filename

        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            img = Image.new("RGB", (self.img_size, self.img_size))

        img = self.transform(img)
        return {
            "image": img,
            "caption": sample.caption,
        }


class UkiyoeExpertDataset(Dataset):
    """Expert dataset with heatmaps and region-level text snippets."""

    def __init__(self, samples: list[Sample], img_size: int = 224):
        self.samples = samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = IMAGES_DIR / sample.filename

        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            img = Image.new("RGB", (self.img_size, self.img_size))

        img = self.transform(img)

        # Build heatmap from all bounding boxes
        heatmap = regions_to_heatmap(sample.regions, sample.width, sample.height, self.img_size)

        # Pick a random region's phrase as the text snippet
        region = sample.regions[torch.randint(len(sample.regions), (1,)).item()]
        snippet = region["phrase"]

        return {
            "image": img,
            "caption": sample.caption,
            "heatmap": heatmap,
            "snippet": snippet,
        }


# ---------------------------------------------------------------------------
# 3-Fold CV dataloader builder
# ---------------------------------------------------------------------------
def build_fold_dataloaders(
    fold: int,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
) -> dict:
    """Build train/val dataloaders for a given fold.

    Pools train+val annotations (9,893 samples), splits into 3 folds.
    Test set (1,100 manually annotated) is never touched here.

    Returns dict with keys: train, train_expert, val
    """
    # Pool train + val (both have pseudo bboxes)
    all_samples = load_annotations(TRAIN_JSONL) + load_annotations(VAL_JSONL)

    # Deterministic shuffle
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_samples))

    # Split into N_FOLDS
    fold_size = len(indices) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start = i * fold_size
        end = start + fold_size if i < N_FOLDS - 1 else len(indices)
        folds.append(indices[start:end])

    # Val = current fold, train = rest
    val_indices = folds[fold]
    train_indices = np.concatenate([f for i, f in enumerate(folds) if i != fold])

    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]

    train_ds = UkiyoeCLIPDataset(train_samples, img_size)
    train_expert_ds = UkiyoeExpertDataset(train_samples, img_size)
    val_ds = UkiyoeExpertDataset(val_samples, img_size)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        "train_expert": DataLoader(train_expert_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
    }


def build_test_dataloader(
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
) -> DataLoader:
    """Build test dataloader. Only call this once at the very end."""
    test_samples = load_annotations(TEST_JSONL)
    test_ds = UkiyoeExpertDataset(test_samples, img_size)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device="cuda") -> dict:
    """Compute image-text retrieval metrics.

    Returns dict with: img2txt_r1, img2txt_r5, img2txt_r10,
                        txt2img_r1, txt2img_r5, txt2img_r10, mean_rank
    """
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    for batch in dataloader:
        images = batch["image"].to(device)
        tokens = tokenizer(
            batch["caption"],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        img_emb = model.encode_image(images)
        txt_emb = model.encode_text(tokens)

        # L2 normalize
        img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
        txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)

        all_image_embeds.append(img_emb.cpu())
        all_text_embeds.append(txt_emb.cpu())

    all_image_embeds = torch.cat(all_image_embeds)
    all_text_embeds = torch.cat(all_text_embeds)

    # Compute similarity matrix
    sim = all_image_embeds @ all_text_embeds.t()
    n = sim.shape[0]

    # Image-to-text retrieval
    i2t_ranks = []
    for i in range(n):
        rank = (sim[i].argsort(descending=True) == i).nonzero().item()
        i2t_ranks.append(rank)

    # Text-to-image retrieval
    t2i_ranks = []
    for i in range(n):
        rank = (sim[:, i].argsort(descending=True) == i).nonzero().item()
        t2i_ranks.append(rank)

    i2t_ranks = torch.tensor(i2t_ranks, dtype=torch.float)
    t2i_ranks = torch.tensor(t2i_ranks, dtype=torch.float)

    metrics = {
        "img2txt_r1": (i2t_ranks < 1).float().mean().item(),
        "img2txt_r5": (i2t_ranks < 5).float().mean().item(),
        "img2txt_r10": (i2t_ranks < 10).float().mean().item(),
        "txt2img_r1": (t2i_ranks < 1).float().mean().item(),
        "txt2img_r5": (t2i_ranks < 5).float().mean().item(),
        "txt2img_r10": (t2i_ranks < 10).float().mean().item(),
        "mean_rank": ((i2t_ranks.mean() + t2i_ranks.mean()) / 2).item(),
    }

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading annotations...")
    train = load_annotations(TRAIN_JSONL)
    val = load_annotations(VAL_JSONL)
    test = load_annotations(TEST_JSONL)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Pool (train+val): {len(train) + len(val)}")
    print(f"3-fold CV: ~{(len(train) + len(val)) * 2 // 3} train, ~{(len(train) + len(val)) // 3} val per fold")

    # Show fold sizes
    for fold in range(N_FOLDS):
        loaders = build_fold_dataloaders(fold, batch_size=32)
        print(f"  Fold {fold}: train={loaders['n_train']}, val={loaders['n_val']}")

    # Show a sample
    s = train[0]
    print(f"\nSample: {s.filename}")
    print(f"Caption: {s.caption}")
    print(f"Regions: {len(s.regions)}")
    for r in s.regions:
        print(f"  - '{r['phrase']}' @ {r['bbox']}")

    # Test heatmap generation
    hm = regions_to_heatmap(s.regions, s.width, s.height)
    print(f"\nHeatmap shape: {hm.shape}, range: [{hm.min():.3f}, {hm.max():.3f}]")
