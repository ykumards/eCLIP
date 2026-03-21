"""
train.py — Mutable training script for autoresearch.
The agent CAN and SHOULD modify this file to run experiments.

Current baseline: eCLIP with ViT-Small + DistilBERT on Ukiyo-eVG.
Runs 3-fold CV, reports averaged mean_rank on val. Test is held out.
"""

import math
import os
import random
import sys
import time
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import build_fold_dataloaders, build_test_dataloader, evaluate, N_FOLDS
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model
IMAGE_ENCODER = "vit_small_patch16_224"  # timm model name
TEXT_ENCODER = "distilbert-base-uncased"
PROJ_DIM = 128
IMG_SIZE = 224

# Training
BATCH_SIZE = 48
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_STEPS = 400
WARMUP_STEPS = 40
LOG_EVERY = 50
EVAL_EVERY = 400  # eval at end of training only

# eCLIP expert settings
EXPERT_PROB = 0.3  # probability of using expert batch each step
MIXUP_ALPHA = 0.3  # beta distribution parameter for mixup
AUX_LOSS_WEIGHT = 0.1  # weight for auxiliary reconstruction loss

# Temperature
INIT_TEMPERATURE = 0.07
MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 2.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    """ViT-based image encoder using timm."""

    def __init__(self, model_name: str = IMAGE_ENCODER, proj_dim: int = PROJ_DIM):
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        embed_dim = self.backbone.num_features
        self.projector = nn.Linear(embed_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projector(features)


class TextEncoder(nn.Module):
    """Transformer-based text encoder."""

    def __init__(self, model_name: str = TEXT_ENCODER, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        embed_dim = self.backbone.config.hidden_size
        self.projector = nn.Linear(embed_dim, proj_dim)

    def forward(self, input_ids, attention_mask, **kwargs) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token pooling
        cls_output = outputs.last_hidden_state[:, 0]
        return self.projector(cls_output)


class HeatmapProcessor(nn.Module):
    """Processes heatmaps to create masked image representations.

    Applies spatial attention based on bounding box heatmaps using
    multi-head attention between heatmap-masked patches and original patches.
    """

    def __init__(self, embed_dim: int = PROJ_DIM, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.heatmap_embed = nn.Conv2d(1, embed_dim, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        # Patchify
        img_patches = self.patch_embed(image)  # (B, D, h, w)
        B, D, h, w = img_patches.shape
        img_patches = img_patches.flatten(2).transpose(1, 2)  # (B, N, D)

        heat_patches = self.heatmap_embed(heatmap)  # (B, D, h, w)
        heat_patches = heat_patches.flatten(2).transpose(1, 2)  # (B, N, D)

        # Heatmap patches as queries, image patches as key/value
        masked_out, _ = self.attention(
            query=self.norm(heat_patches),
            key=self.norm(img_patches),
            value=img_patches,
        )

        # Global average pool
        return masked_out.mean(dim=1)  # (B, D)


class ECLIPModel(nn.Module):
    """eCLIP: CLIP with expert heatmap attention."""

    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.heatmap_processor = HeatmapProcessor()

        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.tensor(math.log(1.0 / INIT_TEMPERATURE)))

    @property
    def temperature(self):
        return torch.clamp(self.log_temperature.exp(), MIN_TEMPERATURE, MAX_TEMPERATURE)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(self, tokens: dict) -> torch.Tensor:
        return self.text_encoder(**tokens)

    def encode_masked_image(self, images: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
        return self.heatmap_processor(images, heatmaps)

    def forward(self, images, tokens, heatmaps=None):
        img_emb = F.normalize(self.encode_image(images), dim=-1)
        txt_emb = F.normalize(self.encode_text(tokens), dim=-1)

        result = {"img_emb": img_emb, "txt_emb": txt_emb}

        if heatmaps is not None:
            masked_emb = F.normalize(self.encode_masked_image(images, heatmaps), dim=-1)
            result["masked_emb"] = masked_emb

        return result


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def contrastive_loss(img_emb, txt_emb, temperature):
    """Symmetric CLIP contrastive loss."""
    logits = (img_emb @ txt_emb.t()) * temperature
    n = logits.shape[0]
    labels = torch.arange(n, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2


def expert_loss(img_emb, masked_emb, txt_emb, snippet_emb, temperature, mixup_alpha=MIXUP_ALPHA):
    """Expert contrastive loss with mixup between original and masked embeddings."""
    lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 0.5
    mixed_emb = F.normalize(lam * img_emb + (1 - lam) * masked_emb, dim=-1)

    # Contrastive loss between mixed image and snippet text
    loss_mixed = contrastive_loss(mixed_emb, snippet_emb, temperature)

    # Also align masked embedding with full text
    loss_masked = contrastive_loss(masked_emb, txt_emb, temperature)

    return (loss_mixed + loss_masked) / 2


# ---------------------------------------------------------------------------
# Single fold training
# ---------------------------------------------------------------------------
def train_fold(fold: int, tokenizer) -> float:
    """Train one fold, return val mean_rank."""
    set_seed(SEED + fold)

    loaders = build_fold_dataloaders(fold, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    print(f"\n  Fold {fold}: train={loaders['n_train']}, val={loaders['n_val']}")

    model = ECLIPModel().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    train_iter = iter(loaders["train"])
    expert_iter = iter(loaders["train_expert"])
    start_time = time.time()

    for step in range(1, MAX_STEPS + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders["train"])
            batch = next(train_iter)

        images = batch["image"].to(DEVICE)
        tokens = tokenizer(batch["caption"], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)

        out = model(images, tokens)
        loss = contrastive_loss(out["img_emb"], out["txt_emb"], model.temperature)

        # Expert batch (probabilistic)
        if random.random() < EXPERT_PROB:
            try:
                expert_batch = next(expert_iter)
            except StopIteration:
                expert_iter = iter(loaders["train_expert"])
                expert_batch = next(expert_iter)

            exp_images = expert_batch["image"].to(DEVICE)
            exp_heatmaps = expert_batch["heatmap"].to(DEVICE)
            exp_tokens = tokenizer(expert_batch["caption"], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            exp_snippet_tokens = tokenizer(expert_batch["snippet"], padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)

            exp_out = model(exp_images, exp_tokens, heatmaps=exp_heatmaps)
            snippet_emb = F.normalize(model.encode_text(exp_snippet_tokens), dim=-1)

            exp_loss = expert_loss(
                exp_out["img_emb"], exp_out["masked_emb"],
                exp_out["txt_emb"], snippet_emb,
                model.temperature,
            )
            loss = loss + AUX_LOSS_WEIGHT * exp_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            print(f"    step {step:>4d}/{MAX_STEPS} | loss {loss.item():.4f} | temp {model.temperature.item():.4f} | time {elapsed:.1f}s")

    # Evaluate on this fold's val set
    metrics = evaluate(model, loaders["val"], tokenizer, DEVICE)
    print(f"    val mean_rank: {metrics['mean_rank']:.2f} | R@1: {metrics['img2txt_r1']:.4f} | R@5: {metrics['img2txt_r5']:.4f}")

    return metrics["mean_rank"]


# ---------------------------------------------------------------------------
# Main: 3-fold CV
# ---------------------------------------------------------------------------
def train():
    print(f"Device: {DEVICE}")
    print(f"Config: img={IMAGE_ENCODER}, txt={TEXT_ENCODER}, proj={PROJ_DIM}, bs={BATCH_SIZE}, lr={LEARNING_RATE}, steps={MAX_STEPS}")
    print(f"Running {N_FOLDS}-fold CV...")

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)

    fold_ranks = []
    for fold in range(N_FOLDS):
        rank = train_fold(fold, tokenizer)
        fold_ranks.append(rank)

    avg_rank = np.mean(fold_ranks)
    std_rank = np.std(fold_ranks)

    print("\n" + "=" * 50)
    print(f"3-Fold CV Results:")
    for i, r in enumerate(fold_ranks):
        print(f"  Fold {i}: mean_rank = {r:.2f}")
    print(f"  Average: {avg_rank:.2f} ± {std_rank:.2f}")
    print("=" * 50)

    # Report val metric for autoresearch keep/revert decision
    print(f"\n>>> val_metric (mean_rank, lower is better): {avg_rank:.4f}")
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f">>> peak_memory_gb: {peak_mem:.2f}")

    return avg_rank


if __name__ == "__main__":
    train()
