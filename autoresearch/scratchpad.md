# Autoresearch Scratchpad

## Current Best
- **mean_rank: 160.93 ± 1.33** (3-fold CV)
- Config: LR=2e-4, BS=64, MAX_STEPS=800, WARMUP=160, EXPERT_PROB=0.8, MIXUP_ALPHA=0.3, AUX_LOSS_WEIGHT=0.1, MAX_TEMP=100, PROJ_DIM=512
- peak_memory_gb: 14.86
- Improvement from original baseline (344.68): **-183.75** (~53% reduction)

## Phase 1–4 Summary (32 experiments)

### What worked (committed changes, cumulative -183.75)
| Change | delta | Insight |
|--------|-------|---------|
| LR 3e-4 → 1e-4 | -20.07 | Original LR too high for this data size |
| MAX_TEMP 2 → 100 | -113.19 | **Biggest win.** Temp was clamped, never learned. |
| BS 48 → 64 | -4.66 | More negatives per contrastive matrix |
| Steps 400 → 800 | -7.55 | More training helps (but 1000+ overfits) |
| EXPERT_PROB 0.3 → 0.8 | -2.27 | Expert mechanism helps modestly |
| PROJ_DIM 128 → 256 → 512 | -13.67 | Linear projection capacity matters |
| LR 1e-4 → 2e-4 | -19.41 | Re-tuned after PROJ_DIM=512; 2e-4 is sweet spot |
| WARMUP 80 → 160 (20%) | -2.93 | Stabilizes early training with higher LR |

### What didn't work (reverted, 22 experiments)
- **Regularization fails**: Label smoothing (+47), dropout (+10), weight decay 1e-3 (+0.7), cosine min LR (+4.5) — all hurt. The model needs every bit of capacity in 800 steps.
- **Architecture changes fail**: 2-layer projectors (+7.6), text mean pooling (+1), pixel-space heatmap masking (+4.8) — the simple architecture is already well-suited.
- **More training fails**: 1000 steps (+7), 1200 steps (+16) — overfitting is the hard constraint.
- **LR extremes fail**: 1e-3 (collapse), 3e-4 (+6.9, unstable), backbone LR=1e-5 (+106, too slow).
- **Expert tuning marginal**: AUX_WEIGHT=0.2 (+2.2), AUX_WEIGHT=0.5 (+7.7), MIXUP_ALPHA=0.1 (+2), MIXUP_ALPHA=0.8 (+1.1) — 0.3/0.1 balance is already optimal.
- **Gradient accumulation**: Proper implementation with shared contrastive matrix, but 400 steps too few (+22). 800 steps would double training time.

### Key insight: the overfitting wall
The model consistently overfits between 800–1000 steps regardless of regularization. The fundamental limit is **data size** (~6.6K train samples per fold). Every approach that adds capacity (more layers, more steps, keeping LR high) makes this worse. The improvements that worked all operate *within* this constraint: better use of existing capacity (temperature fix, PROJ_DIM, LR tuning) rather than adding more.

### What the expert mechanism does
Expert contributes ~6–10 mean_rank points (comparing EXPERT_PROB=0 vs 0.8 across configs). It's consistently helpful but modest. The MHA-based heatmap processor outperforms pixel-space masking. The mechanism is limited by the quality of pseudo bounding boxes (GroundingDINO-generated, not human-annotated).

## Experiment Log

| # | Change | mean_rank | delta | peak_mem | Status |
|---|--------|-----------|-------|----------|--------|
| 0 | CLIP baseline (EXPERT_PROB=0, old config) | 345.91 ± 10.41 | — | 6.3 | reference |
| 1 | eCLIP baseline (EXPERT_PROB=0.3, old config) | 344.68 ± 12.66 | — | 11.3 | reference |
| 2 | LR=1e-3 | 1541.50 ± 119.05 | +1196.82 | 11.3 | REVERTED |
| 3 | LR=1e-4 | 324.61 ± 4.18 | -20.07 | 11.3 | committed |
| 4 | MAX_TEMPERATURE=100 | 211.42 ± 5.23 | -113.19 | 11.3 | committed |
| 5 | BS=64 | 206.76 ± 6.59 | -4.66 | 14.4 | committed |
| 6 | BS=96 | 218.40 ± 9.45 | +11.64 | 21.3 | REVERTED (over mem limit) |
| 7 | MAX_STEPS=800, WARMUP=80 | 199.21 ± 6.06 | -7.55 | 14.6 | committed |
| 8 | MAX_STEPS=1000, WARMUP=100 | 206.25 ± 7.88 | +7.04 | 14.6 | REVERTED |
| 9 | WARMUP=40 (5%) | 200.53 ± 5.92 | +1.32 | 14.6 | REVERTED |
| 10 | EXPERT_PROB=0.0 | 203.21 ± 6.63 | +3.99 | 7.9 | REVERTED (baseline test) |
| 11 | EXPERT_PROB=0.5 | 199.38 ± 2.99 | +0.17 | 14.7 | REVERTED |
| 12 | EXPERT_PROB=0.8 | 196.94 ± 4.99 | -2.27 | 14.6 | committed |
| 13 | EXPERT_PROB=1.0 | 202.87 ± 4.90 | +5.93 | 14.6 | REVERTED |
| 14 | AUX_LOSS_WEIGHT=0.5 | 204.62 ± 6.88 | +7.68 | 14.6 | REVERTED |
| 15 | MIXUP_ALPHA=0.8 | 198.08 ± 5.89 | +1.14 | 14.6 | REVERTED |
| 16 | Text mean pooling (vs CLS) | 197.93 ± 4.04 | +0.99 | 14.6 | REVERTED |
| 17 | 2-layer projectors (GELU+LN) | 204.57 ± 6.29 | +7.63 | 14.6 | REVERTED |
| 18 | PROJ_DIM=256 | 189.01 ± 0.84 | -7.93 | 14.7 | committed |
| 19 | PROJ_DIM=512 | 183.27 ± 4.87 | -5.74 | 14.9 | committed |
| 20 | Label smoothing=0.1 | 230.29 ± 2.24 | +47.02 | 14.9 | REVERTED |
| 21 | Pixel-space heatmap masking | 188.10 ± 3.22 | +4.83 | 18.4 | REVERTED |
| 22 | WEIGHT_DECAY=1e-3 | 183.95 ± 2.94 | +0.68 | 14.9 | REVERTED |
| 23 | Separate backbone LR=1e-5 | 289.20 ± 8.23 | +105.93 | 14.9 | REVERTED |
| 24 | LR=2e-4 | 163.86 ± 3.85 | -19.41 | 14.9 | committed |
| 25 | LR=3e-4 | 170.73 ± 6.52 | +6.87 | 14.9 | REVERTED |
| 26 | AUX_LOSS_WEIGHT=0.2 | 166.05 ± 5.71 | +2.19 | 14.9 | REVERTED |
| 27 | WARMUP_STEPS=160 (20%) | 160.93 ± 1.33 | -2.93 | 14.9 | committed |
| 28 | MAX_STEPS=1200, WARMUP=240 | 176.78 ± 3.25 | +15.85 | 14.9 | REVERTED |
| 29 | Projector dropout=0.1 | 171.17 ± 5.01 | +10.24 | 14.8 | REVERTED |
| 30 | Cosine min LR=5% floor | 165.45 ± 1.83 | +4.52 | 14.9 | REVERTED |
| 31 | MIXUP_ALPHA=0.1 | 162.95 ± 0.86 | +2.02 | 14.9 | REVERTED |
| 32 | Grad accum (eff. BS=128, 400 steps) | 182.83 ± 4.12 | +21.90 | 8.6 | REVERTED |

## Next Experiments (Phase 5 — "Out There" Changes)

Phase 1–4 exhausted hyperparameter tuning and simple architecture variants. The remaining gains require fundamentally different approaches:

- **Pretrained CLIP backbone**: Use OpenAI CLIP ViT-B/32 or similar as the image+text encoder instead of separate ViT-S + DistilBERT. Starts with cross-modal alignment already learned.
- **ImageNet-21K ViT**: Use `vit_small_patch16_224.augreg_in21k` — better pretrained features for fine-tuning on small datasets.
- **Feature-space heatmap attention**: Extract ViT patch tokens and weight by heatmap spatial map, instead of separate Conv2d patchification. Reuses backbone features for expert path.
- **Hard negative mining**: Sample hard negatives from the batch (semi-hard or hardest) for the contrastive loss. Could give stronger gradients.
- **Multi-scale image features**: Use intermediate ViT layer outputs in addition to final layer.
- **Frozen backbone + train only projectors/heatmap**: With 6.6K samples, the backbone might not need fine-tuning if using a stronger pretrained model.

## Failed Patterns
- LR=1e-3: temperature explodes, total collapse
- BS=96: exceeds 20 GB memory, overfits
- 1000+ steps: overfits regardless of config
- AUX_LOSS_WEIGHT=0.5: expert loss too strong, destabilizes training
- EXPERT_PROB=1.0: too much expert, diminishing returns
- 2-layer projectors: more capacity in projector head worsens overfitting
- Text mean pooling: no benefit over CLS for DistilBERT
- Label smoothing: catastrophic — prevents sharp cross-modal distinctions
- Pixel-space heatmap masking: worse than MHA, doubles memory
- Separate backbone LR: backbone can't adapt in 800 steps at 1e-5
- LR=3e-4: too aggressive, unstable
- Dropout/regularization: hurts convergence within 800 step budget
- Gradient accumulation: proper implementation works but halved steps hurt more than larger batch helps
