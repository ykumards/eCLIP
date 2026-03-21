# Autoresearch Scratchpad

## Current Best
- **Val mean_rank: 157.43 ± 0.76** (3-fold CV)
- Config: LR=2e-4, BS=64, MAX_STEPS=800, WARMUP=160, EXPERT_PROB=0.8, MIXUP_ALPHA=0.3, AUX_LOSS_WEIGHT=0.1, MAX_TEMP=100, PROJ_DIM=512, WISE_FT_ALPHA=0.1
- peak_memory_gb: 15.22
- Improvement from original baseline (344.68): **-187.25** (~54% reduction)

## Final Test Results (trained on all 9,893 samples)
| Metric | Score |
|--------|-------|
| Mean Rank | **34.30** |
| img→txt R@1 | 26.82% |
| img→txt R@5 | 53.00% |
| img→txt R@10 | 63.09% |
| txt→img R@1 | 25.64% |
| txt→img R@5 | 51.36% |
| txt→img R@10 | 62.45% |

Model: ViT-Small (22M) + DistilBERT (66M) + HeatmapProcessor, ~90M params total.
Trained 800 steps (~3 min) on a single 4090. Model saved to `eclip_final.pt`.

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
| WiSE-FT α=0.2 | -2.48 | Weight interpolation preserves pretrained features |
| WiSE-FT α=0.1 | -1.02 | Lighter interpolation slightly better than 0.2 |

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
Expert contributes ~5 mean_rank points with current config (162.37→157.43, exp #40). Earlier measurement was ~4 points (exp #10). Consistently helpful but modest. The MHA-based heatmap processor outperforms pixel-space masking and heatmap-weighted pooling. The mechanism is limited by the quality of pseudo bounding boxes (GroundingDINO-generated, not human-annotated).

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
| 33 | SigLIP loss + learnable bias | 190.62 ± 12.01 | +29.69 | 14.9 | REVERTED |
| 34 | WiSE-FT alpha=0.5 | 206.88 ± 5.10 | +45.95 | 15.2 | REVERTED (alpha too high) |
| 35 | WiSE-FT alpha=0.2 | 158.45 ± 0.68 | -2.48 | 15.2 | committed |
| 36 | WiSE-FT alpha=0.1 | 157.43 ± 0.76 | -1.02 | 15.2 | committed |
| 37 | EMA decay=0.999 | 242.63 ± 8.44 | +85.20 | 15.6 | REVERTED (too much smoothing for 800 steps) |
| 38 | Heatmap-weighted pooling (replace MHA) | 168.48 ± 2.15 | +11.05 | 14.9 | REVERTED (MHA is more expressive) |
| 39 | Freeze ViT blocks 0-7 | 212.00 ± 3.27 | +54.57 | 10.1 | REVERTED (backbone needs to adapt photo→art) |
| 40 | EXPERT_PROB=0 diagnostic | 162.37 ± 2.10 | +4.94 | 8.2 | REVERTED (diagnostic: expert adds ~5 pts with current config) |
| 41 | Spatial prompt tokens (49 tokens, 7x7) | 167.65 ± 0.91 | +10.22 | 19.7 | REVERTED (worse than MHA, near OOM) |
| 42 | Attention bias (scale=2.0, no HeatmapProcessor) | 166.85 ± 1.35 | +9.42 | 19.4 | REVERTED (worse than MHA) |

## Next Experiments (Phase 5 — "Out There" Changes)

Phase 1–4 exhausted hyperparameter tuning and simple architecture variants. The remaining gains require fundamentally different approaches.

### Validated ideas (from experiments so far)
- ~~**WiSE-FT alpha tuning**: α=0.1 committed (-1.02). α=0.2 and 0.1 both work; 0.1 is slightly better. Diminishing returns — skip α=0.05.~~
- ~~**Heatmap-weighted pooling**: Tested (#38, +11.05). MHA cross-attention is more expressive than simple weighted pooling.~~
- ~~**EMA**: Tested (#37, +85.20). Too much smoothing for 800 steps.~~
- ~~**Freeze early ViT blocks**: Tested (#39, +54.57). Backbone needs full fine-tuning to adapt photo→art.~~
- ~~**Expert utility check**: Tested (#40). Expert adds ~5 pts (162.37→157.43) with current config.~~

### Remaining experiments (final two) — DONE
- ~~**Spatial prompt tokens** (#41, +10.22): Prepended 49 heatmap tokens to ViT input. Near OOM (19.7 GB), worse than MHA.~~
- ~~**Attention bias** (#42, +9.42): Injected heatmap as additive bias in ViT self-attention. No new params but worse than dedicated MHA.~~

### Key conclusion from Phase 5
All three alternative expert fusion approaches (heatmap-weighted pooling #38, spatial prompt tokens #41, attention bias #42) performed ~10 points worse than the MHA-based HeatmapProcessor. The dedicated cross-attention module with learned query/key/value projections is the right design for this task. The MHA approach can learn complex spatial relationships that simpler alternatives cannot match.

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
- WiSE-FT alpha=0.5: too aggressive, destroys learned representations (+45.95)
- SigLIP loss: sigmoid contrastive loss much worse (+29.69), high variance
- EMA 0.999: catastrophic (+85.20). With 800 steps, ~45% weight still on initial params. Another regularization failure.
- Heatmap-weighted pooling: simpler but worse (+11.05). MHA cross-attention learns richer spatial relationships than simple weighted pooling.
- Freeze ViT blocks 0-7: catastrophic (+54.57). Backbone needs to adapt from photo domain to art domain.
- Spatial prompt tokens: worse (+10.22) and near OOM (19.7 GB). Extra tokens in ViT too expensive.
- Attention bias: worse (+9.42) despite being parameter-free. Biasing existing attention is weaker than dedicated MHA.
- All Phase 5 expert fusion alternatives lost to the MHA HeatmapProcessor by ~10 points.
