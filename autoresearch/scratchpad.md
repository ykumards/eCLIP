# Autoresearch Scratchpad

## Current Best
- **mean_rank: 189.01 ± 0.84** (3-fold CV)
- Config: LR=1e-4, BS=64, MAX_STEPS=800, WARMUP=80, EXPERT_PROB=0.8, MIXUP_ALPHA=0.3, AUX_LOSS_WEIGHT=0.1, MAX_TEMP=100, PROJ_DIM=256
- peak_memory_gb: 14.67
- Improvement from original baseline (344.68): **-155.67** (~45% reduction)

## Key Findings

### Temperature was the biggest win
The original MAX_TEMPERATURE=2.0 was catastrophically wrong. The init `log(1/0.07) ≈ 2.66` → `exp(2.66) ≈ 14.3` gets clamped to 2.0 immediately, so the temperature was never learned. Fixing to MAX_TEMPERATURE=100 gave **-113 mean_rank** in one change. Temperature now learns to ~14.3-14.9 (standard CLIP range).

### Expert mechanism IS helpful (but modestly)
With proper temperature, EXPERT_PROB=0 gives 203.21, EXPERT_PROB=0.8 gives 196.94. Expert contributes ~6 points. 0.3 and 0.5 were similar (~199), 0.8 is best, 1.0 hurts (202.87).

### Training dynamics
- Loss reaches 0.03-0.07 by step 800 — borderline overfitting
- 800 steps is sweet spot; 1000 steps overfits (206.25 vs 199.21)
- 10% warmup ratio works well; 5% slightly worse

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

## Hypotheses
- ~~2-layer projectors (GELU + LayerNorm) could help~~ — tested, worse (+7.63), extra capacity worsens overfitting
- PROJ_DIM=256 confirmed helpful (-7.93, extremely low variance 0.84) — more embedding capacity without adding model params
- ~~Mean pooling for text (instead of CLS) often works better for DistilBERT~~ — tested, no improvement (+0.99), CLS is fine here
- Pixel-space heatmap injection (image * heatmap → backbone) might capture spatial info better than MHA

## Next Experiments (Phase 3)
- PROJ_DIM=512 — push further since 256 helped significantly
- Heatmap injection: pixel-space masking vs current MHA approach
- Label smoothing in contrastive loss
- Gradient accumulation for effective BS=128

## Failed Patterns
- LR=1e-3: temperature explodes, total collapse
- BS=96: exceeds 20 GB memory, overfits
- 1000 steps: overfits (train loss very low, val gets worse)
- AUX_LOSS_WEIGHT=0.5: expert loss too strong, destabilizes training
- EXPERT_PROB=1.0: too much expert, diminishing returns
- 2-layer projectors: more capacity in projector head worsens overfitting (+7.63)
- Text mean pooling: no benefit over CLS for DistilBERT in this setup
