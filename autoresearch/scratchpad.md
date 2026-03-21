# Autoresearch Scratchpad

## Current Best
- **mean_rank: 183.27 ± 4.87** (3-fold CV)
- Config: LR=1e-4, BS=64, MAX_STEPS=800, WARMUP=80, EXPERT_PROB=0.8, MIXUP_ALPHA=0.3, AUX_LOSS_WEIGHT=0.1, MAX_TEMP=100, PROJ_DIM=512
- peak_memory_gb: 14.86
- Improvement from original baseline (344.68): **-161.41** (~47% reduction)

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
| 19 | PROJ_DIM=512 | 183.27 ± 4.87 | -5.74 | 14.9 | committed |
| 20 | Label smoothing=0.1 | 230.29 ± 2.24 | +47.02 | 14.9 | REVERTED |
| 21 | Pixel-space heatmap masking | 188.10 ± 3.22 | +4.83 | 18.4 | REVERTED |
| 22 | WEIGHT_DECAY=1e-3 | 183.95 ± 2.94 | +0.68 | 14.9 | REVERTED |
| 23 | Separate backbone LR=1e-5 | 289.20 ± 8.23 | +105.93 | 14.9 | REVERTED |

## Hypotheses
- ~~2-layer projectors (GELU + LayerNorm) could help~~ — tested, worse (+7.63), extra capacity worsens overfitting
- PROJ_DIM scaling: 128→256 (-7.93), 256→512 (-5.74) — more embedding capacity in projection keeps helping, diminishing returns starting
- ~~Mean pooling for text (instead of CLS) often works better for DistilBERT~~ — tested, no improvement (+0.99), CLS is fine here
- ~~Pixel-space heatmap injection~~ — tested, worse (+4.83) AND doubles expert-path memory. MHA is better than naive masking

## Next Experiments (Phase 3)
- LR=2e-4 — current 1e-4 might be too conservative now with PROJ_DIM=512
- AUX_LOSS_WEIGHT=0.2 — slightly stronger expert signal (0.1 current, 0.5 was too much)
- Gradient accumulation for effective BS=128
- Dropout in projectors to combat overfitting

## Failed Patterns
- LR=1e-3: temperature explodes, total collapse
- BS=96: exceeds 20 GB memory, overfits
- 1000 steps: overfits (train loss very low, val gets worse)
- AUX_LOSS_WEIGHT=0.5: expert loss too strong, destabilizes training
- EXPERT_PROB=1.0: too much expert, diminishing returns
- 2-layer projectors: more capacity in projector head worsens overfitting (+7.63)
- Text mean pooling: no benefit over CLS for DistilBERT in this setup
- Label smoothing=0.1: catastrophic (+47.02) — prevents model from learning sharp cross-modal distinctions
- Pixel-space heatmap masking: worse than MHA (+4.83) and uses 18.4 GB (doubles expert path compute)
- Separate backbone LR=1e-5: catastrophic (+105.93) — backbone can't adapt to art domain in 800 steps with low LR
