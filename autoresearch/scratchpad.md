# Autoresearch Scratchpad

## Current Best
- mean_rank: 344.68 ± 12.66 (3-fold CV)
- CLIP baseline (no expert): 345.91 ± 10.41
- Expert adds almost nothing yet

## Observations
- Exp #2: LR=1e-3 → mean_rank 1541.50 ± 119.05 (MUCH worse). Temperature immediately hit max clamp (2.0) and stayed there all training. Loss plateaued ~3.87. LR way too high — temperature and embeddings diverge.
- The baseline LR=3e-4 seems reasonable; 1e-3 is too aggressive for this model size/data combo.

## Hypotheses
- LR=1e-4 might undertrain in 400 steps but could be more stable. Worth trying.
- The temperature clamping at 2.0 suggests the logit scale is blowing up — may need tighter MAX_TEMPERATURE (e.g., 0.5 or 1.0).
- Batch size increase could help contrastive learning (more negatives per batch).

## Next Ideas
- Try LR=1e-4 (Phase 1.1)
- If LR=3e-4 remains best, move to batch size tuning (Phase 1.2)
- Then temperature init tuning (Phase 1.3)

## Failed Attempts
- LR=1e-3: catastrophic — temp explodes, mean_rank 1541 (4.5x worse than baseline)
