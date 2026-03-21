# Autoresearch Scratchpad

## Current Best
- mean_rank: 344.68 ± 12.66 (3-fold CV)
- CLIP baseline (no expert): 345.91 ± 10.41
- Expert adds almost nothing yet

## Observations
- Exp #2: LR=1e-3 → mean_rank 1541.50 ± 119.05 (MUCH worse). Temperature hit max clamp (2.0), loss plateaued ~3.87. LR way too high.
- Exp #3: LR=1e-4 → mean_rank 324.61 ± 4.18 (prev 344.68). Modest improvement, more stable. But temp still clamped at 2.0.
- Exp #4: MAX_TEMPERATURE=100 → mean_rank 211.42 ± 5.23 (prev 324.61). **HUGE win** (-113). Temperature now properly learned at ~14.3-14.4. Loss drops to 0.5-0.8 by step 400. The original MAX_TEMP=2.0 was catastrophically restrictive — it prevented the standard CLIP temperature scaling from working.

- Exp #5: BS=64 → mean_rank 206.76 ± 6.59 (prev 211.42). Modest improvement. Memory 14.39 GB. More negatives helps.

## Hypotheses
- BS=96 might help further but memory could approach 20 GB limit.
- More steps (800) with current settings could push much lower — loss is still dropping at step 400.
- Expert mechanism may now actually help given the temperature fix.

## Next Ideas
- Try LR=1e-4 (Phase 1.1)
- If LR=3e-4 remains best, move to batch size tuning (Phase 1.2)
- Then temperature init tuning (Phase 1.3)

## Failed Attempts
- LR=1e-3: catastrophic — temp explodes, mean_rank 1541 (4.5x worse than baseline)
