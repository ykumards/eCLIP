# eCLIP Autoresearch Program

## Context

**eCLIP** was published at ECCV 2024 for medical imaging. The core idea: standard CLIP learns global image↔text alignment, but with a small set of *expert spatial annotations* (heatmaps showing where an expert focused + corresponding text snippets), we can teach the model finer-grained region↔phrase alignment.

We're adapting this to **Ukiyo-e Japanese woodblock prints** — a dataset of ~11K prints with bounding box annotations linking image regions to descriptive phrases. The bounding boxes are converted to gaussian heatmaps.

The goal is to find what works through rapid automated experimentation.

## Research Question

**Does injecting spatial expert annotations (bounding box → heatmap) improve CLIP-style image-text retrieval on art images, and what's the best way to do it?**

Sub-questions the experiments should explore:
- Is the expert heatmap mechanism actually helpful, or does vanilla CLIP fine-tuning suffice?
- What's the best way to inject spatial information — pixel-space masking, feature-space attention, input multiplication, or something else?
- How much does the expert scheduling matter (cold start → warmup → cooldown vs fixed probability)?
- Can we get meaningful improvements in just 400-1000 training steps?

## Data

- **11K Ukiyo-e prints** with titles + cleaned descriptions (GPT-4 cleaned)
- **Train**: 8,794 images with pseudo bounding boxes (GroundingDINO-generated) + captions
- **Val**: 1,099 images with pseudo bounding boxes + captions
- **Test**: 1,100 images with 3,880 **manually annotated** phrase→bbox pairs
- Each sample has: image, caption (full description), heatmap (gaussian from bboxes), snippet (single phrase from one region)
- Images are ~300-400px Japanese woodblock prints — stylized art, NOT photographs

## Key Metric

**`mean_rank`** (lower is better) — average position of the correct match in image↔text retrieval on the validation set.

- Random baseline: ~549 (half of 1,099 val samples)
- Anything under 400 is showing learning signal
- Under 200 would be solid
- Under 50 would be impressive

Also report `peak_memory_gb` — must stay under 20 GB (4090 has 24 GB).

## Files

- `prepare.py` — **DO NOT MODIFY**. Data loading, heatmap generation, evaluation, dataloaders.
- `train.py` — **MODIFY THIS**. Model architecture, loss functions, training loop, hyperparameters.

## Current Baseline Architecture

- **Image encoder**: ViT-Small (22M params, timm, ImageNet-pretrained) → linear projector → L2-norm
- **Text encoder**: DistilBERT (66M params, HuggingFace) → CLS pooling → linear projector → L2-norm
- **Heatmap processor**: Conv2d patchify → multi-head cross-attention (heatmap queries, image keys/values) → mean pool
- **Loss**: Symmetric CLIP contrastive with learnable temperature
- **Expert loss**: Mixup of original + masked embeddings, contrastive with snippet text
- **Training**: AdamW, cosine LR with warmup, 400 steps, batch size 48

## Experiment Priorities

Try these roughly in order. Each experiment should change ONE thing.

### Phase 1 — Get the baseline working well
1. Tune learning rate (try 1e-4, 3e-4, 1e-3)
2. Tune batch size (32, 48, 64, 96)
3. Tune temperature init (0.07, 0.1, 0.01) and clamp range
4. Try more steps (400 → 800 → 1000) if time budget allows
5. Try warmup ratio (5%, 10%, 20% of steps)

### Phase 2 — Expert mechanism
6. Compare: expert OFF (EXPERT_PROB=0) vs ON — does it help at all?
7. Try different EXPERT_PROB values (0.1, 0.3, 0.5, 0.8, 1.0)
8. Implement phased scheduling (cold start 10% → warmup to 0.5 → cooldown to 0.1)
9. Try different mixup alpha values (0.1, 0.3, 0.5, 0.8)
10. Add reconstruction loss during cold start (MSE: heatmap_processor(image, ones) ≈ image)

### Phase 3 — Architecture experiments
11. Different heatmap injection: pixel-space masking (image * heatmap → backbone) instead of MHA
12. Feature-space attention: use heatmap to weight ViT patch tokens after backbone
13. Input concatenation: cat(image, heatmap) with modified first conv layer
14. Projection layers: 1-layer vs 2-layer with GELU + LayerNorm
15. Text pooling: CLS vs mean pooling vs max pooling
16. Projection dim: 128 vs 256 vs 512

### Phase 4 — Advanced
17. Separate LR for image encoder, text encoder, heatmap processor
18. Freeze backbone, only train projectors + heatmap processor
19. Use a pretrained CLIP model as backbone instead of separate ViT + DistilBERT
20. Gradient accumulation (effective batch size 128, 256)
21. Label smoothing in contrastive loss
22. Hard negative mining (semi-hard or hardest negatives)
23. Data augmentation: random resized crop, color jitter on images

### Phase 5 — Expert fusion architecture (better ways to inject spatial knowledge)
The current HeatmapProcessor is a bolt-on: the heatmap info never touches the ViT backbone.
These alternatives fuse expert spatial annotations more deeply into the model:

24. **Heatmap-weighted pooling** — replace HeatmapProcessor entirely. Instead of CLS token,
    pool ViT patch tokens weighted by heatmap values. The expert knowledge becomes *how to
    read* the image, not a separate path:
    ```python
    heatmap_down = F.adaptive_avg_pool2d(heatmap, (14, 14)).flatten(1)  # (B, 196)
    weights = F.softmax(heatmap_down, dim=-1)
    patch_tokens = backbone.forward_features(image)[:, 1:]  # skip CLS, (B, 196, D)
    pooled = (patch_tokens * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
    ```
25. **Attention bias** — use the heatmap to bias ViT self-attention directly. Patches inside
    bounding boxes get boosted attention. No new parameters:
    ```python
    attn_scores = Q @ K.T / sqrt(d) + heatmap_bias  # heatmap_bias from downsampled heatmap
    ```
    Note: requires accessing ViT internals (timm's `forward_features` or custom forward).
26. **Spatial prompt tokens** (inspired by Voila-A) — patchify heatmap, project to ViT
    embedding dim, prepend as extra tokens to ViT input. Backbone learns to attend to spatial
    hints naturally. Adds a small projection layer but no backbone changes.
27. **WiSE-FT** — after training, interpolate pretrained and fine-tuned weights:
    `final = α * pretrained + (1-α) * finetuned`. One-line experiment, often helps on small
    datasets by preserving pretrained features. Try α in [0.2, 0.4, 0.6, 0.8].

## Vibe

This is a single 4090, not a cluster. Push hard on the metric, but keep the code simple and scrappy. A clean 50-line change that drops mean_rank by 10 is better than a 300-line refactor that drops it by 12. Don't over-engineer — no custom CUDA kernels, no distributed training, no elaborate config systems. Just edit train.py, run it, check the number.

## What NOT to do

- **Don't make multiple changes at once.** One change per experiment so we know what helped.
- **Don't increase steps beyond 1500.** Keep experiments under 10 minutes.
- **Don't change the eval metric.** Always report mean_rank from `evaluate()` in prepare.py.
- **Don't add new dependencies** unless absolutely necessary.
- **Don't create new files.** Everything goes in train.py.
- **Don't make the model huge.** If peak_memory_gb > 20, the experiment fails.
- **Don't use FP64.** Use FP32 or mixed precision (FP16/BF16).
- **Don't over-engineer.** No abstractions, no config classes, no elaborate logging. Keep train.py readable in one sitting.

## How experiments work

1. Read this file and `train.py` fully before making any changes.
2. Decide on ONE focused change based on the priorities above.
3. Edit `train.py` with the change.
4. Run: `./autoresearch/run.sh` to execute in Docker sandbox.
5. Check the output for `val_metric` and `peak_memory_gb`.
6. **If val_metric improved** → commit with structured message (see format below).
7. **If val_metric did NOT improve** → `git checkout -- autoresearch/train.py` to revert.
8. Update `autoresearch/scratchpad.md` with observations, hypotheses, and next plans.
9. Repeat from step 2.

### Commit message format

Use this exact format so results can be parsed programmatically:

```
experiment: <short description>

mean_rank: <new> ± <std> (prev: <old> ± <std>)
delta: <signed change, negative is better>
peak_memory_gb: <value>
folds: [<f0>, <f1>, <f2>]
phase: <1|2|3|4>
changed: <param_name>=<new_value> (was <old_value>)
```

Example:
```
experiment: increase learning rate to 1e-3

mean_rank: 312.45 ± 8.20 (prev: 344.68 ± 12.66)
delta: -32.23
peak_memory_gb: 11.4
folds: [305.12, 318.91, 313.32]
phase: 1
changed: LEARNING_RATE=1e-3 (was 3e-4)
```

## Experiment Log

Track results here as they come in:

```
| # | Change | mean_rank | peak_mem | Status |
|---|--------|-----------|----------|--------|
| 0 | CLIP baseline (no expert, EXPERT_PROB=0) | 345.91 ± 10.41 | 6.3 GB | reference |
| 1 | eCLIP baseline (EXPERT_PROB=0.3) | 344.68 ± 12.66 | 11.3 GB | committed |
```

Note: Expert mechanism currently adds almost no benefit over vanilla CLIP.
The agent should aim to make expert annotations clearly helpful (beat CLIP baseline by >10 mean_rank).
