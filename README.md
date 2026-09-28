# Part Discriminability for Domain-Generalizable Person Re-ID

Experiments investigating whether per-body-part discriminability and reliability statistics,
measured from a source-domain ReID model, can improve transfer to unseen domains.

## Hypothesis

Re-ID transformers latch onto loud, instance-specific cues (bags, held objects, unusual clothing)
that vary across source and target domains — a *cue-vocabulary shift* distinct from the low-level
style shift that normalization and augmentation target. An anatomical prior over body parts could
regularize against this as a *fallback* when salient cues are absent.

## Repo structure

```
experiments/          Paper-specific analysis scripts
  compute_fk_bot.py       F_k per keypoint slot, BoT ResNet-50 (6 paired slots)
  compute_fk_bot_e2b.py   Same with L2-normalized features (E2b)
  compute_fk_bot_n12.py   Same with 12 individual (unmerged L/R) slots
  compute_e8.py           E8: F_k concentration vs. transfer gap across checkpoints
  compute_fisher.py       Color-feature Fisher ratios (PACM features)
  attention_e3b.py        E3b: per-image attention concentration analysis
  bootstrap_map.py        Bootstrap CI for mAP

results/              Cached outputs
  fk_results_bot.json     BoT F_k, Market-1501 + MSMT17 (n=6 paired)
  fk_results_bot_e2b.json Same, L2-normalized features
  fk_detailed.json        Color-feature F_k
```

## Dependencies

```
pip install -e /workspace/reid-tk   # or: sys.path.insert(0, '/workspace/reid-tk')
```

All scripts assume reid-tk is importable and datasets are at `/data/datasets/reid/`.
Pretrained checkpoints at `/data/papers/low_dimensional_reid/pretrained/`.
ViTPose keypoint cache at `/tmp/pacm_cache_vitpose/keypoints/`.

## Key findings so far

| Result | Finding |
|--------|---------|
| R5 (color F_k) | Shoulder tops color discriminability ranking; Wrist lowest |
| R6 (deep F_k, BoT) | Knee tops in-domain (Market); Hip tops cross-domain (MSMT17) |
| R8 (E2b, normalized) | Scale artifact confirmed: Wrist raw 1/S_W was inflated. Normalized 1/S_W: Hi > Kn > El > Sh > Wr > An |
| E4 stability | ρ(Market F_k, MSMT17 F_k) = 0.714, n=6, p≈0.11 — gate still open |
| E8 (running) | Correlating F_k concentration with transfer gap across 5 checkpoints |

## Experiment status

See the living design doc: https://claude.ai/code/artifact/c4173ec6-1500-44cc-9d35-88a3b4a3b8f1

Current priority: **E8** (concentration vs. transfer gap) → **E7** (differential-region suppression)
H1 training held pending E4 gate and E2b/E8 results.
