# The "Part-Aligned Color Matching" Hypothesis in Person Re-Identification: Proposal and Results

**Status (2026-09-17):** Market-1501 arm complete, hypothesis rejected. §§1–4 record the pre-registered design as written; §5 reports results; §6 revises the significance claims in light of them. MSMT17 and the PRCC/LTCC arm were not run.

## 1. Objective

The primary objective of this research is to evaluate the extent to which state-of-the-art (SOTA) Person Re-Identification (Re-ID) models rely on implicit part-aligned color/texture matching. By constructing a baseline that explicitly uses pose-driven local color descriptors, we aim to determine whether modern, complex latent-embedding models provide significant improvements over a well-calibrated, interpretable, "dumb" feature-matching approach.

## 2. Research Hypothesis

> Current deep learning-based Re-ID models perform primarily as high-dimensional, implicit part-aligned color/texture matchers.

Therefore, an explicit model that extracts local color/texture features from canonical keypoints and performs alignment using Optimal Transport (OT) or Dynamic Time Warping (DTW) will achieve performance within a bounded margin of SOTA baselines on standard Re-ID benchmarks.

### Success Criterion

The hypothesis is **supported** if the explicit model lands within **5 mAP points and 3 Rank-1 points** of the SOTA baseline on Market-1501 and MSMT17. It is **rejected** if the gap exceeds that on either dataset.

A second, separable prediction follows from the first: if the baseline's advantage is largely color-driven, that advantage should *widen sharply* on clothes-changing benchmarks — where color ceases to be identity-bearing for both models. Concretely, we predict the SOTA baseline's margin over the explicit model grows by **less than 5 mAP points** moving from Market-1501 to PRCC/LTCC. A margin that grows substantially more indicates the deep model encodes identity signal beyond part-aligned color, and the hypothesis fails in its strong form.

## 3. Proposed Methodology

### A. Preprocessing and Keypoint Extraction

- **Pose estimation:** Utilize a high-performance pose estimator (e.g., HRNet or ViTPose) to extract 17–25 standard body keypoints (e.g., neck, shoulders, elbows, hips, knees).
- **Gating mechanism:** Implement a confidence threshold $\tau$ based on the pose estimator's output. Nodes with a confidence score $S_k < \tau$ are flagged as occluded and excluded from the matching process to prevent background noise contamination.

### B. Feature Extraction

- **Color/texture descriptors:** For each valid keypoint, crop a small $N \times N$ patch.
- **Representation:** Extract histograms or mean vectors in a robust color space (e.g., LAB, to separate luminosity from chromaticity) and potentially include Local Binary Patterns (LBP) to capture local texture.
- **Topology:** Maintain normalized $(x, y)$ coordinates to preserve the skeleton's spatial structure.

### C. Matching Metric

- **Distance function:** Utilize the Earth Mover's Distance (EMD) or DTW to compute the cost of aligning the probe skeleton graph against the gallery skeleton graph.
- **Optimization:** The objective is to minimize the total cost of matching local descriptors $d_i$ at keypoints $k_i$ across the two images, subject to spatial constraints.

## 4. Experimental Plan

| Phase | Task | Datasets |
| --- | --- | --- |
| Baseline | Train a standard SOTA Re-ID baseline (e.g., OSNet or a Vision Transformer). | Market-1501, MSMT17 |
| Ablation | Execute the "Part-Aligned Color" model (proposed method). | Market-1501, MSMT17 |
| Evaluation | Measure Rank-1 accuracy and mAP for both models; evaluate against the success criterion in §2. | Market-1501, MSMT17 |
| Disambiguation | Re-evaluate both models where clothing color is no longer identity-bearing, to separate "models are color matchers" from "these datasets are color-solvable." | PRCC, LTCC |
| Analysis | Perform a gap analysis across all four datasets, comparing metrics and identifying failure modes. | All |

### Note on the Disambiguation Phase

Market-1501 and MSMT17 alone cannot separate the two competing explanations. Strong performance by the explicit model on both is equally consistent with (a) SOTA models functioning as implicit color matchers and (b) these particular benchmarks being solvable by color regardless of what the models encode. The clothes-changing datasets break the tie: under (a), both models degrade comparably; under (b), the deep model retains signal the explicit model cannot access.

Both models are evaluated zero-shot on PRCC and LTCC in the primary comparison — no retraining on clothes-changing data — so that the measured gap reflects representational content rather than fitting capacity. A retrained variant may be reported separately.

> **Superseded (2026-09-17):** the Market-1501 result in §5 makes this phase uninformative as designed. With the explicit model at 8.6% mAP, its degradation on PRCC/LTCC carries no signal about dataset bias. The replacement test — color ablation applied to the SOTA model itself — is described in §6.

## 5. Experimental Results (Market-1501, 2026-09-16/17)

### Setup

| Component | Configuration |
| --- | --- |
| Pose | ViTPose-base-simple, top-down on 64×128 chips (letterboxed to 256×192), $\tau = 0.3$ |
| Matching | Index-based (anatomical alignment), L2 distance, coverage penalty (unshared slots → fallback 1.0), L/R chirality flip (element-wise min) |
| Patch | 10 px Gaussian-weighted patch per keypoint |
| Slot weights | Fisher criterion (between-ID / within-ID distance), computed on the training split (751 IDs, disjoint from test) |

### Ablation Table

| Feature | Dims | mAP | Rank-1 | Rank-5 |
|---|---|---|---|---|
| hue_point | 1 | 2.3% | 5.8% | 13.5% |
| hue_mean | 1 | 2.5% | 4.8% | 13.3% |
| hue_gaussian | 1 | 2.5% | 4.5% | 13.8% |
| hsv_histogram | 16 | 4.5% | 10.7% | 23.1% |
| lab_mean | 3 | 6.0% | 15.1% | 31.5% |
| lab_histogram | 48 | 6.8% | **20.2%** | 36.9% |
| lab_histogram_lbp | 64 | 6.5% | 19.5% | **37.6%** |
| **hs_mean** (Gaussian-weighted, Fisher slots) | **2** | **8.6%** | 19.7% | 37.0% |

Reference: SOTA (OSNet / TransReID) on Market-1501 is ~85–91% mAP, ~93–95% Rank-1.

> **Note on comparability:** the `hs_mean` row differs from the rows above it on more than the descriptor — it also carries Fisher slot weighting. Its margin over `lab_histogram` is therefore not attributable to the descriptor alone. A clean attribution requires running `lab_histogram` and `lab_mean` under Fisher weighting as well.

### Verdict: Hypothesis Rejected

The best PACM variant reaches **8.6% mAP / 19.7% Rank-1**, roughly 76–83 mAP points below SOTA — far outside the ±5 mAP success criterion set in §2. Explicit part-aligned color matching, as implemented here, is not competitive with SOTA Re-ID.

### Scope of the Rejection

The result falsifies the operational claim (an explicit part-aligned color matcher lands within a bounded margin of SOTA) but does not by itself falsify the underlying claim in §2 (that SOTA models function primarily as implicit part-aligned color matchers). A deep network could be doing largely color-driven matching while still vastly outperforming a 2-dimensional hand-crafted descriptor, by virtue of higher-dimensional color representation, denser spatial sampling, and learned illumination invariance. Failure mode 2 below — cross-camera appearance shift — is precisely the nuisance a learned embedding is trained to absorb, and absorbing it is compatible with the representation remaining color-dominated.

Distinguishing these requires an intervention on the SOTA model rather than a competing baseline. See §6.

### Key Failure Modes Identified

1. **Background contamination:** 10 px patches on 64×128 chips frequently straddle body/background boundaries, especially lower-body keypoints near bikes, walls, and ground.
2. **Cross-camera appearance variation:** The same person's LAB values at a given keypoint shift substantially across cameras (lighting, viewpoint, floor reflections). Same-ID within-distance is only modestly lower than between-ID distance, leaving little margin for discrimination.
3. **Achromatic clothing:** Grey/black clothing produces near-zero saturation, leaving hue undefined. The `hs_mean` descriptor handles this structurally (achromatic → near-origin); LAB-based modes do not.
4. **Uneven keypoint discriminability:** Fisher analysis on the training split gave shoulders ≈ 2.4 against wrists ≈ 1.1. Equal slot weighting discards usable signal.
5. **Sparse-detection free-riding:** Gallery images with few detected keypoints could score a low distance from a single favourable slot — addressed by the coverage penalty.

### Best Descriptor: hs_mean

Each keypoint patch is summarised as a 2D Gaussian-weighted colour centroid in polar→Cartesian HS space: $(s \cos h,\; s \sin h)$. Achromatic pixels collapse near the origin regardless of hue noise; saturated pixels pull toward their hue direction. This is more lighting-invariant than LAB histograms and handles achromatic clothing without a special case. Combined with Fisher-weighted slot averaging and L/R chirality flip, it is the best configuration found **by mAP**; `lab_histogram` remains marginally ahead on Rank-1 and `lab_histogram_lbp` on Rank-5.

## 6. Significance

The rejection, not the confirmation, is the reportable result. What it supports:

- **Lower bound on the color-only ceiling:** Part-aligned color matching under a competent pose front-end tops out near 8.6% mAP on Market-1501. This is a useful reference point: any claim that a benchmark is "solvable by color" now has a measured number to argue against.
- **Dataset bias, reframed:** The original paired standard/clothes-changing test is no longer informative for the explicit model — a model at 8.6% on Market-1501 cannot meaningfully degrade on PRCC/LTCC. The bias question now has to be asked of the SOTA model directly, by ablating color at test time (grayscale conversion, hue randomization, per-channel shuffling) and measuring the drop. A large drop supports the color-dominance hypothesis that this baseline failed to establish; a small drop refutes it.
- **Interpretability:** Explicit alignment still makes per-keypoint contributions directly inspectable, and the Fisher analysis produces a usable discriminability ranking over body parts independent of the matching result.

What it does **not** support: the model-parsimony claim from the original proposal. Architectural complexity is not shown to be redundant; if anything, the size of the gap is evidence that the learned representation carries substantial signal beyond aligned local color.
