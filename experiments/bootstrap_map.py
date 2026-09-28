#!/usr/bin/env python3
"""
Paired bootstrap CI for mAP differences between Fisher-on and Fisher-off
configurations (R2 replication with uncertainty quantification).

Resamples the 3,368 Market-1501 queries 500 times (with replacement),
computing mAP for each configuration per bootstrap sample.  Reports the
mean difference and 95% CI for each pair.

Usage:
  python scripts/bootstrap_map.py \\
      --cache /tmp/pacm_cache_vitpose \\
      --slot-weights /tmp/kpt_weights_hs_mean.npy \\
      --n-boot 500 \\
      --output /tmp/bootstrap_r2.json
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from reid_tk.datasets import get_dataset
from reid_tk.pacm import pacm_distance_matrix
from reid_tk.pacm.descriptors import BODY_KPT_INDICES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def per_query_ap(
    dist_mat: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cameras: np.ndarray,
    gallery_cameras: np.ndarray,
) -> np.ndarray:
    """Compute per-query AP (Market-1501 protocol). Returns (N_q,) float array."""
    aps = np.full(len(query_ids), np.nan)
    for q_idx in range(len(query_ids)):
        q_id = query_ids[q_idx]
        q_cam = query_cameras[q_idx]
        dists = dist_mat[q_idx]
        sorted_idx = np.argsort(dists)
        junk = (gallery_ids == -1) | ((gallery_ids == q_id) & (gallery_cameras == q_cam))
        sorted_idx = sorted_idx[~junk[sorted_idx]]
        matches = gallery_ids[sorted_idx] == q_id
        if not matches.any():
            continue
        pos = np.where(matches)[0]
        precisions = [(i + 1) / (p + 1) for i, p in enumerate(pos)]
        aps[q_idx] = float(np.mean(precisions))
    return aps


def bootstrap_ci(
    aps_a: np.ndarray,
    aps_b: np.ndarray,
    n_boot: int = 500,
    rng_seed: int = 42,
) -> dict:
    """Paired bootstrap CI on mAP(b) - mAP(a).

    Both arrays must have the same length (one entry per query).
    NaN queries are excluded consistently from both.
    """
    rng = np.random.default_rng(rng_seed)
    valid = ~(np.isnan(aps_a) | np.isnan(aps_b))
    a = aps_a[valid]
    b = aps_b[valid]
    n = len(a)
    obs_diff = float(b.mean() - a.mean())
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(float(b[idx].mean() - a[idx].mean()))
    diffs = np.array(diffs)
    return {
        "obs_map_a": round(float(a.mean()), 5),
        "obs_map_b": round(float(b.mean()), 5),
        "obs_diff": round(obs_diff, 5),
        "ci_lo": round(float(np.percentile(diffs, 2.5)), 5),
        "ci_hi": round(float(np.percentile(diffs, 97.5)), 5),
        "n_valid_queries": int(valid.sum()),
        "n_boot": n_boot,
    }


def load_descriptors(cache_dir: Path, tag: str, split: str):
    p = cache_dir / "descriptors" / tag / f"{split}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Descriptor cache missing: {p}")
    log.info(f"Loading {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def build_dist_mat(qkds, gkds, slot_weights=None):
    return pacm_distance_matrix(
        qkds, gkds, dist_fn="l2", chunk_size=500, slot_weights=slot_weights
    )


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap CI for R2 Fisher gains")
    parser.add_argument("--dataset", default="market1501")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/pacm_cache_vitpose"))
    parser.add_argument("--slot-weights", type=Path,
                        default=Path("/tmp/kpt_weights_hs_mean.npy"),
                        help="Per-slot Fisher weights for hs_mean (used for Fisher-on run)")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset, sample=False)
    query_ids = np.array([s.person_id for s in dataset.query])
    gallery_ids = np.array([s.person_id for s in dataset.gallery])
    query_cams = np.array([s.camera_id for s in dataset.query])
    gallery_cams = np.array([s.camera_id for s in dataset.gallery])

    slot_weights = np.load(args.slot_weights)
    log.info(f"Fisher slot weights: {np.round(slot_weights, 3)}")

    base_tag = f"{args.dataset}_vitpose_conf0.3_patch10"
    configs = [
        ("lab_histogram", None, "lab_histogram_fisher_off"),
        ("lab_histogram", slot_weights, "lab_histogram_fisher_on"),
        ("hs_mean", None, "hs_mean_fisher_off"),
        ("hs_mean", slot_weights, "hs_mean_fisher_on"),
    ]

    aps_map: dict[str, np.ndarray] = {}

    for feature, weights, name in configs:
        log.info(f"\n--- {name} ---")
        tag = f"{base_tag}_{feature}"
        qkds = load_descriptors(args.cache, tag, "query")
        gkds = load_descriptors(args.cache, tag, "gallery")
        dm = build_dist_mat(qkds, gkds, slot_weights=weights)
        aps = per_query_ap(dm, query_ids, gallery_ids, query_cams, gallery_cams)
        aps_map[name] = aps
        log.info(f"  mAP={100*np.nanmean(aps):.2f}%  n_valid={np.sum(~np.isnan(aps))}")

    results = {
        "lab_histogram_fisher_effect": bootstrap_ci(
            aps_map["lab_histogram_fisher_off"],
            aps_map["lab_histogram_fisher_on"],
            n_boot=args.n_boot,
        ),
        "hs_mean_fisher_effect": bootstrap_ci(
            aps_map["hs_mean_fisher_off"],
            aps_map["hs_mean_fisher_on"],
            n_boot=args.n_boot,
        ),
    }

    log.info("\n=== Bootstrap CIs ===")
    for key, r in results.items():
        log.info(
            f"{key}: diff={100*r['obs_diff']:.2f} pp  "
            f"95% CI [{100*r['ci_lo']:.2f}, {100*r['ci_hi']:.2f}] pp"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"\nSaved → {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
