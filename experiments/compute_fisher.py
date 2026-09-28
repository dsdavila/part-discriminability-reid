#!/usr/bin/env python3
"""
Compute per-part Fisher statistics (S_B, S_W, F_k) on Market-1501 test split.

Reports:
  - Observed S_B, S_W, F_k per body-part group
  - 500-bootstrap identity CIs
  - Valid label-shuffle null (shuffle individual images across groups, not group labels)

Feature modes:
  color  -- hs_mean descriptors from evaluate_pacm.py cache (fast, no GPU)
  deep   -- BoT ResNet-50 Gaussian-weighted spatial feature pooling

Usage:
  python scripts/compute_fisher.py --feature color \\
      --cache /tmp/pacm_cache_vitpose --output /tmp/fisher_color.json

  python scripts/compute_fisher.py --feature deep \\
      --ckpt /data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth \\
      --output /tmp/fisher_deep_bot.json
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from reid_tk.datasets import get_dataset
from reid_tk.pacm.descriptors import BODY_KPT_INDICES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# COCO body keypoints → 6 anatomically-tied groups (left+right merged)
# BODY_KPT_INDICES = [5,6, 7,8, 9,10, 11,12, 13,14, 15,16]
# slots:              0 1  2 3  4  5   6  7   8  9  10 11
PART_GROUPS = {
    "Shoulder": [0, 1],   # slots 0,1  → kpts 5,6
    "Elbow":    [2, 3],   # slots 2,3  → kpts 7,8
    "Wrist":    [4, 5],   # slots 4,5  → kpts 9,10
    "Hip":      [6, 7],   # slots 6,7  → kpts 11,12
    "Knee":     [8, 9],   # slots 8,9  → kpts 13,14
    "Ankle":    [10, 11], # slots 10,11 → kpts 15,16
}
PART_ORDER = ["Shoulder", "Hip", "Knee", "Elbow", "Ankle", "Wrist"]


# ── Fisher statistics ──────────────────────────────────────────────────────────

def compute_sb_sw(vecs_by_id: dict[int, np.ndarray]) -> tuple[float, float]:
    """Between-ID (S_B) and within-ID (S_W) variance for one body-part group.

    vecs_by_id: {identity: (N_i, D) float32 array of valid descriptors}
    Returns (S_B, S_W) scalars, or (nan, nan) if fewer than 2 identities.
    """
    ids = [pid for pid, v in vecs_by_id.items() if len(v) >= 1]
    if len(ids) < 2:
        return float("nan"), float("nan")

    all_vecs = np.concatenate([vecs_by_id[pid] for pid in ids], axis=0)
    grand_mean = all_vecs.mean(axis=0)

    # Between-ID variance: weighted variance of per-ID means around grand mean
    n_total = len(all_vecs)
    sb = 0.0
    for pid in ids:
        v = vecs_by_id[pid]
        mu = v.mean(axis=0)
        sb += len(v) * np.mean((mu - grand_mean) ** 2)
    sb /= n_total

    # Within-ID variance: average per-ID variance
    sw = 0.0
    n_sw = 0
    for pid in ids:
        v = vecs_by_id[pid]
        if len(v) < 2:
            continue
        sw += np.mean((v - v.mean(axis=0)) ** 2) * len(v)
        n_sw += len(v)
    sw = sw / n_sw if n_sw > 0 else float("nan")

    return float(sb), float(sw)


def fisher_from_sb_sw(sb: float, sw: float) -> float:
    if sw == 0 or np.isnan(sw) or np.isnan(sb):
        return float("nan")
    return sb / sw


def bootstrap_fisher(
    vecs_by_id: dict[int, np.ndarray],
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap CI (2.5th, 97.5th percentile) on F_k by resampling identities."""
    if rng is None:
        rng = np.random.default_rng(42)
    ids = list(vecs_by_id.keys())
    n_ids = len(ids)
    fks = []
    for _ in range(n_boot):
        boot_ids = rng.choice(n_ids, size=n_ids, replace=True)
        boot = {i: vecs_by_id[ids[b]] for i, b in enumerate(boot_ids)}
        sb, sw = compute_sb_sw(boot)
        fks.append(fisher_from_sb_sw(sb, sw))
    fks = np.array([f for f in fks if not np.isnan(f)])
    return float(np.percentile(fks, 2.5)), float(np.percentile(fks, 97.5))


def label_shuffle_null(
    vecs_by_id: dict[int, np.ndarray],
    n_shuffles: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Valid label-shuffle null: shuffle individual images across identity groups.

    Each shuffle pools all feature vectors, shuffles them, and redistributes
    into groups matching the original per-identity counts. This destroys
    within-identity structure while preserving the marginal descriptor
    distribution.  F_k from a null shuffle should be ~1 if the observed
    ranking has no real signal.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ids = list(vecs_by_id.keys())
    sizes = [len(vecs_by_id[pid]) for pid in ids]
    all_vecs = np.concatenate([vecs_by_id[pid] for pid in ids], axis=0)
    N = len(all_vecs)

    fks = []
    for _ in range(n_shuffles):
        perm = rng.permutation(N)
        shuffled = all_vecs[perm]
        offset = 0
        boot = {}
        for i, (pid, sz) in enumerate(zip(ids, sizes)):
            boot[pid] = shuffled[offset : offset + sz]
            offset += sz
        sb, sw = compute_sb_sw(boot)
        fks.append(fisher_from_sb_sw(sb, sw))
    return np.array(fks)


# ── Feature builders ───────────────────────────────────────────────────────────

def load_color_descriptors(cache_dir: Path, dataset_tag: str) -> tuple[list, list[int]]:
    """Load cached hs_mean gallery descriptors and corresponding person IDs.

    Returns (kds, pids) where kds[i] is a KeypointDescriptors object.
    """
    desc_path = cache_dir / "descriptors" / f"{dataset_tag}_patch10_hs_mean" / "gallery.pkl"
    if not desc_path.exists():
        log.error(f"Descriptor cache not found: {desc_path}")
        sys.exit(1)
    log.info(f"Loading color descriptors from {desc_path}")
    with open(desc_path, "rb") as f:
        kds = pickle.load(f)
    return kds


def build_color_vecs_by_id(
    kds, pids: np.ndarray, conf_threshold: float = 0.3
) -> dict[str, dict[int, np.ndarray]]:
    """Build {part_group: {person_id: (N_i, 2) array}} from KeypointDescriptors."""
    from reid_tk.pacm.descriptors import BODY_KPT_INDICES

    # Accumulate per part-group, per identity
    result: dict[str, dict] = {p: {} for p in PART_GROUPS}

    for kd, pid in zip(kds, pids):
        if kd.is_empty:
            continue
        idx_to_row = {int(k): r for r, k in enumerate(kd.keypoint_indices)}

        for part, slots in PART_GROUPS.items():
            vecs = []
            for slot in slots:
                kpt_idx = BODY_KPT_INDICES[slot]
                if kpt_idx in idx_to_row:
                    vecs.append(kd.descriptors[idx_to_row[kpt_idx]])
            if not vecs:
                continue
            # Average left+right descriptors if both visible
            v = np.mean(vecs, axis=0).reshape(1, -1)
            if pid not in result[part]:
                result[part][pid] = []
            result[part][pid].append(v[0])

    # Convert lists to arrays
    for part in result:
        result[part] = {
            pid: np.stack(vecs, axis=0)
            for pid, vecs in result[part].items()
            if vecs
        }
    return result


def build_bot_feature_extractor(ckpt_path: Path):
    """Load BoT ResNet50 (last_stride=1, BNNeck) for spatial feature extraction.

    Returns a callable that takes a batch of (B, 3, 256, 128) tensors and
    returns (B, 2048, 16, 8) spatial feature maps (before BNNeck / GAP).
    """
    import torch
    import torch.nn as nn
    import torchvision.models as models

    resnet = models.resnet50(weights=None)
    # last_stride = 1
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    base = nn.Sequential(*list(resnet.children())[:-2])

    # Load weights: checkpoint keys are '0.*' for base, '3.*' for BNNeck
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_state = {k[2:]: v for k, v in ckpt.items() if k.startswith("0.")}
    missing, unexpected = base.load_state_dict(base_state, strict=True)
    log.info(f"BoT backbone loaded — missing: {missing}, unexpected: {unexpected}")

    base.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = base.to(device)
    return base, device


def gaussian_pool_keypoints(
    feat_map: np.ndarray,
    keypoints: np.ndarray,
    img_hw: tuple[int, int],
    kpt_indices: list[int],
    conf_threshold: float = 0.3,
    sigma_frac: float = 1.0,
) -> dict[int, np.ndarray]:
    """Gaussian-weighted pool feature map at each valid keypoint.

    feat_map: (C, H_f, W_f)  spatial features
    keypoints: (17, 3)  raw (x, y, conf) in pixel coordinates
    img_hw: (img_H, img_W) original image dimensions
    kpt_indices: which COCO indices to process
    sigma_frac: Gaussian sigma in feature-map patches (1.0 = 1 patch width)

    Returns {coco_kpt_idx: (C,) pooled feature}.
    """
    C, H_f, W_f = feat_map.shape
    img_H, img_W = img_hw
    # pixel → feature-map scale
    scale_x = W_f / img_W
    scale_y = H_f / img_H

    result = {}
    ys = np.arange(H_f, dtype=np.float32)
    xs = np.arange(W_f, dtype=np.float32)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")  # (H_f, W_f)

    for idx in kpt_indices:
        if idx >= len(keypoints):
            continue
        x, y, conf = keypoints[idx]
        if conf < conf_threshold:
            continue
        cx = float(x) * scale_x
        cy = float(y) * scale_y
        sigma = sigma_frac
        w = np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma ** 2))
        w_sum = w.sum()
        if w_sum < 1e-7:
            continue
        w = w / w_sum  # (H_f, W_f)
        pooled = (feat_map * w[None, :, :]).sum(axis=(1, 2))  # (C,)
        result[idx] = pooled.astype(np.float32)
    return result


def build_deep_vecs_by_id(
    dataset,
    kpt_cache_path: Path,
    ckpt_path: Path,
    batch_size: int = 64,
    conf_threshold: float = 0.3,
) -> dict[str, dict[int, np.ndarray]]:
    """Extract BoT deep features for gallery images and group by identity."""
    import torch
    import torchvision.transforms as T
    from PIL import Image

    log.info("Loading keypoints from cache...")
    with open(kpt_cache_path, "rb") as f:
        kpts_list = pickle.load(f)

    base, device = build_bot_feature_extractor(ckpt_path)

    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    result: dict[str, dict] = {p: {} for p in PART_GROUPS}
    samples = dataset.gallery
    IMG_HW = (256, 128)

    log.info(f"Extracting deep features for {len(samples)} gallery images...")
    for batch_start in tqdm(range(0, len(samples), batch_size)):
        batch_end = min(batch_start + batch_size, len(samples))
        imgs_t = []
        for i in range(batch_start, batch_end):
            img = Image.open(samples[i].image_path).convert("RGB")
            imgs_t.append(transform(img))
        imgs_t = torch.stack(imgs_t).to(device)

        with torch.no_grad():
            feat_maps = base(imgs_t).cpu().numpy()  # (B, 2048, 16, 8)

        for b_i, i in enumerate(range(batch_start, batch_end)):
            pid = samples[i].person_id
            kpts = kpts_list[i]  # (17, 3) or None
            if kpts is None:
                continue

            # Use actual image dimensions so keypoint pixel coords map correctly.
            orig_img = Image.open(samples[i].image_path)
            orig_w, orig_h = orig_img.size  # PIL returns (width, height)
            feat_map = feat_maps[b_i]  # (2048, 16, 8)
            pooled = gaussian_pool_keypoints(
                feat_map, kpts, (orig_h, orig_w), BODY_KPT_INDICES,
                conf_threshold=conf_threshold, sigma_frac=1.0,
            )

            for part, slots in PART_GROUPS.items():
                vecs = []
                for slot in slots:
                    kpt_idx = BODY_KPT_INDICES[slot]
                    if kpt_idx in pooled:
                        vecs.append(pooled[kpt_idx])
                if not vecs:
                    continue
                v = np.mean(vecs, axis=0)  # (2048,)
                if pid not in result[part]:
                    result[part][pid] = []
                result[part][pid].append(v)

    for part in result:
        result[part] = {
            pid: np.stack(vecs, axis=0)
            for pid, vecs in result[part].items()
            if vecs
        }
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def run_fisher_analysis(
    vecs_by_part: dict[str, dict[int, np.ndarray]],
    n_boot: int = 500,
    n_null: int = 500,
    rng_seed: int = 42,
) -> dict:
    """Run full Fisher analysis: observed stats, bootstrap CI, shuffle null."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for part in PART_ORDER:
        vecs = vecs_by_part.get(part, {})
        if not vecs:
            continue
        n_ids = len(vecs)
        n_imgs = sum(len(v) for v in vecs.values())

        sb, sw = compute_sb_sw(vecs)
        fk = fisher_from_sb_sw(sb, sw)
        ci_lo, ci_hi = bootstrap_fisher(vecs, n_boot=n_boot, rng=rng)
        null_fks = label_shuffle_null(vecs, n_shuffles=n_null, rng=rng)
        null_mean = float(np.nanmean(null_fks))
        null_p95 = float(np.nanpercentile(null_fks, 95))

        rows.append({
            "part": part,
            "n_ids": n_ids,
            "n_imgs": n_imgs,
            "S_B": round(sb, 4),
            "S_W": round(sw, 4),
            "F_k": round(fk, 4),
            "CI_lo": round(ci_lo, 4),
            "CI_hi": round(ci_hi, 4),
            "null_mean": round(null_mean, 4),
            "null_p95": round(null_p95, 4),
        })
        log.info(
            f"  {part:10s}  S_B={sb:.4f}  S_W={sw:.4f}  F_k={fk:.4f} "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]  null_mean={null_mean:.4f}"
        )
    return {"parts": rows}


def main():
    parser = argparse.ArgumentParser(description="Compute per-part Fisher statistics")
    parser.add_argument("--feature", choices=["color", "deep"], default="color")
    parser.add_argument("--dataset", default="market1501", choices=["market1501", "msmt17"])
    parser.add_argument("--cache", type=Path, default=Path("/tmp/pacm_cache_vitpose"),
                        help="evaluate_pacm.py descriptor cache root")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth"),
                        help="BoT checkpoint path (deep mode only)")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--n-null", type=int, default=500)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset, sample=False)
    gallery_pids = np.array([s.person_id for s in dataset.gallery])
    test_mask = gallery_pids > 0  # exclude junk/distractor
    gallery_samples = [s for s in dataset.gallery if s.person_id > 0]
    gallery_pids = gallery_pids[test_mask]

    log.info(f"Gallery test-split images: {len(gallery_samples)} over {len(set(gallery_pids))} IDs")

    if args.feature == "color":
        tag = f"{args.dataset}_vitpose_conf{args.conf}"
        kds_all = load_color_descriptors(args.cache, tag)
        # Filter to test-split (person_id > 0)
        kds = [kd for kd, s in zip(kds_all, dataset.gallery) if s.person_id > 0]
        log.info("Building per-part color descriptor groups...")
        vecs_by_part = build_color_vecs_by_id(kds, gallery_pids, conf_threshold=args.conf)

    else:  # deep
        kpt_tag = f"{args.dataset}_vitpose_conf{args.conf}"
        kpt_cache = args.cache / "keypoints" / kpt_tag / "gallery.pkl"
        # Filter gallery to test-split inside build_deep
        # Use original dataset.gallery with test-split person_id filter
        class FilteredDataset:
            def __init__(self, samples):
                self.gallery = samples
        fd = FilteredDataset(gallery_samples)

        with open(kpt_cache, "rb") as f:
            all_kpts = pickle.load(f)
        # Filter keypoints to test-split images
        kpts_filtered = [kpt for kpt, s in zip(all_kpts, dataset.gallery) if s.person_id > 0]

        # Inline deep extraction
        import torch, torchvision.transforms as T
        from PIL import Image
        base, device = build_bot_feature_extractor(args.ckpt)
        transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        IMG_HW = (256, 128)
        vecs_by_part: dict[str, dict] = {p: {} for p in PART_GROUPS}

        log.info(f"Extracting BoT deep features for {len(gallery_samples)} images...")
        for bs in tqdm(range(0, len(gallery_samples), args.batch_size)):
            be = min(bs + args.batch_size, len(gallery_samples))
            pil_imgs = [Image.open(gallery_samples[i].image_path).convert("RGB") for i in range(bs, be)]
            imgs_t = torch.stack([transform(img) for img in pil_imgs]).to(device)
            with torch.no_grad():
                feat_maps = base(imgs_t).cpu().numpy()  # (B, 2048, 16, 8)
            for b_i, i in enumerate(range(bs, be)):
                pid = gallery_samples[i].person_id
                kpts = kpts_filtered[i]
                if kpts is None:
                    continue
                orig_w, orig_h = pil_imgs[b_i].size  # PIL returns (width, height)
                feat_map = feat_maps[b_i]
                pooled = gaussian_pool_keypoints(
                    feat_map, kpts, (orig_h, orig_w), BODY_KPT_INDICES,
                    conf_threshold=args.conf, sigma_frac=1.0,
                )
                for part, slots in PART_GROUPS.items():
                    vecs = [pooled[BODY_KPT_INDICES[s]] for s in slots if BODY_KPT_INDICES[s] in pooled]
                    if not vecs:
                        continue
                    v = np.mean(vecs, axis=0)
                    vecs_by_part[part].setdefault(pid, []).append(v)

        for part in vecs_by_part:
            vecs_by_part[part] = {
                pid: np.stack(vs, axis=0)
                for pid, vs in vecs_by_part[part].items() if vs
            }

    log.info("=== Fisher Analysis ===")
    results = run_fisher_analysis(
        vecs_by_part, n_boot=args.n_boot, n_null=args.n_null
    )
    results["feature"] = args.feature
    results["dataset"] = args.dataset

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved → {args.output}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
