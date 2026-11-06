#!/usr/bin/env python3
"""
E10 — C2 style-shift control (inference only).

Per-channel mean/std transfer of MSMT17 images to Market channel statistics,
then recompute M1 (effective rank), M4 (hubness skew), M5 (identity-vs-camera R²).

If the collapse signature reverses → style shift is the cause.
If not → collapse is deeper than first/second-order appearance statistics.

Channel stats are cached separately; styled MSMT17 features are cached to
--cache-dir/msmt17_styled_test.npz.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from scipy.stats import skew
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reid-tk"))
from reid_tk.datasets import get_dataset

_IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_RESIZE_TENSOR = T.Compose([T.Resize((256, 128)), T.ToTensor()])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_bot_extractor(checkpoint: Path, device: str) -> nn.Module:
    resnet = models.resnet50(weights=None)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    base = nn.Sequential(*list(resnet.children())[:-2])
    bn = nn.BatchNorm1d(2048)
    bn.bias.requires_grad_(False)
    model = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bn)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Channel statistics
# ---------------------------------------------------------------------------

class _StatDS(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return _RESIZE_TENSOR(self.samples[idx].load_image())


def compute_channel_stats(samples, batch_size=256, num_workers=0):
    """Global per-channel (R,G,B) mean and std in [0,1] pixel space."""
    ds = _StatDS(samples)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    n_pixels = 0
    ch_sum = torch.zeros(3)
    ch_sq_sum = torch.zeros(3)
    for batch in loader:           # (B, 3, H, W)
        B, C, H, W = batch.shape
        ch_sum    += batch.sum(dim=(0, 2, 3))
        ch_sq_sum += (batch ** 2).sum(dim=(0, 2, 3))
        n_pixels  += B * H * W
    mean = (ch_sum / n_pixels).numpy()
    std  = torch.sqrt(ch_sq_sum / n_pixels - (ch_sum / n_pixels) ** 2).numpy()
    return mean, std


def load_channel_stats(samples, cache_path: Path, batch_size=256, num_workers=0):
    if cache_path.exists():
        d = np.load(cache_path)
        return d["mean"], d["std"]
    print(f"  Computing channel stats for {len([s for s in samples if not s.is_junk])} images...")
    mean, std = compute_channel_stats(samples, batch_size, num_workers)
    np.savez(cache_path, mean=mean, std=std)
    print(f"  Cached to {cache_path}")
    return mean, std


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class _StyleTransferTransform:
    """Per-channel mean/std transfer from src domain to tgt domain in [0,1]."""
    def __init__(self, src_mean, src_std, tgt_mean, tgt_std):
        self.src_mean = torch.tensor(src_mean, dtype=torch.float32).view(3, 1, 1)
        self.src_std  = torch.tensor(src_std,  dtype=torch.float32).view(3, 1, 1)
        self.tgt_mean = torch.tensor(tgt_mean, dtype=torch.float32).view(3, 1, 1)
        self.tgt_std  = torch.tensor(tgt_std,  dtype=torch.float32).view(3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.src_mean) / (self.src_std + 1e-8) * self.tgt_std + self.tgt_mean
        return x.clamp(0.0, 1.0)


class _InferenceDS(Dataset):
    def __init__(self, samples, transform):
        self.samples   = [s for s in samples if not s.is_junk]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return self.transform(s.load_image()), s.person_id, s.camera_id


def _build_transform(style_transfer=None):
    ops = [T.Resize((256, 128)), T.ToTensor()]
    if style_transfer is not None:
        ops.append(style_transfer)
    ops.append(_IMAGENET_NORM)
    return T.Compose(ops)


@torch.no_grad()
def _extract_raw(model, samples, device, transform, batch_size=256):
    ds = _InferenceDS(samples, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    embs, pids, cids = [], [], []
    for imgs, ids, cams in loader:
        f = model(imgs.to(device))
        f = F.normalize(f, p=2, dim=1)
        embs.append(f.cpu().numpy())
        pids.extend(ids.numpy())
        cids.extend(cams.numpy())
    return np.vstack(embs), np.array(pids), np.array(cids)


def extract_cached(model, samples, device, cache_path: Path, transform=None):
    if cache_path.exists():
        print(f"  Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data["emb"], data["pids"], data["cids"]
    n = len([s for s in samples if not s.is_junk])
    print(f"  Extracting features ({n} samples)...")
    tf = transform if transform is not None else _build_transform()
    emb, pids, cids = _extract_raw(model, samples, device, tf)
    np.savez(cache_path, emb=emb, pids=pids, cids=cids)
    print(f"  Cached to {cache_path}")
    return emb, pids, cids


# ---------------------------------------------------------------------------
# Geometry metrics (shared with compute_e12c)
# ---------------------------------------------------------------------------

def effective_rank_spectral_entropy(X):
    Xc = X - X.mean(0)
    _, s, _ = np.linalg.svd(Xc / np.sqrt(len(X) - 1), full_matrices=False)
    lam = np.maximum(s ** 2, 0)
    lam = lam / lam.sum()
    lam = lam[lam > 0]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def hubness_skew_sample(X, k=10, rng=None, n=5000):
    if rng is None:
        rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs = X[idx]
    sim = Xs @ Xs.T
    np.fill_diagonal(sim, -2.0)
    knn = np.argpartition(sim, -k, axis=1)[:, -k:]
    k_occ = np.bincount(knn.ravel(), minlength=len(Xs))
    return float(skew(k_occ))


def variance_decomp_r2(X, ids, cam_ids):
    mu = X.mean(0)
    total = float(np.mean(np.sum((X - mu) ** 2, axis=1)))
    N = len(X)

    def between_r2(labels):
        return sum(
            (labels == u).sum() / N * float(np.sum((X[labels == u].mean(0) - mu) ** 2))
            for u in np.unique(labels)
        ) / (total + 1e-8)

    return {
        "identity_r2": between_r2(ids),
        "camera_r2":   between_r2(cam_ids),
        "num_ids":     int(len(np.unique(ids))),
        "num_cameras": int(len(np.unique(cam_ids))),
        "total_variance": total,
    }


def run_geometry(label, X, ids, cids, rng, hubness_k, ref_n_ids):
    """Return M1, M4, M5 summary dict for a given feature matrix."""
    er = effective_rank_spectral_entropy(X)

    # M4: size-match to ref_n_ids (same as Market half-B count)
    unique_ids = np.unique(ids)
    if len(unique_ids) > ref_n_ids:
        chosen = rng.choice(unique_ids, ref_n_ids, replace=False)
        mask_hub = np.isin(ids, chosen)
        hub_matched = hubness_skew_sample(X[mask_hub], k=hubness_k, rng=rng)
    else:
        hub_matched = hubness_skew_sample(X, k=hubness_k, rng=rng)
        chosen = unique_ids
    hub_full = hubness_skew_sample(X, k=hubness_k, rng=rng)

    vd = variance_decomp_r2(X, ids, cids)

    return {
        "label": label,
        "n_samples": len(X),
        "n_ids": int(len(unique_ids)),
        "n_cameras": int(len(np.unique(cids))),
        "m1_effective_rank": er,
        "m4_hubness_matched": hub_matched,
        "m4_hubness_full": hub_full,
        "m4_n_ids_matched": int(min(len(unique_ids), ref_n_ids)),
        "m5_identity_r2": vd["identity_r2"],
        "m5_camera_r2":   vd["camera_r2"],
        "m5_ratio": vd["identity_r2"] / (vd["camera_r2"] + 1e-8),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/tmp/e12_feature_cache"))
    parser.add_argument("--out", type=Path,
                        default=Path("/data/papers/low_dimensional_reid/results/e10_results.json"))
    parser.add_argument("--hubness-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading model: {args.checkpoint}")
    model = load_bot_extractor(args.checkpoint, args.device)

    print("Loading datasets...")
    market = get_dataset("market1501", sample=False)
    msmt   = get_dataset("msmt17",     sample=False)
    src_samples = market.query + market.gallery
    tgt_samples = msmt.query   + msmt.gallery

    # ------------------------------------------------------------------
    # Channel statistics
    # ------------------------------------------------------------------
    print("\nComputing/loading channel statistics...")
    market_mean, market_std = load_channel_stats(
        src_samples, args.cache_dir / "market_channel_stats.npz")
    msmt_mean, msmt_std = load_channel_stats(
        tgt_samples, args.cache_dir / "msmt17_channel_stats.npz")

    print(f"  Market: mean={market_mean.round(4)}, std={market_std.round(4)}")
    print(f"  MSMT17: mean={msmt_mean.round(4)}, std={msmt_std.round(4)}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    print("\nLoading/extracting features...")

    # Market baseline (reuse existing cache if present)
    src_emb, src_ids, src_cids = extract_cached(
        model, src_samples, args.device,
        args.cache_dir / "market_test.npz")

    # MSMT17 baseline (reuse existing cache)
    tgt_emb, tgt_ids, tgt_cids = extract_cached(
        model, tgt_samples, args.device,
        args.cache_dir / "msmt17_test.npz")

    # MSMT17 style-transferred (MSMT17 → Market channel stats)
    style_tf = _StyleTransferTransform(msmt_mean, msmt_std, market_mean, market_std)
    styled_transform = _build_transform(style_tf)
    tgt_styled_emb, tgt_styled_ids, tgt_styled_cids = extract_cached(
        model, tgt_samples, args.device,
        args.cache_dir / "msmt17_styled_test.npz",
        transform=styled_transform)

    print(f"\nMarket: {len(src_emb)} imgs / {len(np.unique(src_ids))} IDs / {len(np.unique(src_cids))} cameras")
    print(f"MSMT17 baseline: {len(tgt_emb)} imgs / {len(np.unique(tgt_ids))} IDs")
    print(f"MSMT17 styled:   {len(tgt_styled_emb)} imgs / {len(np.unique(tgt_styled_ids))} IDs")

    # ------------------------------------------------------------------
    # Top-6 MSMT17 cameras (camera-count-matched for M5)
    # ------------------------------------------------------------------
    cam_counts = {c: (tgt_cids == c).sum() for c in np.unique(tgt_cids)}
    top6_cams  = sorted(cam_counts, key=cam_counts.get, reverse=True)[:6]
    print(f"\nMSMT17 top-6 cameras: {sorted(top6_cams)}")

    # ------------------------------------------------------------------
    # Reference: Market (for M4 size-match denominator)
    # ------------------------------------------------------------------
    ref_n_ids = len(np.unique(src_ids))  # 750 — use full Market as ref

    # ------------------------------------------------------------------
    # Run geometry on three conditions
    # ------------------------------------------------------------------
    print("\nRunning M1/M4/M5 geometry...")

    # Market (source reference)
    geo_src = run_geometry("Market (source)", src_emb, src_ids, src_cids, rng, args.hubness_k, ref_n_ids)

    # MSMT17 baseline (full 15 cameras)
    geo_tgt = run_geometry("MSMT17 baseline (15 cam)", tgt_emb, tgt_ids, tgt_cids, rng, args.hubness_k, ref_n_ids)

    # MSMT17 baseline (top-6 cameras)
    m6 = np.isin(tgt_cids, top6_cams)
    geo_tgt6 = run_geometry("MSMT17 baseline (top-6 cam)", tgt_emb[m6], tgt_ids[m6], tgt_cids[m6], rng, args.hubness_k, ref_n_ids)

    # MSMT17 styled (full 15 cameras)
    geo_styled = run_geometry("MSMT17 styled (15 cam)", tgt_styled_emb, tgt_styled_ids, tgt_styled_cids, rng, args.hubness_k, ref_n_ids)

    # MSMT17 styled (top-6 cameras)
    m6s = np.isin(tgt_styled_cids, top6_cams)
    geo_styled6 = run_geometry("MSMT17 styled (top-6 cam)", tgt_styled_emb[m6s], tgt_styled_ids[m6s], tgt_styled_cids[m6s], rng, args.hubness_k, ref_n_ids)

    conditions = [geo_src, geo_tgt, geo_tgt6, geo_styled, geo_styled6]

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"{'Condition':<36} {'Eff.rank':>9} {'Hubness':>9} {'Id R²':>8} {'Cam R²':>8} {'ratio':>7}")
    print("=" * 90)
    for g in conditions:
        print(f"  {g['label']:<34} {g['m1_effective_rank']:>9.1f} "
              f"{g['m4_hubness_matched']:>9.3f} "
              f"{g['m5_identity_r2']:>8.4f} "
              f"{g['m5_camera_r2']:>8.4f} "
              f"{g['m5_ratio']:>7.1f}×")

    print("\nDelta (styled − baseline, 15-cam):")
    delta_er  = geo_styled["m1_effective_rank"]  - geo_tgt["m1_effective_rank"]
    delta_hub = geo_styled["m4_hubness_matched"] - geo_tgt["m4_hubness_matched"]
    delta_idr = geo_styled["m5_identity_r2"]     - geo_tgt["m5_identity_r2"]
    delta_cr  = geo_styled["m5_camera_r2"]       - geo_tgt["m5_camera_r2"]
    print(f"  Eff. rank Δ:  {delta_er:+.1f}")
    print(f"  Hubness Δ:    {delta_hub:+.3f}")
    print(f"  Identity R² Δ:{delta_idr:+.4f}")
    print(f"  Camera R² Δ:  {delta_cr:+.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results = {
        "conditions": conditions,
        "channel_stats": {
            "market_mean": market_mean.tolist(),
            "market_std":  market_std.tolist(),
            "msmt17_mean": msmt_mean.tolist(),
            "msmt17_std":  msmt_std.tolist(),
        },
        "top6_cameras": sorted(int(c) for c in top6_cams),
        "delta_styled_vs_baseline": {
            "eff_rank":    delta_er,
            "hubness":     delta_hub,
            "identity_r2": delta_idr,
            "camera_r2":   delta_cr,
        },
        "interpretation": (
            "If eff_rank increases toward Market (~241) and hubness drops toward Market (~0.28), "
            "style shift explains the collapse. "
            "If metrics are unchanged, collapse is deeper than appearance statistics."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
