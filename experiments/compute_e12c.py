#!/usr/bin/env python3
"""
E12c — Within-domain unseen-identity controls and M5 camera fix.

Controls:
  A. Market half-A PCs → Market half-B alignment efficiency (within-source, no domain shift)
  B. Market PCs → MSMT17 alignment efficiency (cross-domain, main result from E12v2)
  C. MSMT17 half-A PCs → MSMT17 half-B alignment efficiency (within-target, per Daniel)

M5 camera fix:
  MSMT17 restricted to its 6 largest cameras (matching Market's 6-camera count).

Summary table compares A vs B vs C at k=10,20,50,100 for alignment efficiency.
If A ≈ B ≈ 44%, the loss is open-set novelty, not domain shift.
If A ≈ 80-90% and B ≈ 44%, domain shift is the driver.

Features cached to --cache-dir to avoid re-extracting (extraction ~30 min for MSMT17).
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

TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Feature extraction (with disk cache)
# ---------------------------------------------------------------------------

class _InferenceDS(Dataset):
    def __init__(self, samples, transform):
        self.samples = [s for s in samples if not s.is_junk]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return self.transform(s.load_image()), s.person_id, s.camera_id


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


@torch.no_grad()
def _extract_raw(model, samples, device, batch_size=256):
    ds = _InferenceDS(samples, TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    embs, pids, cids = [], [], []
    for imgs, ids, cams in loader:
        f = model(imgs.to(device))
        f = F.normalize(f, p=2, dim=1)
        embs.append(f.cpu().numpy())
        pids.extend(ids.numpy())
        cids.extend(cams.numpy())
    return np.vstack(embs), np.array(pids), np.array(cids)


def extract_cached(model, samples, device, cache_path: Path):
    if cache_path.exists():
        print(f"  Loading cached features from {cache_path}")
        data = np.load(cache_path)
        return data["emb"], data["pids"], data["cids"]
    print(f"  Extracting features ({len([s for s in samples if not s.is_junk])} samples)...")
    emb, pids, cids = _extract_raw(model, samples, device)
    np.savez(cache_path, emb=emb, pids=pids, cids=cids)
    print(f"  Cached to {cache_path}")
    return emb, pids, cids


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def between_id_covariance(X, ids):
    unique_ids = np.unique(ids)
    gm = X.mean(0)
    N = len(X)
    d = X.shape[1]
    S_B = np.zeros((d, d))
    for uid in unique_ids:
        mask = ids == uid
        n_k = mask.sum()
        diff = (X[mask].mean(0) - gm).reshape(-1, 1)
        S_B += (n_k / N) * (diff @ diff.T)
    return S_B


def effective_rank_spectral_entropy(X):
    """Effective rank via spectral entropy of sample covariance (exp(-sum p log p))."""
    Xc = X - X.mean(0)
    _, s, _ = np.linalg.svd(Xc / np.sqrt(len(X) - 1), full_matrices=False)
    lam = s ** 2
    lam = np.maximum(lam, 0)
    lam = lam / lam.sum()
    lam = lam[lam > 0]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def alignment_efficiency_curve(S_B_query, V_basis, n_pcs):
    """
    For each k=1..n_pcs:
      - source-basis capture: fraction of tr(S_B_query) in top-k of V_basis
      - query-optimal capture: fraction in top-k eigenvectors of S_B_query itself
      - alignment efficiency: source / optimal
    Returns arrays of length n_pcs.
    """
    total = np.trace(S_B_query)

    # Basis capture
    VtSBV = V_basis.T @ S_B_query @ V_basis          # (n_pcs, n_pcs)
    basis_capture = np.cumsum(np.diag(VtSBV)) / total

    # Optimal capture (eigenvectors of S_B_query)
    eig_vals, eig_vecs = np.linalg.eigh(S_B_query)
    idx = np.argsort(eig_vals)[::-1]
    V_opt = eig_vecs[:, idx[:n_pcs]]
    VtSBV_opt = V_opt.T @ S_B_query @ V_opt
    opt_capture = np.cumsum(np.diag(VtSBV_opt)) / total

    eff = basis_capture / np.maximum(opt_capture, 1e-8)
    return basis_capture, opt_capture, eff


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


def source_pcs(X, n_pcs):
    """Return top n_pcs principal directions of X (columns of returned matrix)."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc / np.sqrt(len(Xc) - 1), full_matrices=False)
    return Vt[:n_pcs].T  # (d, n_pcs)


def variance_decomp_r2(X, ids, cam_ids):
    """One-way R² for identity and camera (fixed estimator)."""
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
        "camera_r2": between_r2(cam_ids),
        "num_ids": int(len(np.unique(ids))),
        "num_cameras": int(len(np.unique(cam_ids))),
        "total_variance": total,
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
                        default=Path("/data/papers/low_dimensional_reid/results/e12c_results.json"))
    parser.add_argument("--n-pcs", type=int, default=128)
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
    msmt = get_dataset("msmt17", sample=False)

    # Extract / load features
    src_emb, src_ids, src_cids = extract_cached(
        model, market.query + market.gallery, args.device,
        args.cache_dir / "market_test.npz")
    tgt_emb, tgt_ids, tgt_cids = extract_cached(
        model, msmt.query + msmt.gallery, args.device,
        args.cache_dir / "msmt17_test.npz")

    n_pcs = args.n_pcs
    d = src_emb.shape[1]
    results = {}

    # -----------------------------------------------------------------------
    # Split Market test IDs 50/50 (half-A, half-B)
    # -----------------------------------------------------------------------
    src_unique = np.unique(src_ids)
    perm = rng.permutation(len(src_unique))
    half = len(src_unique) // 2
    src_A_ids = src_unique[perm[:half]]
    src_B_ids = src_unique[perm[half:]]
    mask_A = np.isin(src_ids, src_A_ids)
    mask_B = np.isin(src_ids, src_B_ids)
    X_srcA, ids_srcA, cids_srcA = src_emb[mask_A], src_ids[mask_A], src_cids[mask_A]
    X_srcB, ids_srcB, cids_srcB = src_emb[mask_B], src_ids[mask_B], src_cids[mask_B]
    print(f"\nMarket split: half-A {len(X_srcA)} imgs/{len(src_A_ids)} IDs, "
          f"half-B {len(X_srcB)} imgs/{len(src_B_ids)} IDs")

    # Split MSMT17 test IDs 50/50
    tgt_unique = np.unique(tgt_ids)
    perm_t = rng.permutation(len(tgt_unique))
    half_t = len(tgt_unique) // 2
    tgt_A_ids = tgt_unique[perm_t[:half_t]]
    tgt_B_ids = tgt_unique[perm_t[half_t:]]
    mask_tA = np.isin(tgt_ids, tgt_A_ids)
    mask_tB = np.isin(tgt_ids, tgt_B_ids)
    X_tgtA, ids_tgtA = tgt_emb[mask_tA], tgt_ids[mask_tA]
    X_tgtB, ids_tgtB = tgt_emb[mask_tB], tgt_ids[mask_tB]
    print(f"MSMT17 split: half-A {len(X_tgtA)} imgs/{len(tgt_A_ids)} IDs, "
          f"half-B {len(X_tgtB)} imgs/{len(tgt_B_ids)} IDs")

    # -----------------------------------------------------------------------
    # Compute PCA bases (full halves for A, B, C, D)
    # -----------------------------------------------------------------------
    print("\nFitting full PCA bases...")
    V_srcA  = source_pcs(X_srcA, n_pcs)   # Market half-A → measure on Market half-B (A)
    V_src   = source_pcs(src_emb, n_pcs)  # Market full   → measure on MSMT17 full (B)
    V_tgtA  = source_pcs(X_tgtA, n_pcs)   # MSMT17 half-A → measure on MSMT17 half-B (C)
    # Direction D: MSMT half-A PCs → Market half-B (falsification test)

    # -----------------------------------------------------------------------
    # Compute S_B for query sets
    # -----------------------------------------------------------------------
    print("Computing between-ID covariance matrices...")
    print("  Market half-B S_B...", flush=True)
    S_B_srcB = between_id_covariance(X_srcB, ids_srcB)
    print("  MSMT17 full S_B...", flush=True)
    S_B_tgt  = between_id_covariance(tgt_emb, tgt_ids)
    print("  MSMT17 half-B S_B...", flush=True)
    S_B_tgtB = between_id_covariance(X_tgtB, ids_tgtB)

    # -----------------------------------------------------------------------
    # C-matched: rebuild MSMT17 basis from 750 IDs to match Market's count
    # -----------------------------------------------------------------------
    N_DRAWS_C = 5
    N_IDS_MATCH = len(src_unique)  # 750
    print(f"\nBuilding C-matched bases ({N_DRAWS_C} draws of {N_IDS_MATCH} MSMT17 IDs from half-A)...")
    cond_C_matched_effs = []
    for draw in range(N_DRAWS_C):
        chosen = rng.choice(tgt_A_ids, min(N_IDS_MATCH, len(tgt_A_ids)), replace=False)
        mask_sub = np.isin(ids_tgtA, chosen)
        V_tgtA_sub = source_pcs(X_tgtA[mask_sub], n_pcs)
        _, _, eff_sub = alignment_efficiency_curve(S_B_tgtB, V_tgtA_sub, n_pcs)
        cond_C_matched_effs.append(eff_sub)
        print(f"  Draw {draw+1}: {mask_sub.sum()} imgs, {len(chosen)} IDs → eff@50={eff_sub[49]:.3f}")
    cond_C_matched_mean = np.mean(cond_C_matched_effs, axis=0)
    cond_C_matched_std  = np.std(cond_C_matched_effs,  axis=0)

    # -----------------------------------------------------------------------
    # Alignment efficiency: A, B, C, D (2×2 matrix)
    # -----------------------------------------------------------------------
    print("\nComputing alignment efficiency curves...")

    cond_A_basis, cond_A_opt, cond_A_eff = alignment_efficiency_curve(S_B_srcB, V_srcA, n_pcs)
    cond_B_basis, cond_B_opt, cond_B_eff = alignment_efficiency_curve(S_B_tgt,  V_src,  n_pcs)
    cond_C_basis, cond_C_opt, cond_C_eff = alignment_efficiency_curve(S_B_tgtB, V_tgtA, n_pcs)
    # D: MSMT-derived basis predicting into Market (reverse direction)
    cond_D_basis, cond_D_opt, cond_D_eff = alignment_efficiency_curve(S_B_srcB, V_tgtA, n_pcs)

    ks = [10, 20, 50, 100]
    def _row(eff, basis, opt, label, eff_std=None):
        row = {"label": label}
        for k in ks:
            if k <= n_pcs:
                row[f"k{k}_eff"] = float(eff[k-1])
                row[f"k{k}_basis"] = float(basis[k-1])
                row[f"k{k}_opt"] = float(opt[k-1])
                if eff_std is not None:
                    row[f"k{k}_eff_std"] = float(eff_std[k-1])
        return row

    cond_A_row = _row(cond_A_eff, cond_A_basis, cond_A_opt, "A: within-source  (Market A→B)")
    cond_B_row = _row(cond_B_eff, cond_B_basis, cond_B_opt, "B: cross-domain   (Market→MSMT17)")
    cond_C_row = _row(cond_C_eff, cond_C_basis, cond_C_opt, "C: within-target  (MSMT17 A→B, full basis)")
    cond_Cm_row = _row(cond_C_matched_mean, cond_C_basis, cond_C_opt,
                       "C': within-target (MSMT17, 750-ID basis, matched)", cond_C_matched_std)
    cond_D_row = _row(cond_D_eff, cond_D_basis, cond_D_opt, "D: reverse        (MSMT17 A→Market B)")

    results["m2_alignment_efficiency_comparison"] = {
        "conditions": [cond_A_row, cond_B_row, cond_C_row, cond_Cm_row, cond_D_row],
        "curves": {
            "A_within_source_eff": cond_A_eff.tolist(),
            "B_cross_domain_eff": cond_B_eff.tolist(),
            "C_within_target_eff": cond_C_eff.tolist(),
            "Cm_within_target_matched_mean": cond_C_matched_mean.tolist(),
            "Cm_within_target_matched_std": cond_C_matched_std.tolist(),
            "D_reverse_eff": cond_D_eff.tolist(),
        },
        "note": (
            "Primary comparison: B vs C' (both predict MSMT17, basis origin varies). "
            "D is the falsification test: if asymmetric (D >> B), geometry tracks retrieval asymmetry. "
            "A is the within-source baseline (partly reflects Market's higher dim, not a clean novelty floor)."
        ),
    }

    print("\nAlignment efficiency summary (@ k=10, 20, 50, 100):")
    print(f"{'Condition':<48} {'k=10':>7} {'k=20':>7} {'k=50':>7} {'k=100':>7}")
    for row in [cond_A_row, cond_B_row, cond_C_row, cond_Cm_row, cond_D_row]:
        vals = " ".join(f"{row[f'k{k}_eff']:>7.3f}" for k in ks if f"k{k}_eff" in row)
        print(f"  {row['label']:<46} {vals}")

    # -----------------------------------------------------------------------
    # M1 effective rank: same three conditions
    # -----------------------------------------------------------------------
    print("\nM1: Effective rank comparison...")
    er_srcB = effective_rank_spectral_entropy(X_srcB)
    er_tgt  = effective_rank_spectral_entropy(tgt_emb)
    er_tgtB = effective_rank_spectral_entropy(X_tgtB)
    er_srcA = effective_rank_spectral_entropy(X_srcA)
    er_tgtA = effective_rank_spectral_entropy(X_tgtA)

    results["m1_effective_rank"] = {
        "market_halfA": er_srcA,
        "market_halfB": er_srcB,
        "msmt17_halfA": er_tgtA,
        "msmt17_halfB": er_tgtB,
        "msmt17_full": er_tgt,
        "estimator": "spectral_entropy: exp(-sum p_i log p_i), p_i = lambda_i / sum(lambda)",
    }
    print(f"  Market A: {er_srcA:.1f}  B: {er_srcB:.1f}")
    print(f"  MSMT17 A: {er_tgtA:.1f}  B: {er_tgtB:.1f}  full: {er_tgt:.1f}")

    # -----------------------------------------------------------------------
    # M4 hubness: same three conditions
    # -----------------------------------------------------------------------
    print("\nM4: Hubness skew (size-matched: 375 IDs, ~8k imgs each)...")
    # Market half-B: 375 IDs, ~8200 imgs — use as-is (subsample images to MAX_HUB)
    hub_srcB = hubness_skew_sample(X_srcB, k=args.hubness_k, rng=rng)
    # MSMT17 matched: subsample to 375 IDs from tgt_B (same as Market half count)
    n_ids_hub = len(src_B_ids)  # 375
    chosen_hub = rng.choice(tgt_B_ids, min(n_ids_hub, len(tgt_B_ids)), replace=False)
    mask_hub = np.isin(ids_tgtB, chosen_hub)
    X_tgtB_hub = X_tgtB[mask_hub]
    hub_tgtB_matched = hubness_skew_sample(X_tgtB_hub, k=args.hubness_k, rng=rng)
    # Also full tgt_B for reference
    hub_tgtB_full = hubness_skew_sample(X_tgtB, k=args.hubness_k, rng=rng)

    results["m4_hubness"] = {
        "market_halfB_skew": hub_srcB,
        "msmt17_halfB_matched_skew": hub_tgtB_matched,
        "msmt17_halfB_full_skew": hub_tgtB_full,
        "n_ids_matched": n_ids_hub,
        "note": "matched = same number of IDs as Market half-B to control for sample size",
    }
    print(f"  Market half-B ({len(src_B_ids)} IDs): {hub_srcB:.3f}")
    print(f"  MSMT17 half-B matched ({n_ids_hub} IDs): {hub_tgtB_matched:.3f}  (unmatched: {hub_tgtB_full:.3f})")

    # -----------------------------------------------------------------------
    # M5: Camera group-count fix — restrict MSMT17 to top-6 cameras
    # -----------------------------------------------------------------------
    print("\nM5: Camera-count-matched variance decomposition...")
    # Top-6 MSMT17 cameras by sample count
    cam_counts = {c: (tgt_cids == c).sum() for c in np.unique(tgt_cids)}
    top6_cams = sorted(cam_counts, key=cam_counts.get, reverse=True)[:6]
    mask_6cam = np.isin(tgt_cids, top6_cams)
    X_tgt6, ids_tgt6, cids_tgt6 = tgt_emb[mask_6cam], tgt_ids[mask_6cam], tgt_cids[mask_6cam]
    print(f"  MSMT17 top-6 cameras: {sorted(top6_cams)}, {mask_6cam.sum()} images")

    vd_src  = variance_decomp_r2(src_emb, src_ids, src_cids)
    vd_tgt6 = variance_decomp_r2(X_tgt6, ids_tgt6, cids_tgt6)
    vd_tgt  = variance_decomp_r2(tgt_emb, tgt_ids, tgt_cids)

    results["m5_variance_decomp"] = {
        "source": vd_src,
        "target_6cam_matched": vd_tgt6,
        "target_full_15cam": vd_tgt,
        "note": "target_6cam_matched controls for camera group-count (Market=6, MSMT17 top-6)",
    }
    print(f"  Source      id R²:{vd_src['identity_r2']:.4f}  cam R²:{vd_src['camera_r2']:.4f}  ratio:{vd_src['identity_r2']/(vd_src['camera_r2']+1e-8):.1f}×  (C_id={vd_src['num_ids']}, C_cam={vd_src['num_cameras']})")
    print(f"  MSMT17 6cam id R²:{vd_tgt6['identity_r2']:.4f}  cam R²:{vd_tgt6['camera_r2']:.4f}  ratio:{vd_tgt6['identity_r2']/(vd_tgt6['camera_r2']+1e-8):.1f}×  (C_id={vd_tgt6['num_ids']}, C_cam={vd_tgt6['num_cameras']})")
    print(f"  MSMT17 full id R²:{vd_tgt['identity_r2']:.4f}  cam R²:{vd_tgt['camera_r2']:.4f}  ratio:{vd_tgt['identity_r2']/(vd_tgt['camera_r2']+1e-8):.1f}×  (C_id={vd_tgt['num_ids']}, C_cam={vd_tgt['num_cameras']})")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
