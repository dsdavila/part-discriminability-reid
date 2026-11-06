#!/usr/bin/env python3
"""
E12 — Embedding geometry and the source-subspace hypothesis.

Five measurements on BoT ResNet-50 features (Market-trained):
  1. Covariance spectrum + effective rank (source test vs target test)
  2. Fraction of target between-ID variance in top-k source PCs,
     with target-optimal ceiling and random-subspace floor
  3. Neural-collapse metrics across source-train / source-test / target-test IDs
  4. Hubness: skew of k-occurrence distribution (source vs target)
  5. Variance explained by camera/domain label vs identity label (source vs target)

Fixes vs. v1:
  - M2: adds target-optimal ceiling (target's own PCA of S_B), random null (k/d),
        alignment efficiency = source-basis capture / target-optimal capture
  - M1–M4: all repeated on size-matched target (750 IDs, ~23k images, 5 draws)
            to separate dataset-size effects from domain effects
  - M5: fixes between_var bug (mean→sum over classes); reports correct one-way R²

Usage:
    python experiments/compute_e12.py \
        --checkpoint /data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth \
        --out /data/papers/low_dimensional_reid/results/e12_results.json
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
# Model loading
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
def extract(model, samples, device, batch_size=256):
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


def subsample_ids(X, ids, cids, n_ids, rng):
    """Return a random subset of n_ids unique IDs with all their images."""
    unique = np.unique(ids)
    chosen = rng.choice(unique, min(n_ids, len(unique)), replace=False)
    mask = np.isin(ids, chosen)
    return X[mask], ids[mask], cids[mask]


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def effective_rank(eigenvalues: np.ndarray) -> float:
    lam = np.maximum(eigenvalues, 0)
    lam = lam / lam.sum()
    lam = lam[lam > 0]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def covariance_spectrum(X: np.ndarray, n_components: int = 128):
    """Top-n eigenvalues of sample covariance, plus effective rank."""
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    _, s, _ = np.linalg.svd(Xc / np.sqrt(len(X) - 1), full_matrices=False)
    eigenvalues = s ** 2
    total = eigenvalues.sum()
    return eigenvalues[:n_components], eigenvalues[:n_components] / total, effective_rank(eigenvalues)


def between_id_covariance(X: np.ndarray, ids: np.ndarray):
    """Between-ID covariance matrix S_B = sum_k (n_k/N)(mu_k - mu)(mu_k - mu)^T."""
    unique_ids = np.unique(ids)
    global_mean = X.mean(0)
    N = len(X)
    d = X.shape[1]
    S_B = np.zeros((d, d))
    for uid in unique_ids:
        mask = ids == uid
        n_k = mask.sum()
        mu_k = X[mask].mean(0)
        diff = (mu_k - global_mean).reshape(-1, 1)
        S_B += (n_k / N) * (diff @ diff.T)
    return S_B


def capture_curve(S_B: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Fraction of tr(S_B) captured by top-k columns of V, for k=1..n_pcs.
    V: (d, n_pcs), columns are basis vectors.
    """
    total = np.trace(S_B)
    # Precompute V^T S_B V once — (n_pcs, n_pcs) — then take cumsum of diagonal
    VtSBV = V.T @ S_B @ V
    return np.cumsum(np.diag(VtSBV)) / total


# ---------------------------------------------------------------------------
# Measurement 2: source-subspace capture with ceiling and floor
# ---------------------------------------------------------------------------

def m2_subspace_capture(S_B_src, S_B_tgt, V_src, n_pcs, d):
    """
    Returns dict with source-self, target-by-source, target-optimal, random-null,
    and alignment efficiency curves.
    """
    # Source self-capture (sanity check: should approach 1 quickly)
    src_self = capture_curve(S_B_src, V_src)

    # Target captured by source PCs
    tgt_by_src = capture_curve(S_B_tgt, V_src)

    # Target-optimal: eigenvectors of S_B_tgt (symmetric, use eigh)
    eigenvalues_tgt, eigenvectors_tgt = np.linalg.eigh(S_B_tgt)
    idx = np.argsort(eigenvalues_tgt)[::-1]
    V_tgt_opt = eigenvectors_tgt[:, idx[:n_pcs]]
    tgt_optimal = capture_curve(S_B_tgt, V_tgt_opt)

    # Random null: k/d (analytical expected value for a uniformly random k-dim subspace)
    random_null = np.arange(1, n_pcs + 1) / d

    # Alignment efficiency: how much of the target-optimal is recovered by source PCs
    alignment_eff = tgt_by_src / np.maximum(tgt_optimal, 1e-8)

    def _summary(curve, prefix):
        ks = [10, 20, 50, 100]
        return {f"capture_at_k{k}_{prefix}": float(curve[k-1]) for k in ks if k <= n_pcs}

    result = {
        "source_total_id_variance": float(np.trace(S_B_src)),
        "target_total_id_variance": float(np.trace(S_B_tgt)),
        "source_self_capture_curve": src_self.tolist(),
        "target_by_source_capture_curve": tgt_by_src.tolist(),
        "target_optimal_capture_curve": tgt_optimal.tolist(),
        "random_null_curve": random_null.tolist(),
        "alignment_efficiency_curve": alignment_eff.tolist(),
        "k_at_50pct_source_self": int(np.searchsorted(src_self, 0.5) + 1),
        "k_at_50pct_target_by_source": int(np.searchsorted(tgt_by_src, 0.5) + 1),
        "k_at_50pct_target_optimal": int(np.searchsorted(tgt_optimal, 0.5) + 1),
    }
    result.update(_summary(src_self, "src_self"))
    result.update(_summary(tgt_by_src, "tgt_by_src"))
    result.update(_summary(tgt_optimal, "tgt_optimal"))
    result.update(_summary(alignment_eff, "align_eff"))
    return result


# ---------------------------------------------------------------------------
# Measurement 3: Neural collapse
# ---------------------------------------------------------------------------

def neural_collapse_metrics(X: np.ndarray, ids: np.ndarray):
    unique_ids = np.unique(ids)
    C = len(unique_ids)

    class_means = np.stack([X[ids == uid].mean(0) for uid in unique_ids])
    global_mean = class_means.mean(0)

    within_var = float(np.mean([
        np.mean(np.sum((X[ids == uid] - class_means[i]) ** 2, axis=1))
        for i, uid in enumerate(unique_ids)
    ]))
    between_var = float(np.mean(np.sum((class_means - global_mean) ** 2, axis=1)))
    wcvr = within_var / (between_var + 1e-8)

    M = class_means - global_mean
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
    M_norm = M / norms
    cosmat = M_norm @ M_norm.T
    target_cos = -1.0 / (C - 1)
    off_diag = ~np.eye(C, dtype=bool)
    etf_dev = float(np.mean(np.abs(cosmat[off_diag] - target_cos)))

    return {
        "num_ids": C,
        "within_class_variance": within_var,
        "between_class_variance": between_var,
        "wcvr": float(wcvr),
        "etf_target_cosine": float(target_cos),
        "etf_mean_deviation": etf_dev,
    }


# ---------------------------------------------------------------------------
# Measurement 4: Hubness
# ---------------------------------------------------------------------------

def hubness_skew(X: np.ndarray, k: int = 10) -> dict:
    sim = X @ X.T
    np.fill_diagonal(sim, -2.0)
    knn = np.argpartition(sim, -k, axis=1)[:, -k:]
    k_occ = np.bincount(knn.ravel(), minlength=len(X))
    return {
        "k": k,
        "n_samples": len(X),
        "mean_k_occ": float(k_occ.mean()),
        "std_k_occ": float(k_occ.std()),
        "skewness": float(skew(k_occ)),
        "max_k_occ": int(k_occ.max()),
        "fraction_zero_occ": float((k_occ == 0).mean()),
    }


# ---------------------------------------------------------------------------
# Measurement 5: Variance decomposition (fixed)
# ---------------------------------------------------------------------------

def variance_decomposition(X: np.ndarray, ids: np.ndarray, cam_ids: np.ndarray) -> dict:
    """
    One-way R² for identity and camera labels.
    R²_identity = between-ID SS / total SS
               = sum_k n_k/N ||mu_k - mu||^2  /  mean_i ||x_i - mu||^2
    (note: between_SS/N divided by total_SS/N = between_SS / total_SS)
    """
    mu = X.mean(0)
    total_var = float(np.mean(np.sum((X - mu) ** 2, axis=1)))  # total SS / N

    def between_r2(X, labels, mu):
        unique = np.unique(labels)
        N = len(X)
        # between SS / N = sum_k (n_k/N) * ||mu_k - mu||^2
        bss_over_N = sum(
            (labels == u).sum() / N * float(np.sum((X[labels == u].mean(0) - mu) ** 2))
            for u in unique
        )
        return bss_over_N / (total_var + 1e-8)

    id_r2 = between_r2(X, ids, mu)
    cam_r2 = between_r2(X, cam_ids, mu)

    return {
        "total_variance": total_var,
        "identity_r2": id_r2,
        "camera_r2": cam_r2,
        "num_ids": int(len(np.unique(ids))),
        "num_cameras": int(len(np.unique(cam_ids))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth"))
    parser.add_argument("--out", type=Path,
                        default=Path("/data/papers/low_dimensional_reid/results/e12_results.json"))
    parser.add_argument("--n-pcs", type=int, default=128)
    parser.add_argument("--hubness-k", type=int, default=10)
    parser.add_argument("--size-match-draws", type=int, default=5,
                        help="Number of random target-subsampling draws for size-matched comparisons")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model = load_bot_extractor(args.checkpoint, args.device)

    print("Loading datasets...")
    market = get_dataset("market1501", sample=False)
    msmt = get_dataset("msmt17", sample=False)

    print("Extracting source-train features (Market-1501 train)...")
    src_train_emb, src_train_ids, src_train_cids = extract(model, market.train, args.device)

    print("Extracting source-test features (Market-1501 query+gallery)...")
    src_test_emb, src_test_ids, src_test_cids = extract(
        model, market.query + market.gallery, args.device)

    print("Extracting target features (MSMT17 query+gallery)...")
    tgt_emb, tgt_ids, tgt_cids = extract(
        model, msmt.query + msmt.gallery, args.device)

    n_src_ids = len(np.unique(src_test_ids))
    n_tgt_ids = len(np.unique(tgt_ids))
    print(f"  src-train: {len(src_train_emb)} imgs, {len(np.unique(src_train_ids))} IDs")
    print(f"  src-test:  {len(src_test_emb)} imgs, {n_src_ids} IDs")
    print(f"  target:    {len(tgt_emb)} imgs, {n_tgt_ids} IDs")

    n_pcs = args.n_pcs
    d = src_test_emb.shape[1]
    rng = np.random.default_rng(42)
    results = {}

    # ------------------------------------------------------------------
    # Shared: source PCA and between-ID covariances (full target)
    # ------------------------------------------------------------------
    src_mu = src_test_emb.mean(0)
    Xc = src_test_emb - src_mu
    _, _, Vt = np.linalg.svd(Xc / np.sqrt(len(Xc) - 1), full_matrices=False)
    V_src = Vt[:n_pcs].T  # (d, n_pcs)

    print("\nComputing between-ID covariance matrices...")
    S_B_src = between_id_covariance(src_test_emb, src_test_ids)
    print("  Source S_B done.")
    S_B_tgt = between_id_covariance(tgt_emb, tgt_ids)
    print("  Target S_B done.")

    # ------------------------------------------------------------------
    # M1: Covariance spectrum (full datasets)
    # ------------------------------------------------------------------
    print("\nM1: Covariance spectrum...")
    src_eig, src_frac, src_er = covariance_spectrum(src_test_emb, n_components=n_pcs)
    tgt_eig, tgt_frac, tgt_er = covariance_spectrum(tgt_emb, n_components=n_pcs)
    print(f"  Source effective rank: {src_er:.1f}   Target (full): {tgt_er:.1f}")

    results["m1_spectrum"] = {
        "source_effective_rank": src_er,
        "target_effective_rank": tgt_er,
        "source_cumvar_fractions": np.cumsum(src_frac).tolist(),
        "target_cumvar_fractions": np.cumsum(tgt_frac).tolist(),
        "source_top_eigenvalues": src_eig.tolist(),
        "target_top_eigenvalues": tgt_eig.tolist(),
    }

    # ------------------------------------------------------------------
    # M2: Subspace capture with ceiling and floor (full + size-matched)
    # ------------------------------------------------------------------
    print("\nM2: Source-subspace capture (full target)...")
    m2_full = m2_subspace_capture(S_B_src, S_B_tgt, V_src, n_pcs, d)
    print(f"  Target by source @50: {m2_full['capture_at_k50_tgt_by_src']:.3f}  "
          f"target optimal @50: {m2_full['capture_at_k50_tgt_optimal']:.3f}  "
          f"alignment eff @50: {m2_full['capture_at_k50_align_eff']:.3f}")

    results["m2_subspace_capture_full"] = m2_full

    # ------------------------------------------------------------------
    # Size-matched comparisons: subsample target to n_src_ids IDs
    # ------------------------------------------------------------------
    print(f"\nSize-matched runs ({args.size_match_draws} draws, {n_src_ids} IDs each)...")

    m1_er_matched = []
    m2_tgt_by_src_at50 = []
    m2_tgt_opt_at50 = []
    m2_align_eff_at50 = []
    m3_wcvr_matched = []
    m4_skew_matched = []
    m5_id_r2_matched = []
    m5_cam_r2_matched = []

    MAX_HUB = 5000

    for draw in range(args.size_match_draws):
        X_m, ids_m, cids_m = subsample_ids(tgt_emb, tgt_ids, tgt_cids, n_src_ids, rng)
        print(f"  Draw {draw+1}: {len(X_m)} imgs, {len(np.unique(ids_m))} IDs")

        # M1 matched
        _, _, er_m = covariance_spectrum(X_m, n_components=n_pcs)
        m1_er_matched.append(er_m)

        # M2 matched
        S_B_m = between_id_covariance(X_m, ids_m)
        m2_m = m2_subspace_capture(S_B_src, S_B_m, V_src, n_pcs, d)
        m2_tgt_by_src_at50.append(m2_m.get("capture_at_k50_tgt_by_src", float("nan")))
        m2_tgt_opt_at50.append(m2_m.get("capture_at_k50_tgt_optimal", float("nan")))
        m2_align_eff_at50.append(m2_m.get("capture_at_k50_align_eff", float("nan")))

        # M3 matched
        nc_m = neural_collapse_metrics(X_m, ids_m)
        m3_wcvr_matched.append(nc_m["wcvr"])

        # M4 matched
        n_hub = min(MAX_HUB, len(X_m))
        idx_hub = rng.choice(len(X_m), n_hub, replace=False)
        hub_m = hubness_skew(X_m[idx_hub], k=args.hubness_k)
        m4_skew_matched.append(hub_m["skewness"])

        # M5 matched
        vd_m = variance_decomposition(X_m, ids_m, cids_m)
        m5_id_r2_matched.append(vd_m["identity_r2"])
        m5_cam_r2_matched.append(vd_m["camera_r2"])

    def _mean_std(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    results["size_matched_summary"] = {
        "n_ids_matched": n_src_ids,
        "n_draws": args.size_match_draws,
        "m1_target_effective_rank": _mean_std(m1_er_matched),
        "m2_target_by_source_at_k50": _mean_std(m2_tgt_by_src_at50),
        "m2_target_optimal_at_k50": _mean_std(m2_tgt_opt_at50),
        "m2_alignment_efficiency_at_k50": _mean_std(m2_align_eff_at50),
        "m3_target_wcvr": _mean_std(m3_wcvr_matched),
        "m4_target_hubness_skew": _mean_std(m4_skew_matched),
        "m5_target_identity_r2": _mean_std(m5_id_r2_matched),
        "m5_target_camera_r2": _mean_std(m5_cam_r2_matched),
        "source_reference": {
            "m1_effective_rank": src_er,
            "m2_source_self_at_k50": float(np.array(m2_full["source_self_capture_curve"])[49]),
            "m3_source_test_wcvr": None,  # filled below
            "m4_source_hubness_skew": None,
            "m5_source_identity_r2": None,
            "m5_source_camera_r2": None,
        }
    }

    # ------------------------------------------------------------------
    # M3: Neural collapse (full datasets)
    # ------------------------------------------------------------------
    print("\nM3: Neural collapse metrics...")
    nc_src_train = neural_collapse_metrics(src_train_emb, src_train_ids)
    nc_src_test = neural_collapse_metrics(src_test_emb, src_test_ids)
    nc_tgt = neural_collapse_metrics(tgt_emb, tgt_ids)

    results["m3_neural_collapse"] = {
        "source_train": nc_src_train,
        "source_test": nc_src_test,
        "target_full": nc_tgt,
        "note": "WCVR on target_full confounded by ID count; see size_matched_summary.m3_target_wcvr",
    }
    results["size_matched_summary"]["source_reference"]["m3_source_test_wcvr"] = nc_src_test["wcvr"]
    print(f"  WCVR src-train:{nc_src_train['wcvr']:.3f}  src-test:{nc_src_test['wcvr']:.3f}  tgt-full:{nc_tgt['wcvr']:.3f}  tgt-matched:{np.mean(m3_wcvr_matched):.3f}±{np.std(m3_wcvr_matched):.3f}")

    # ------------------------------------------------------------------
    # M4: Hubness (full datasets)
    # ------------------------------------------------------------------
    print("\nM4: Hubness skew (full)...")
    idx_src_hub = rng.choice(len(src_test_emb), min(MAX_HUB, len(src_test_emb)), replace=False)
    hub_src = hubness_skew(src_test_emb[idx_src_hub], k=args.hubness_k)
    idx_tgt_hub = rng.choice(len(tgt_emb), min(MAX_HUB, len(tgt_emb)), replace=False)
    hub_tgt = hubness_skew(tgt_emb[idx_tgt_hub], k=args.hubness_k)

    results["m4_hubness"] = {
        "source": hub_src,
        "target_full": hub_tgt,
        "note": "Full-target hubness confounded by sample size; see size_matched_summary.m4_target_hubness_skew",
    }
    results["size_matched_summary"]["source_reference"]["m4_source_hubness_skew"] = hub_src["skewness"]
    print(f"  Skew src:{hub_src['skewness']:.3f}  tgt-full:{hub_tgt['skewness']:.3f}  tgt-matched:{np.mean(m4_skew_matched):.3f}±{np.std(m4_skew_matched):.3f}")

    # ------------------------------------------------------------------
    # M5: Variance decomposition (fixed estimator)
    # ------------------------------------------------------------------
    print("\nM5: Variance decomposition (corrected)...")
    vd_src = variance_decomposition(src_test_emb, src_test_ids, src_test_cids)
    vd_tgt = variance_decomposition(tgt_emb, tgt_ids, tgt_cids)

    results["m5_variance_decomp"] = {
        "source": vd_src,
        "target_full": vd_tgt,
        "note": "Full-target confounded by ID/camera count; see size_matched_summary",
    }
    results["size_matched_summary"]["source_reference"]["m5_source_identity_r2"] = vd_src["identity_r2"]
    results["size_matched_summary"]["source_reference"]["m5_source_camera_r2"] = vd_src["camera_r2"]
    print(f"  Source  identity R²:{vd_src['identity_r2']:.4f}  camera R²:{vd_src['camera_r2']:.4f}  ratio:{vd_src['identity_r2']/(vd_src['camera_r2']+1e-8):.1f}×")
    print(f"  Tgt-full  identity R²:{vd_tgt['identity_r2']:.4f}  camera R²:{vd_tgt['camera_r2']:.4f}  ratio:{vd_tgt['identity_r2']/(vd_tgt['camera_r2']+1e-8):.1f}×")
    print(f"  Tgt-matched  id R²:{np.mean(m5_id_r2_matched):.4f}  cam R²:{np.mean(m5_cam_r2_matched):.4f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
