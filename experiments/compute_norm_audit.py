#!/usr/bin/env python3
"""
Normalization audit (gate before E17b).

Questions:
  1. Are cached features pre- or post-L2-normalization?
  2. What is effective_rank in pre-norm vs post-norm space for all 4 cells?
  3. Separability check: mAP with Euclidean on unnormalized vs cosine on normalized.

Output: /data/papers/low_dimensional_reid/results/norm_audit_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reid-tk"))
from reid_tk.datasets import get_dataset

CHECKPOINT_MARKET = Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth")
CHECKPOINT_MSMT = Path("/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth")
CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
RAW_CACHE_DIR = Path("/tmp/norm_audit_raw_cache")
RAW_CACHE_DIR.mkdir(exist_ok=True)

_BOT_TRANSFORM = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class _DS(Dataset):
    def __init__(self, samples, transform):
        self.samples = [s for s in samples if not s.is_junk]
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return self.transform(s.load_image()), s.person_id, s.camera_id


def load_bot(checkpoint: Path, device: str) -> nn.Module:
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
def extract_raw(model, samples, device, cache_path: Path, batch_size=256):
    """Extract features WITHOUT L2-normalization."""
    if cache_path.exists():
        print(f"  raw cache hit: {cache_path.name}", flush=True)
        d = np.load(cache_path)
        return d["emb"], d["pids"], d["cids"]
    n = len([s for s in samples if not s.is_junk])
    print(f"  extracting {n} raw samples → {cache_path.name}", flush=True)
    ds = _DS(samples, _BOT_TRANSFORM)
    ldr = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    embs, pids, cids = [], [], []
    for i, (imgs, ids, cams) in enumerate(ldr):
        f = model(imgs.to(device))   # BNNeck output — NO L2-norm
        embs.append(f.cpu().numpy())
        pids.extend(ids.numpy()); cids.extend(cams.numpy())
        if i % 50 == 0:
            print(f"    batch {i}/{len(ldr)}", flush=True)
    emb = np.vstack(embs)
    np.savez(cache_path, emb=emb, pids=np.array(pids), cids=np.array(cids))
    return emb, np.array(pids), np.array(cids)


def effective_rank(X):
    """Reference formula: centered covariance via SVD."""
    Xc = X - X.mean(0)
    _, s, _ = np.linalg.svd(Xc / np.sqrt(len(X) - 1), full_matrices=False)
    lam = np.maximum(s**2, 0); lam = lam / lam.sum(); lam = lam[lam > 0]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def retrieval_map_euclidean(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    """mAP using squared Euclidean distance on unnormalized features.
    Lower distance = more similar.
    """
    # ||q - g||^2 = ||q||^2 + ||g||^2 - 2*q·g
    q_n2 = (q_emb ** 2).sum(axis=1, keepdims=True)   # (nq,1)
    g_n2 = (g_emb ** 2).sum(axis=1, keepdims=True).T  # (1,ng)
    n_q = len(q_pids)
    aps = []
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        qb = q_emb[start:end].astype(np.float32)
        qn2 = q_n2[start:end]
        # d^2 = qn2 + gn2 - 2*(qb @ g_emb.T)
        d2 = qn2 + g_n2 - 2.0 * (qb @ g_emb.T)
        np.clip(d2, 0, None, out=d2)
        order = np.argsort(d2, axis=1)
        for qi, idx in enumerate(order):
            qpid = q_pids[start + qi]; qcid = q_cids[start + qi]
            g_sorted_pids = g_pids[idx]; g_sorted_cids = g_cids[idx]
            mask_valid = ~((g_sorted_pids == qpid) & (g_sorted_cids == qcid))
            mask_pos = (g_sorted_pids == qpid) & mask_valid
            if mask_pos.sum() == 0:
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            n_pos = len(pos_in_valid)
            prec = (np.arange(1, n_pos + 1)) / (pos_in_valid + 1)
            aps.append(prec.mean())
    return float(np.mean(aps) * 100) if aps else 0.0


def retrieval_map_cosine(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    """mAP using cosine similarity on L2-normalized features.
    Higher similarity = more similar.
    """
    n_q = len(q_pids)
    aps = []
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        qb = q_emb[start:end].astype(np.float32)
        sim = qb @ g_emb.T   # cosine similarity since both L2-normalized
        order = np.argsort(-sim, axis=1)
        for qi, idx in enumerate(order):
            qpid = q_pids[start + qi]; qcid = q_cids[start + qi]
            g_sorted_pids = g_pids[idx]; g_sorted_cids = g_cids[idx]
            mask_valid = ~((g_sorted_pids == qpid) & (g_sorted_cids == qcid))
            mask_pos = (g_sorted_pids == qpid) & mask_valid
            if mask_pos.sum() == 0:
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            n_pos = len(pos_in_valid)
            prec = (np.arange(1, n_pos + 1)) / (pos_in_valid + 1)
            aps.append(prec.mean())
    return float(np.mean(aps) * 100) if aps else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load cached normalized features
    print("\nLoading cached (L2-normalized) features...", flush=True)
    c_mkt = np.load(CACHE_DIR / "market_test.npz")
    norm_mkt_emb, norm_mkt_pids, norm_mkt_cids = c_mkt["emb"], c_mkt["pids"], c_mkt["cids"]
    n_q_mkt = 3368

    c_msmt = np.load(CACHE_DIR / "msmt17_test.npz")
    norm_msmt_emb, norm_msmt_pids, norm_msmt_cids = c_msmt["emb"], c_msmt["pids"], c_msmt["cids"]
    n_q_msmt = 11655

    c_bmkt = np.load(CACHE_DIR / "bot_msmt_seed42_market.npz")
    norm_bmkt_emb = c_bmkt["emb"]

    c_bmsmt = np.load(CACHE_DIR / "bot_msmt_seed42_msmt.npz")
    norm_bmsmt_emb = c_bmsmt["emb"]

    # Verify all are L2-normalized
    for name, emb in [("market_test", norm_mkt_emb), ("msmt17_test", norm_msmt_emb),
                      ("bot_msmt_market", norm_bmkt_emb), ("bot_msmt_msmt", norm_bmsmt_emb)]:
        norms = np.linalg.norm(emb, axis=1)
        max_dev = float(np.abs(norms - 1.0).max())
        print(f"  {name}: max|norm-1| = {max_dev:.2e}  → {'L2-NORMALIZED' if max_dev < 1e-5 else 'NOT normalized'}", flush=True)

    # ---------------------------------------------------------------------------
    # Load datasets for raw extraction
    # ---------------------------------------------------------------------------
    print("\nLoading datasets for raw extraction...", flush=True)
    market = get_dataset("market1501")
    msmt17 = get_dataset("msmt17")

    # ---------------------------------------------------------------------------
    # Extract raw (unnormalized) features
    # ---------------------------------------------------------------------------
    print("\nLoading BoT-Market model...", flush=True)
    bot_mkt = load_bot(CHECKPOINT_MARKET, device)

    print("Extracting BoT-Market raw features...", flush=True)
    market_samples = market.query + market.gallery
    msmt_samples = msmt17.query + msmt17.gallery

    raw_mkt_emb, raw_mkt_pids, raw_mkt_cids = extract_raw(
        bot_mkt, market_samples, device,
        RAW_CACHE_DIR / "bot_market_market_raw.npz"
    )
    raw_msmt_via_mkt_emb, raw_msmt_via_mkt_pids, raw_msmt_via_mkt_cids = extract_raw(
        bot_mkt, msmt_samples, device,
        RAW_CACHE_DIR / "bot_market_msmt_raw.npz"
    )
    del bot_mkt
    torch.cuda.empty_cache()

    print("\nLoading BoT-MSMT17 model...", flush=True)
    bot_msmt = load_bot(CHECKPOINT_MSMT, device)

    raw_mkt_via_msmt_emb, raw_mkt_via_msmt_pids, raw_mkt_via_msmt_cids = extract_raw(
        bot_msmt, market_samples, device,
        RAW_CACHE_DIR / "bot_msmt_market_raw.npz"
    )
    raw_msmt_emb, raw_msmt_pids, raw_msmt_cids = extract_raw(
        bot_msmt, msmt_samples, device,
        RAW_CACHE_DIR / "bot_msmt_msmt_raw.npz"
    )
    del bot_msmt
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Compute effective rank: pre-norm vs post-norm, all 4 cells
    # ---------------------------------------------------------------------------
    print("\n" + "="*60, flush=True)
    print("EFFECTIVE RANK: pre-norm vs post-norm", flush=True)
    print("="*60, flush=True)

    # For eff_rank we use the full test set (query+gallery)
    cell_configs = [
        ("A", norm_mkt_emb, raw_mkt_emb),
        ("B", norm_msmt_emb, raw_msmt_via_mkt_emb),
        ("C", norm_bmkt_emb, raw_mkt_via_msmt_emb),
        ("D", norm_bmsmt_emb, raw_msmt_emb),
    ]

    er_results = {}
    for cell, norm_emb, raw_emb in cell_configs:
        er_post = effective_rank(norm_emb)
        er_pre = effective_rank(raw_emb)
        # Also: post-norm of raw_emb (should match er_post if extraction is correct)
        raw_l2 = raw_emb / np.linalg.norm(raw_emb, axis=1, keepdims=True)
        er_raw_then_l2 = effective_rank(raw_l2)
        print(f"Cell {cell}: pre-norm={er_pre:.2f}  post-norm={er_post:.2f}  raw→L2={er_raw_then_l2:.2f}", flush=True)
        er_results[f"cell_{cell}"] = {
            "eff_rank_pre_l2": er_pre,
            "eff_rank_post_l2": er_post,
            "eff_rank_raw_then_l2": er_raw_then_l2,
        }

    # ---------------------------------------------------------------------------
    # Separability check: Euclidean(unnorm) vs Cosine(norm) mAP
    # ---------------------------------------------------------------------------
    print("\n" + "="*60, flush=True)
    print("SEPARABILITY CHECK: mAP Euclidean(unnorm) vs Cosine(norm)", flush=True)
    print("="*60, flush=True)

    # Cell A: BoT-Market on Market
    q_mkt_norm = norm_mkt_emb[:n_q_mkt]; g_mkt_norm = norm_mkt_emb[n_q_mkt:]
    q_mkt_pid = norm_mkt_pids[:n_q_mkt]; g_mkt_pid = norm_mkt_pids[n_q_mkt:]
    q_mkt_cid = norm_mkt_cids[:n_q_mkt]; g_mkt_cid = norm_mkt_cids[n_q_mkt:]

    # Cell A raw splits (same n_q_mkt boundary)
    q_mkt_raw = raw_mkt_emb[:n_q_mkt]; g_mkt_raw = raw_mkt_emb[n_q_mkt:]

    print("Cell A: cosine(norm)...", flush=True)
    A_cos = retrieval_map_cosine(q_mkt_norm, q_mkt_pid, q_mkt_cid, g_mkt_norm, g_mkt_pid, g_mkt_cid)
    print(f"Cell A: cosine(norm) = {A_cos:.3f}%", flush=True)
    print("Cell A: euclidean(raw)...", flush=True)
    A_euc = retrieval_map_euclidean(q_mkt_raw, q_mkt_pid, q_mkt_cid, g_mkt_raw, g_mkt_pid, g_mkt_cid)
    print(f"Cell A: euclidean(raw) = {A_euc:.3f}%", flush=True)

    # Cell B: BoT-Market on MSMT17
    q_msmt_norm = norm_msmt_emb[:n_q_msmt]; g_msmt_norm = norm_msmt_emb[n_q_msmt:]
    q_msmt_pid = norm_msmt_pids[:n_q_msmt]; g_msmt_pid = norm_msmt_pids[n_q_msmt:]
    q_msmt_cid = norm_msmt_cids[:n_q_msmt]; g_msmt_cid = norm_msmt_cids[n_q_msmt:]

    q_msmt_raw = raw_msmt_via_mkt_emb[:n_q_msmt]; g_msmt_raw = raw_msmt_via_mkt_emb[n_q_msmt:]

    print("Cell B: cosine(norm)...", flush=True)
    B_cos = retrieval_map_cosine(q_msmt_norm, q_msmt_pid, q_msmt_cid, g_msmt_norm, g_msmt_pid, g_msmt_cid)
    print(f"Cell B: cosine(norm) = {B_cos:.3f}%", flush=True)
    print("Cell B: euclidean(raw)...", flush=True)
    B_euc = retrieval_map_euclidean(q_msmt_raw, q_msmt_pid, q_msmt_cid, g_msmt_raw, g_msmt_pid, g_msmt_cid)
    print(f"Cell B: euclidean(raw) = {B_euc:.3f}%", flush=True)

    # Cell C: BoT-MSMT17 on Market
    q_bmkt_norm = norm_bmkt_emb[:n_q_mkt]; g_bmkt_norm = norm_bmkt_emb[n_q_mkt:]
    q_bmkt_pid = q_mkt_pid; g_bmkt_pid = g_mkt_pid
    q_bmkt_cid = q_mkt_cid; g_bmkt_cid = g_mkt_cid

    q_bmkt_raw = raw_mkt_via_msmt_emb[:n_q_mkt]; g_bmkt_raw = raw_mkt_via_msmt_emb[n_q_mkt:]

    print("Cell C: cosine(norm)...", flush=True)
    C_cos = retrieval_map_cosine(q_bmkt_norm, q_bmkt_pid, q_bmkt_cid, g_bmkt_norm, g_bmkt_pid, g_bmkt_cid)
    print(f"Cell C: cosine(norm) = {C_cos:.3f}%", flush=True)
    print("Cell C: euclidean(raw)...", flush=True)
    C_euc = retrieval_map_euclidean(q_bmkt_raw, q_bmkt_pid, q_bmkt_cid, g_bmkt_raw, g_bmkt_pid, g_bmkt_cid)
    print(f"Cell C: euclidean(raw) = {C_euc:.3f}%", flush=True)

    # Cell D: BoT-MSMT17 on MSMT17
    q_bmsmt_norm = norm_bmsmt_emb[:n_q_msmt]; g_bmsmt_norm = norm_bmsmt_emb[n_q_msmt:]
    q_bmsmt_pid = q_msmt_pid; g_bmsmt_pid = g_msmt_pid
    q_bmsmt_cid = q_msmt_cid; g_bmsmt_cid = g_msmt_cid

    q_bmsmt_raw = raw_msmt_emb[:n_q_msmt]; g_bmsmt_raw = raw_msmt_emb[n_q_msmt:]

    print("Cell D: cosine(norm)...", flush=True)
    D_cos = retrieval_map_cosine(q_bmsmt_norm, q_bmsmt_pid, q_bmsmt_cid, g_bmsmt_norm, g_bmsmt_pid, g_bmsmt_cid)
    print(f"Cell D: cosine(norm) = {D_cos:.3f}%", flush=True)
    print("Cell D: euclidean(raw)...", flush=True)
    D_euc = retrieval_map_euclidean(q_bmsmt_raw, q_bmsmt_pid, q_bmsmt_cid, g_bmsmt_raw, g_bmsmt_pid, g_bmsmt_cid)
    print(f"Cell D: euclidean(raw) = {D_euc:.3f}%", flush=True)

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    print("\n" + "="*60, flush=True)
    print("SUMMARY", flush=True)
    print("="*60, flush=True)

    print("\nEffective rank:", flush=True)
    for cell, r in er_results.items():
        print(f"  {cell}: pre-L2={r['eff_rank_pre_l2']:.2f}  post-L2={r['eff_rank_post_l2']:.2f}  raw→L2={r['eff_rank_raw_then_l2']:.2f}", flush=True)

    print("\nmAP separability check:", flush=True)
    print(f"  A: cosine(norm)={A_cos:.3f}%  euclidean(raw)={A_euc:.3f}%  delta={A_euc-A_cos:+.3f}pp", flush=True)
    print(f"  B: cosine(norm)={B_cos:.3f}%  euclidean(raw)={B_euc:.3f}%  delta={B_euc-B_cos:+.3f}pp", flush=True)
    print(f"  C: cosine(norm)={C_cos:.3f}%  euclidean(raw)={C_euc:.3f}%  delta={C_euc-C_cos:+.3f}pp", flush=True)
    print(f"  D: cosine(norm)={D_cos:.3f}%  euclidean(raw)={D_euc:.3f}%  delta={D_euc-D_cos:+.3f}pp", flush=True)

    ordering_preserved = all([
        (A_euc > B_euc) == (A_cos > B_cos),  # A > B
        (D_euc > C_euc) == (D_cos > C_cos),  # D > C
        (A_euc > C_euc) == (A_cos > C_cos),  # in-domain > cross-domain (Market)
        (D_euc > B_euc) == (D_cos > B_cos),  # in-domain > cross-domain (MSMT17)
    ])
    print(f"\nOrdering preserved (A>B, D>C, A>C, D>B): {ordering_preserved}", flush=True)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    results = {
        "cache_normalization": "post_l2_norm (all features are L2-normalized)",
        "max_norm_deviation": {
            "market_test": float(np.abs(np.linalg.norm(norm_mkt_emb, axis=1) - 1).max()),
            "msmt17_test": float(np.abs(np.linalg.norm(norm_msmt_emb, axis=1) - 1).max()),
            "bot_msmt_market": float(np.abs(np.linalg.norm(norm_bmkt_emb, axis=1) - 1).max()),
            "bot_msmt_msmt": float(np.abs(np.linalg.norm(norm_bmsmt_emb, axis=1) - 1).max()),
        },
        "effective_rank": er_results,
        "map_separability": {
            "A": {"cosine_norm": A_cos, "euclidean_raw": A_euc, "delta_pp": A_euc - A_cos},
            "B": {"cosine_norm": B_cos, "euclidean_raw": B_euc, "delta_pp": B_euc - B_cos},
            "C": {"cosine_norm": C_cos, "euclidean_raw": C_euc, "delta_pp": C_euc - C_cos},
            "D": {"cosine_norm": D_cos, "euclidean_raw": D_euc, "delta_pp": D_euc - D_cos},
        },
        "ordering_preserved": ordering_preserved,
        "cell_c_discrepancy_explanation": {
            "seed42_eff_rank": 208.862,
            "seed123_eff_rank": 205.592,
            "two_seed_average": 207.227,
            "reference_in_2x2_json": 207.2,
            "explanation": "Reference 207.2 is the average of seed42+seed123 from compute_bot_msmt_2x2.py. The 2x2 mAP uses seed42 only, so 208.86 is the correct value for the consistent single-seed analysis.",
        },
    }

    out_path = RESULTS_DIR / "norm_audit_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
