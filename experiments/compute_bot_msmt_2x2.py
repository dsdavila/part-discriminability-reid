#!/usr/bin/env python3
"""
Within-architecture 2×2: BoT ResNet-50 (Market-trained vs MSMT17-trained)
on Market and MSMT17 test sets.

Tests whether collapse tracks out-of-domain direction within a single
architecture+recipe, ruling out the ResNet-50 vs ViT-B/16 confound.

                    Market test    MSMT17 test
  BoT-Market        (cached)       (cached)
  BoT-MSMT17        [NEW seed42]   [NEW seed42]
                    [NEW seed123]  [NEW seed123]

Flip test: BoT-MSMT shows lower eff. rank / higher hubness on Market (OOD)
than on MSMT17 (in-domain)?  Same direction as CLIP-ReID → architecture-
independent.  Opposite → confounded by architecture.
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
from scipy.stats import skew
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reid-tk"))
from reid_tk.datasets import get_dataset

# ---------------------------------------------------------------------------
# BoT ResNet-50 extractor
# ---------------------------------------------------------------------------

def _load_bot(checkpoint: Path, device: str) -> nn.Module:
    resnet = models.resnet50(weights=None)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    base = nn.Sequential(*list(resnet.children())[:-2])
    bn = nn.BatchNorm1d(2048)
    bn.bias.requires_grad_(False)
    model = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bn)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    return model.to(device).eval()


_IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_BOT_TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(), _IMAGENET_NORM])


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class _DS(Dataset):
    def __init__(self, samples, transform):
        self.samples   = [s for s in samples if not s.is_junk]
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return self.transform(s.load_image()), s.person_id, s.camera_id


@torch.no_grad()
def extract(model, samples, device, cache_path: Path, batch_size=128):
    if cache_path.exists():
        print(f"  cache hit: {cache_path.name}")
        d = np.load(cache_path)
        return d["emb"], d["pids"], d["cids"]
    n = len([s for s in samples if not s.is_junk])
    print(f"  extracting {n} samples → {cache_path.name}")
    ds  = _DS(samples, _BOT_TRANSFORM)
    ldr = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    embs, pids, cids = [], [], []
    for imgs, ids, cams in ldr:
        f = model(imgs.to(device))
        embs.append(F.normalize(f, p=2, dim=1).cpu().numpy())
        pids.extend(ids.numpy()); cids.extend(cams.numpy())
    emb = np.vstack(embs)
    np.savez(cache_path, emb=emb, pids=np.array(pids), cids=np.array(cids))
    return emb, np.array(pids), np.array(cids)


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def effective_rank(X):
    Xc = X - X.mean(0)
    _, s, _ = np.linalg.svd(Xc / np.sqrt(len(X) - 1), full_matrices=False)
    lam = np.maximum(s**2, 0); lam = lam / lam.sum(); lam = lam[lam > 0]
    return float(np.exp(-np.sum(lam * np.log(lam))))


def hubness_skew(X, k=10, rng=None, n=5000):
    if rng is None: rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(n, len(X)), replace=False)
    Xs  = X[idx]; sim = Xs @ Xs.T; np.fill_diagonal(sim, -2.0)
    knn = np.argpartition(sim, -k, axis=1)[:, -k:]
    return float(skew(np.bincount(knn.ravel(), minlength=len(Xs))))


def variance_r2(X, ids, cids):
    mu = X.mean(0); total = float(np.mean(np.sum((X - mu)**2, axis=1))); N = len(X)
    def br2(labels):
        return sum(
            (labels == u).sum() / N * float(np.sum((X[labels == u].mean(0) - mu)**2))
            for u in np.unique(labels)
        ) / (total + 1e-8)
    return br2(ids), br2(cids)


def geometry(label, X, ids, cids, rng, ref_n_ids):
    er  = effective_rank(X)
    unique = np.unique(ids)
    Xh  = X[np.isin(ids, rng.choice(unique, ref_n_ids, replace=False))] if len(unique) > ref_n_ids else X
    hub = hubness_skew(Xh, rng=rng)
    idr, cr = variance_r2(X, ids, cids)
    return {"label": label, "n": len(X),
            "n_ids": int(len(unique)), "n_cams": int(len(np.unique(cids))),
            "eff_rank": er, "hubness_matched": hub,
            "identity_r2": idr, "camera_r2": cr,
            "ratio": idr / (cr + 1e-8)}


def avg_geo(label, geos):
    """Average geometry metrics across seed runs."""
    keys = ["eff_rank", "hubness_matched", "identity_r2", "camera_r2", "ratio"]
    out  = {"label": label, "n": geos[0]["n"], "n_ids": geos[0]["n_ids"],
            "n_cams": geos[0]["n_cams"], "n_seeds": len(geos)}
    for k in keys:
        vals = [g[k] for g in geos]
        out[k]          = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    return out


def print_row(g):
    std = g.get("eff_rank_std", 0)
    seed_note = f" (±{std:.1f}, n={g['n_seeds']} seeds)" if "n_seeds" in g else ""
    print(f"  {g['label']:<44} er={g['eff_rank']:6.1f}{seed_note}  "
          f"hub={g['hubness_matched']:5.3f}  "
          f"id={g['identity_r2']:.4f}  cam={g['camera_r2']:.4f}  ratio={g['ratio']:6.1f}×")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = Path("/tmp/e12_feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123]
    ckpts = {
        s: Path(f"/data/papers/low_dimensional_reid/pretrained/"
                f"bot_resnet50_msmt17_seed{s}_final.pth")
        for s in seeds
    }
    out_path = Path("/data/papers/low_dimensional_reid/results/bot_msmt_2x2_results.json")

    rng = np.random.default_rng(42)

    print("Loading datasets...")
    market = get_dataset("market1501", sample=False)
    msmt   = get_dataset("msmt17",     sample=False)
    market_samples = market.query + market.gallery
    msmt_samples   = msmt.query   + msmt.gallery

    ref_n_ids = len(np.unique([s.person_id for s in market_samples if not s.is_junk]))
    print(f"  ref_n_ids (Market): {ref_n_ids}")

    # ---- Row 1: BoT-Market (load from existing cache) ----------------------
    print("\n--- Row 1: BoT ResNet-50 (Market-trained) ---")
    bot_market_ckpt = Path("/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth")
    bot_market = _load_bot(bot_market_ckpt, device)
    emb_bm_mkt, pids_bm_mkt, cids_bm_mkt = extract(
        bot_market, market_samples, device, cache_dir / "market_test.npz")
    emb_bm_mmt, pids_bm_mmt, cids_bm_mmt = extract(
        bot_market, msmt_samples, device, cache_dir / "msmt17_test.npz")
    del bot_market; torch.cuda.empty_cache()

    geo_bm_mkt = geometry("BoT-Market / Market", emb_bm_mkt, pids_bm_mkt, cids_bm_mkt, rng, ref_n_ids)
    geo_bm_mmt = geometry("BoT-Market / MSMT17", emb_bm_mmt, pids_bm_mmt, cids_bm_mmt, rng, ref_n_ids)

    # ---- Row 2: BoT-MSMT17 (seed 42 + 123) --------------------------------
    print("\n--- Row 2: BoT ResNet-50 (MSMT17-trained, 2 seeds) ---")
    seed_geos_mkt, seed_geos_mmt = [], []

    for seed in seeds:
        print(f"\n  seed {seed}:")
        bot_msmt = _load_bot(ckpts[seed], device)
        emb_s_mkt, pids_s_mkt, cids_s_mkt = extract(
            bot_msmt, market_samples, device,
            cache_dir / f"bot_msmt_seed{seed}_market.npz")
        emb_s_mmt, pids_s_mmt, cids_s_mmt = extract(
            bot_msmt, msmt_samples, device,
            cache_dir / f"bot_msmt_seed{seed}_msmt.npz")
        del bot_msmt; torch.cuda.empty_cache()

        g_mkt = geometry(f"BoT-MSMT s{seed} / Market", emb_s_mkt, pids_s_mkt, cids_s_mkt, rng, ref_n_ids)
        g_mmt = geometry(f"BoT-MSMT s{seed} / MSMT17", emb_s_mmt, pids_s_mmt, cids_s_mmt, rng, ref_n_ids)
        seed_geos_mkt.append(g_mkt); seed_geos_mmt.append(g_mmt)
        print_row(g_mkt); print_row(g_mmt)

    geo_bmsmt_mkt = avg_geo("BoT-MSMT17 / Market (avg 2 seeds)", seed_geos_mkt)
    geo_bmsmt_mmt = avg_geo("BoT-MSMT17 / MSMT17 (avg 2 seeds)", seed_geos_mmt)

    # ---- Print 2×2 ---------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'Condition':<44} {'Eff.rank':>10} {'Hubness':>8} {'Id R²':>8} {'Cam R²':>8} {'ratio':>8}")
    print("=" * 100)
    for g in [geo_bm_mkt, geo_bm_mmt, geo_bmsmt_mkt, geo_bmsmt_mmt]:
        print_row(g)

    print("\n2×2 summary:")
    print(f"{'':30}  Market test              MSMT17 test")
    print(f"  BoT-Market             "
          f"er={geo_bm_mkt['eff_rank']:.0f}  hub={geo_bm_mkt['hubness_matched']:.2f}  ratio={geo_bm_mkt['ratio']:.0f}×  |  "
          f"er={geo_bm_mmt['eff_rank']:.0f}  hub={geo_bm_mmt['hubness_matched']:.2f}  ratio={geo_bm_mmt['ratio']:.0f}×")
    print(f"  BoT-MSMT17             "
          f"er={geo_bmsmt_mkt['eff_rank']:.0f}  hub={geo_bmsmt_mkt['hubness_matched']:.2f}  ratio={geo_bmsmt_mkt['ratio']:.0f}×  |  "
          f"er={geo_bmsmt_mmt['eff_rank']:.0f}  hub={geo_bmsmt_mmt['hubness_matched']:.2f}  ratio={geo_bmsmt_mmt['ratio']:.0f}×")

    # Flip test: BoT-MSMT17 on Market (OOD) vs MSMT17 (in-domain)
    flip_rank = geo_bmsmt_mkt["eff_rank"] < geo_bmsmt_mmt["eff_rank"]
    flip_hub  = geo_bmsmt_mkt["hubness_matched"] > geo_bmsmt_mmt["hubness_matched"]
    print(f"\nFlip test (BoT-MSMT17: lower rank on OOD Market, higher hub on OOD Market):")
    print(f"  Rank flips: {flip_rank}  Hub flips: {flip_hub}")

    # ---- Save --------------------------------------------------------------
    results = {
        "bot_market_on_market":  geo_bm_mkt,
        "bot_market_on_msmt17":  geo_bm_mmt,
        "bot_msmt_on_market":    geo_bmsmt_mkt,
        "bot_msmt_on_msmt17":    geo_bmsmt_mmt,
        "per_seed":              {"market": seed_geos_mkt, "msmt17": seed_geos_mmt},
        "flip_rank_bot_msmt":    bool(flip_rank),
        "flip_hub_bot_msmt":     bool(flip_hub),
        "note": (
            "flip_rank_bot_msmt=True: BoT-MSMT17 shows lower eff. rank on Market (OOD) "
            "than MSMT17 (in-domain) — collapse tracks out-of-domain, architecture-independent."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
