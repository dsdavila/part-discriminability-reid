"""
E18 Phase 2: Shared-stretch fraction + containment/interpolation.

Phase 2a (from test covariances, no train features):
  - Shared-stretch fraction: for each cross-domain cell, fraction of gallery
    variance explained by the corresponding in-domain cell's top-k subspace.

Phase 2b (needs train features extracted here):
  - Per-ID prototype containment: does each test gallery ID centroid fall
    within the angular neighborhood of source train ID centroids?
  - Affine interpolation: can test gallery ID centroids be expressed as
    affine combos of source train ID prototypes?
"""
import numpy as np
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

sys.path.insert(0, "/workspace/reid-tk")
from reid_tk.datasets import get_dataset

CACHE_DIR = Path("/tmp/e12_feature_cache")
TRAIN_CACHE_DIR = Path("/tmp/e18_train_cache")
TRAIN_CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Cache stores combined query+gallery (query first).
N_QUERY_MARKET = 3368
N_QUERY_MSMT   = 11655

_BOT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Load test gallery features ───────────────────────────────────────────────

def load_gallery(fname, dataset):
    """Load combined npz and return gallery portion only."""
    d = np.load(CACHE_DIR / fname)
    n_q = N_QUERY_MARKET if dataset == "market" else N_QUERY_MSMT
    return d["emb"][n_q:], d["pids"][n_q:], d["cids"][n_q:]


# ─── Model loading ────────────────────────────────────────────────────────────

def _load_bot(checkpoint_path, device):
    resnet = models.resnet50(weights=None)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    base = nn.Sequential(*list(resnet.children())[:-2])
    bn = nn.BatchNorm1d(2048)
    bn.bias.requires_grad_(False)
    model = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bn)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    return model.to(device).eval()


# ─── Dataset for train splits ─────────────────────────────────────────────────

class _DS(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        return self.transform(img), s.person_id, s.camera_id


@torch.no_grad()
def extract_train(model, samples, device, cache_path, batch_size=256):
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"    loaded from cache: {len(d['pids'])} samples", flush=True)
        return d["emb"], d["pids"], d["cids"]
    ds = _DS(samples, _BOT_TRANSFORM)
    ldr = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    embs, pids, cids = [], [], []
    for i, (imgs, ids, cams) in enumerate(ldr):
        f = model(imgs.to(device))
        f = F.normalize(f, p=2, dim=1)
        embs.append(f.cpu().numpy())
        pids.extend(ids.numpy())
        cids.extend(cams.numpy())
        if i % 20 == 0:
            print(f"    batch {i}/{len(ldr)}", flush=True)
    emb = np.vstack(embs)
    np.savez(cache_path, emb=emb, pids=np.array(pids), cids=np.array(cids))
    return emb, np.array(pids), np.array(cids)


# ─── Phase 2a: Shared-stretch fraction ────────────────────────────────────────

def shared_stretch_fraction(src_emb, tgt_emb, k_vals=(32, 64, 128, 256, 512)):
    """Fraction of tgt gallery variance explained by src gallery's top-k subspace."""
    Xs = src_emb - src_emb.mean(0)
    _, _, Vt_s = np.linalg.svd(Xs / np.sqrt(len(src_emb) - 1), full_matrices=False)
    V_s = Vt_s.T  # (D, D)

    Xt = tgt_emb - tgt_emb.mean(0)
    _, s_t, _ = np.linalg.svd(Xt / np.sqrt(len(tgt_emb) - 1), full_matrices=False)
    lam_t = s_t**2
    total_var = lam_t.sum()

    results = {}
    for k in k_vals:
        V_k = V_s[:, :k]
        Xt_proj = Xt @ V_k
        proj_var = (Xt_proj**2).sum() / (len(tgt_emb) - 1)
        fraction = float(proj_var / total_var)
        results[str(k)] = fraction

    return results


# ─── Phase 2b: Containment + interpolation ────────────────────────────────────

def compute_per_id_centroids(emb, pids):
    unique_ids = np.unique(pids)
    centroids = []
    for pid in unique_ids:
        mask = pids == pid
        c = emb[mask].mean(0)
        c = c / (np.linalg.norm(c) + 1e-10)
        centroids.append(c)
    return np.array(centroids), unique_ids


def containment_fraction(src_centroids, tgt_centroids):
    """Mean max-cosine-similarity of each tgt centroid to its nearest src centroid."""
    sim = tgt_centroids @ src_centroids.T  # (n_tgt, n_src)
    max_sim = sim.max(axis=1)

    fractions = {}
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        fractions[str(thresh)] = float((max_sim > thresh).mean())

    return {
        "mean_max_sim": float(max_sim.mean()),
        "median_max_sim": float(np.median(max_sim)),
        "fractions": fractions,
    }


def affine_interpolation_error(src_centroids, tgt_centroids, k=20):
    """
    For each tgt centroid, fit best convex combination of k nearest src centroids.
    Report mean residual angular distance.
    """
    from scipy.optimize import minimize

    sim = tgt_centroids @ src_centroids.T  # (n_tgt, n_src)
    errors = []
    for i in range(len(tgt_centroids)):
        idx = np.argsort(-sim[i])[:k]
        S = src_centroids[idx].T  # (D, k)
        t = tgt_centroids[i]

        def obj(w):
            approx = S @ w
            approx = approx / (np.linalg.norm(approx) + 1e-10)
            return float(1 - np.dot(approx, t))

        w0 = np.ones(k) / k
        res = minimize(obj, w0,
                       method='SLSQP',
                       bounds=[(0, 1)] * k,
                       constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                       options={'maxiter': 100, 'ftol': 1e-6})
        errors.append(res.fun)

        if i % 100 == 0:
            print(f"      interpolation {i}/{len(tgt_centroids)}", flush=True)

    return {
        "mean_cos_error": float(np.mean(errors)),
        "median_cos_error": float(np.median(errors)),
        "p90_cos_error": float(np.percentile(errors, 90)),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

print("E18 Phase 2: Shared-stretch + containment/interpolation", flush=True)
print("=" * 60, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# Load test gallery features (gallery portion only)
print("\nLoading test gallery features...", flush=True)
A_emb, A_pids, A_cids = load_gallery("market_test.npz",          "market")  # BoT-Mkt / Market
B_emb, B_pids, B_cids = load_gallery("msmt17_test.npz",          "msmt")    # BoT-Mkt / MSMT17
C_emb, C_pids, C_cids = load_gallery("bot_msmt_seed42_market.npz", "market") # BoT-MSMT / Market
D_emb, D_pids, D_cids = load_gallery("bot_msmt_seed42_msmt.npz",   "msmt")   # BoT-MSMT / MSMT17
print(f"  A: {len(A_pids)} ({len(np.unique(A_pids))} IDs), B: {len(B_pids)} ({len(np.unique(B_pids))} IDs)", flush=True)
print(f"  C: {len(C_pids)} ({len(np.unique(C_pids))} IDs), D: {len(D_pids)} ({len(np.unique(D_pids))} IDs)", flush=True)

# ─── Phase 2a: Shared-stretch fraction ───────────────────────────────────────

print("\n" + "=" * 60, flush=True)
print("PHASE 2a: SHARED-STRETCH FRACTION", flush=True)
print("=" * 60, flush=True)

print("\nMarket eval — A (in-domain) vs C (cross-domain):", flush=True)
print("  A->C (src=A, tgt=C):", flush=True)
ss_AC = shared_stretch_fraction(A_emb, C_emb)
for k, v in ss_AC.items():
    print(f"    k={k}: {v:.4f}", flush=True)

print("  C->A (src=C, tgt=A):", flush=True)
ss_CA = shared_stretch_fraction(C_emb, A_emb)
for k, v in ss_CA.items():
    print(f"    k={k}: {v:.4f}", flush=True)

print("\nMSMT17 eval — D (in-domain) vs B (cross-domain):", flush=True)
print("  D->B (src=D, tgt=B):", flush=True)
ss_DB = shared_stretch_fraction(D_emb, B_emb)
for k, v in ss_DB.items():
    print(f"    k={k}: {v:.4f}", flush=True)

print("  B->D (src=B, tgt=D):", flush=True)
ss_BD = shared_stretch_fraction(B_emb, D_emb)
for k, v in ss_BD.items():
    print(f"    k={k}: {v:.4f}", flush=True)

# ─── Phase 2b: Containment/interpolation ─────────────────────────────────────

print("\n" + "=" * 60, flush=True)
print("PHASE 2b: TRAIN FEATURE EXTRACTION", flush=True)
print("=" * 60, flush=True)

market_ds = get_dataset("market1501", sample=False)
msmt_ds = get_dataset("msmt17", sample=False)
print(f"  Market train: {len(market_ds.train)} images", flush=True)
print(f"  MSMT17 train: {len(msmt_ds.train)} images", flush=True)

BOT_MARKET_CKPT = "/data/papers/low_dimensional_reid/pretrained/bot_market1501_final.pth"
BOT_MSMT_CKPT   = "/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth"

print("\nLoading BoT-Market model...", flush=True)
model_market = _load_bot(BOT_MARKET_CKPT, device)

print("  Extracting BoT-Market on Market train...", flush=True)
mkt_on_mkt_train_emb, mkt_on_mkt_train_pids, _ = extract_train(
    model_market, market_ds.train, device,
    TRAIN_CACHE_DIR / "bot_market_market_train.npz"
)
print(f"  done: {len(mkt_on_mkt_train_pids)} images", flush=True)

print("  Extracting BoT-Market on MSMT17 train...", flush=True)
mkt_on_msmt_train_emb, mkt_on_msmt_train_pids, _ = extract_train(
    model_market, msmt_ds.train, device,
    TRAIN_CACHE_DIR / "bot_market_msmt_train.npz"
)
print(f"  done: {len(mkt_on_msmt_train_pids)} images", flush=True)
del model_market
torch.cuda.empty_cache()

print("\nLoading BoT-MSMT17 model...", flush=True)
model_msmt = _load_bot(BOT_MSMT_CKPT, device)

print("  Extracting BoT-MSMT17 on Market train...", flush=True)
msmt_on_mkt_train_emb, msmt_on_mkt_train_pids, _ = extract_train(
    model_msmt, market_ds.train, device,
    TRAIN_CACHE_DIR / "bot_msmt_market_train.npz"
)
print(f"  done: {len(msmt_on_mkt_train_pids)} images", flush=True)

print("  Extracting BoT-MSMT17 on MSMT17 train...", flush=True)
msmt_on_msmt_train_emb, msmt_on_msmt_train_pids, _ = extract_train(
    model_msmt, msmt_ds.train, device,
    TRAIN_CACHE_DIR / "bot_msmt_msmt_train.npz"
)
print(f"  done: {len(msmt_on_msmt_train_pids)} images", flush=True)
del model_msmt
torch.cuda.empty_cache()

# Compute per-ID centroids
print("\nComputing per-ID centroids...", flush=True)
mkt_mkt_src,  _ = compute_per_id_centroids(mkt_on_mkt_train_emb,  mkt_on_mkt_train_pids)
mkt_msmt_src, _ = compute_per_id_centroids(mkt_on_msmt_train_emb, mkt_on_msmt_train_pids)
msmt_mkt_src, _ = compute_per_id_centroids(msmt_on_mkt_train_emb, msmt_on_mkt_train_pids)
msmt_msmt_src,_ = compute_per_id_centroids(msmt_on_msmt_train_emb,msmt_on_msmt_train_pids)

A_tgt, _ = compute_per_id_centroids(A_emb, A_pids)
B_tgt, _ = compute_per_id_centroids(B_emb, B_pids)
C_tgt, _ = compute_per_id_centroids(C_emb, C_pids)
D_tgt, _ = compute_per_id_centroids(D_emb, D_pids)

print(f"  Train centroids: mkt/mkt={len(mkt_mkt_src)}, mkt/msmt={len(mkt_msmt_src)}", flush=True)
print(f"                   msmt/mkt={len(msmt_mkt_src)}, msmt/msmt={len(msmt_msmt_src)}", flush=True)
print(f"  Test centroids:  A={len(A_tgt)}, B={len(B_tgt)}, C={len(C_tgt)}, D={len(D_tgt)}", flush=True)

# ─── Containment ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("CONTAINMENT (max cosine sim to nearest train centroid)", flush=True)
print("=" * 60, flush=True)

cont = {}
for label, src_c, tgt_c, desc in [
    ("A", mkt_mkt_src,  A_tgt, "BoT-Mkt train -> Market test (in-domain)"),
    ("B", mkt_msmt_src, B_tgt, "BoT-Mkt train -> MSMT17 test (cross-domain)"),
    ("C", msmt_mkt_src, C_tgt, "BoT-MSMT train -> Market test (cross-domain)"),
    ("D", msmt_msmt_src,D_tgt, "BoT-MSMT train -> MSMT17 test (in-domain)"),
]:
    print(f"\n  Cell {label} — {desc}:", flush=True)
    r = containment_fraction(src_c, tgt_c)
    print(f"    mean_max_sim={r['mean_max_sim']:.4f}  median={r['median_max_sim']:.4f}", flush=True)
    for thresh, frac in r['fractions'].items():
        print(f"    frac(sim>{thresh})={frac:.3f}", flush=True)
    cont[label] = r

# ─── Affine interpolation ─────────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("AFFINE INTERPOLATION (residual from convex combo of train centroids)", flush=True)
print("=" * 60, flush=True)

interp = {}
for label, src_c, tgt_c in [
    ("A", mkt_mkt_src,  A_tgt),
    ("B", mkt_msmt_src, B_tgt),
    ("C", msmt_mkt_src, C_tgt),
    ("D", msmt_msmt_src,D_tgt),
]:
    print(f"\n  Cell {label} ({len(tgt_c)} test IDs, {len(src_c)} train IDs):", flush=True)
    r = affine_interpolation_error(src_c, tgt_c, k=20)
    print(f"    mean_cos_error={r['mean_cos_error']:.4f}  median={r['median_cos_error']:.4f}  p90={r['p90_cos_error']:.4f}", flush=True)
    interp[label] = r

# ─── Save ─────────────────────────────────────────────────────────────────────
out = RESULTS_DIR / "e18_phase2_results.json"
results = {
    "shared_stretch": {
        "A->C": ss_AC, "C->A": ss_CA,
        "D->B": ss_DB, "B->D": ss_BD,
    },
    "containment": cont,
    "interpolation": interp,
}
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}")
