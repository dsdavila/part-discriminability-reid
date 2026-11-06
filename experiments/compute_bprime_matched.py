"""
B'_matched: no-REA Market model evaluated on MSMT17 test, then same E27 50-draw
stratified subsampling used for A', B, C, D matched asymmetry.

Recipe-matched normalised matched asymmetry:
  (C_matched / A'_matched) / (B'_matched / D_matched)

where:
  A' = no-REA Market → Market (A' mean from E27 50 draws: need to compute)
  B' = no-REA Market → MSMT17 (this script)
  C  = no-REA MSMT17 → Market  (from e12 cache; matched mean already known: ~26.66)
  D  = no-REA MSMT17 → MSMT17  (from e12 cache; matched mean already known: ~41.62)

We also need A'_matched (no-REA Market → Market, matched). From context, A=80.40%
unmatched. E27 already ran with E27_matched ~ 79.40% for A (close to unmatched since
Market is so high). Let's compute A' properly.

Cache files:
  /tmp/e12_feature_cache/bot_msmt_seed42_market.npz  → C cell (nq=3368, ng=13115)
  /tmp/e12_feature_cache/bot_msmt_seed42_msmt.npz    → D cell (nq=11659, ng=82128)
  /tmp/e12_feature_cache/market_test.npz             → A cell (with-REA) — NOT used
  /tmp/e12_feature_cache/msmt17_test.npz             → B cell (with-REA) — NOT used

For B': load rea_ablation/bot_resnet50_market1501_seed42_final.pth, extract on MSMT17.
For A': load same checkpoint, extract on Market-1501.
"""

import sys, os, json, time
sys.path.insert(0, '/workspace/reid-tk')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict

from scripts.train_bag_of_tricks import BagOfTricksModel

DEVICE = 'cpu'  # GPU occupied by E16 training
RNG = np.random.default_rng(42)
N_DRAWS = 50
N_GALLERY_TARGET = 13115  # same as Market gallery size (used in E27)
N_QUERY_TARGET   = 3368   # same as Market query size
BATCH = 64

CKPT = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')
MSMT_ROOT = Path('/data/datasets/reid/MSMT17_V1')
MARKET_ROOT = Path('/data/datasets/reid/Market-1501-v15.09.15')
CACHE_DIR = Path('/tmp/e12_feature_cache')
OUT_JSON = Path('/data/papers/low_dimensional_reid/results/bprime_matched_results.json')
BPRIME_CACHE = Path('/tmp/bprime_msmt_embeddings.npz')   # cache B' to avoid re-extraction
APRIME_CACHE = Path('/tmp/aprime_mkt_embeddings.npz')    # cache A' to avoid re-extraction

# ── transforms ──────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class SimpleReIDDataset(Dataset):
    def __init__(self, img_paths, pids, cids):
        self.img_paths = img_paths
        self.pids = pids
        self.cids = cids
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert('RGB')
        return transform(img), self.pids[i], self.cids[i]

def parse_market_dir(split_dir):
    """Parse Market-1501 query or gallery dir. Returns paths, pids, cids."""
    paths, pids, cids = [], [], []
    for fname in sorted(os.listdir(split_dir)):
        if not fname.endswith('.jpg'): continue
        try:
            pid = int(fname.split('_')[0])
        except ValueError:
            continue
        if pid == -1: continue
        cam = int(fname[6]) - 1
        paths.append(os.path.join(split_dir, fname))
        pids.append(pid)
        cids.append(cam)
    return paths, np.array(pids), np.array(cids)

def parse_msmt_list(list_file, root):
    """Parse MSMT17 list file. Images live under root/test/. Returns paths, pids, cids."""
    paths, pids, cids = [], [], []
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            rel_path, pid = parts[0], int(parts[1])
            fname = Path(rel_path).name  # e.g. "0000_000_01_0303morning_0015_0.jpg"
            cam = int(fname.split('_')[2]) - 1
            paths.append(str(root / 'test' / rel_path))
            pids.append(pid)
            cids.append(cam)
    return paths, np.array(pids), np.array(cids)

def extract_features(model, paths, pids, cids):
    ds = SimpleReIDDataset(paths, pids, cids)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
    feats = []
    model.eval()
    with torch.no_grad():
        for imgs, _, _ in dl:
            f = model(imgs.to(DEVICE))
            f = F.normalize(f, dim=1)
            feats.append(f.cpu().numpy())
    return np.vstack(feats)

def compute_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Vectorised mAP: remove same-cam same-id from gallery."""
    sims = q_emb @ g_emb.T  # (nq, ng)
    aps = []
    for i in range(len(q_emb)):
        pid, cid = q_pids[i], q_cids[i]
        # valid gallery: different pid OR different cam
        valid = (g_pids != pid) | (g_cids != cid)
        # positives: same pid, different cam
        pos   = (g_pids == pid) & (g_cids != cid)
        if not pos[valid].any():
            continue
        scores = sims[i][valid]
        labels = pos[valid].astype(np.float32)
        order = np.argsort(-scores)
        labels = labels[order]
        cum = np.cumsum(labels)
        rank = np.arange(1, len(labels)+1, dtype=np.float32)
        prec = cum / rank
        ap = (prec * labels).sum() / labels.sum()
        aps.append(float(ap))
    return float(np.mean(aps)) * 100.0

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading no-REA Market checkpoint (751 cls, old flat-key format)...", flush=True)
sd_raw = torch.load(CKPT, map_location='cpu')
num_classes = 751

model = BagOfTricksModel(num_classes=num_classes, last_stride=1, backbone='resnet50').to(DEVICE)

# Remap old flat keys: "0.X" → "base.X",  "3.X" → "bottleneck.X"
# Exclude num_batches_tracked (BN layers track this internally).
# Classifier weight absent in old checkpoint — eval mode doesn't use it.
def remap_old_keys(sd_raw):
    new_sd = {}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k:
            continue
        if k.startswith('0.'):
            new_k = 'base.' + k[2:]
        elif k.startswith('3.'):
            new_k = 'bottleneck.' + k[2:]
        else:
            new_k = k
        new_sd[new_k] = v
    return new_sd

new_sd = remap_old_keys(sd_raw)
missing = model.load_state_dict(new_sd, strict=False)
print(f"  Remapped load OK. Missing={missing.missing_keys[:3]}, unexpected={missing.unexpected_keys[:3]}", flush=True)

model.eval()
print("Model loaded.", flush=True)

# ── Extract B' features: no-REA Market → MSMT17 ─────────────────────────────
if BPRIME_CACHE.exists():
    print(f"\nLoading B' embeddings from cache {BPRIME_CACHE}...", flush=True)
    c = np.load(BPRIME_CACHE)
    q_emb_msmt, q_pids, q_cids = c['q_emb'], c['q_pids'], c['q_cids']
    g_emb_msmt, g_pids, g_cids = c['g_emb'], c['g_pids'], c['g_cids']
    bprime_full = float(c['bprime_full'])
    print(f"  B'_full mAP = {bprime_full:.4f}%  (cached)", flush=True)
else:
    print("\nExtracting B' (no-REA Market → MSMT17)...", flush=True)
    list_query = MSMT_ROOT / 'list_query.txt'
    list_gallery = MSMT_ROOT / 'list_gallery.txt'
    q_paths, q_pids_msmt, q_cids_msmt = parse_msmt_list(list_query, MSMT_ROOT)
    g_paths, g_pids_msmt, g_cids_msmt = parse_msmt_list(list_gallery, MSMT_ROOT)
    q_pids, q_cids = q_pids_msmt, q_cids_msmt
    g_pids, g_cids = g_pids_msmt, g_cids_msmt
    print(f"  MSMT17 query={len(q_paths)}, gallery={len(g_paths)}", flush=True)

    t0 = time.time()
    q_emb_msmt = extract_features(model, q_paths, q_pids, q_cids)
    print(f"  Queries done in {(time.time()-t0)/60:.1f}m", flush=True)
    t0 = time.time()
    g_emb_msmt = extract_features(model, g_paths, g_pids, g_cids)
    print(f"  Gallery done in {(time.time()-t0)/60:.1f}m", flush=True)

    print("  Computing full B' mAP...", flush=True)
    bprime_full = compute_map(q_emb_msmt, q_pids, q_cids, g_emb_msmt, g_pids, g_cids)
    print(f"  B'_full mAP = {bprime_full:.4f}%", flush=True)

    # Save cache
    np.savez_compressed(BPRIME_CACHE, q_emb=q_emb_msmt, q_pids=q_pids, q_cids=q_cids,
                        g_emb=g_emb_msmt, g_pids=g_pids, g_cids=g_cids,
                        bprime_full=np.array(bprime_full))
    print(f"  Cached to {BPRIME_CACHE}", flush=True)

# ── Extract A' features: no-REA Market → Market ─────────────────────────────
if APRIME_CACHE.exists():
    print(f"\nLoading A' embeddings from cache {APRIME_CACHE}...", flush=True)
    ca = np.load(APRIME_CACHE)
    q_emb_mkt, mq_pids, mq_cids = ca['q_emb'], ca['q_pids'], ca['q_cids']
    g_emb_mkt, mg_pids, mg_cids = ca['g_emb'], ca['g_pids'], ca['g_cids']
    aprime_full = float(ca['aprime_full'])
    print(f"  A'_full mAP = {aprime_full:.4f}%  (cached)", flush=True)
else:
    print("\nExtracting A' (no-REA Market → Market)...", flush=True)
    mq_paths, mq_pids, mq_cids = parse_market_dir(str(MARKET_ROOT / 'query'))
    mg_paths, mg_pids, mg_cids = parse_market_dir(str(MARKET_ROOT / 'bounding_box_test'))
    print(f"  Market query={len(mq_paths)}, gallery={len(mg_paths)}", flush=True)

    t0 = time.time()
    q_emb_mkt = extract_features(model, mq_paths, mq_pids, mq_cids)
    g_emb_mkt = extract_features(model, mg_paths, mg_pids, mg_cids)
    print(f"  Market extraction done in {(time.time()-t0)/60:.1f}m", flush=True)

    aprime_full = compute_map(q_emb_mkt, mq_pids, mq_cids, g_emb_mkt, mg_pids, mg_cids)
    print(f"  A'_full mAP = {aprime_full:.4f}%", flush=True)
    np.savez_compressed(APRIME_CACHE, q_emb=q_emb_mkt, q_pids=mq_pids, q_cids=mq_cids,
                        g_emb=g_emb_mkt, g_pids=mg_pids, g_cids=mg_cids,
                        aprime_full=np.array(aprime_full))
    print(f"  Cached to {APRIME_CACHE}", flush=True)

# ── Load C and D from e12 cache ──────────────────────────────────────────────
print("\nLoading C, D from e12 cache...", flush=True)
# e12 cache format: emb[N_Q + N_G, 2048], pids, cids — query rows first
NQ_MKT  = 3368
NQ_MSMT = 11659
c_cache = np.load(CACHE_DIR / 'bot_msmt_seed42_market.npz')
d_cache = np.load(CACHE_DIR / 'bot_msmt_seed42_msmt.npz')

cq_emb  = c_cache['emb'][:NQ_MKT].astype(np.float32)
cq_pids = c_cache['pids'][:NQ_MKT]
cq_cids = c_cache['cids'][:NQ_MKT]
cg_emb  = c_cache['emb'][NQ_MKT:].astype(np.float32)
cg_pids = c_cache['pids'][NQ_MKT:]
cg_cids = c_cache['cids'][NQ_MKT:]

dq_emb  = d_cache['emb'][:NQ_MSMT].astype(np.float32)
dq_pids = d_cache['pids'][:NQ_MSMT]
dq_cids = d_cache['cids'][:NQ_MSMT]
dg_emb  = d_cache['emb'][NQ_MSMT:].astype(np.float32)
dg_pids = d_cache['pids'][NQ_MSMT:]
dg_cids = d_cache['cids'][NQ_MSMT:]

print(f"  C: nq={len(cq_emb)}, ng={len(cg_emb)}", flush=True)
print(f"  D: nq={len(dq_emb)}, ng={len(dg_emb)}", flush=True)

# ── 50-draw stratified subsampling for matched asymmetry ─────────────────────
# E27 protocol: sample N_GALLERY_TARGET gallery imgs from MSMT17, take all
# queries whose IDs appear in sampled gallery (capped at N_QUERY_TARGET).
# Same protocol applied to Market for A'/C.

def sample_matched(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids,
                   n_gal_target, n_q_target, rng, n_draws=50):
    """50-draw stratified gallery subsample → matched mAP distribution."""
    mAPs = []
    unique_ids = np.unique(g_pids)
    for draw_i in range(n_draws):
        if len(g_emb) > n_gal_target:
            # Stratified: sample uniformly from gallery
            sampled_idx = rng.choice(len(g_emb), n_gal_target, replace=False)
            sg_emb  = g_emb[sampled_idx]
            sg_pids = g_pids[sampled_idx]
            sg_cids = g_cids[sampled_idx]
        else:
            sg_emb, sg_pids, sg_cids = g_emb, g_pids, g_cids

        # Queries: all query IDs that appear in sampled gallery
        gal_ids = set(sg_pids.tolist())
        qmask = np.isin(q_pids, list(gal_ids))
        sq_emb  = q_emb[qmask]
        sq_pids = q_pids[qmask]
        sq_cids = q_cids[qmask]

        if n_q_target and len(sq_emb) > n_q_target:
            qi = rng.choice(len(sq_emb), n_q_target, replace=False)
            sq_emb  = sq_emb[qi]
            sq_pids = sq_pids[qi]
            sq_cids = sq_cids[qi]

        if len(sq_emb) == 0:
            continue
        ap_val = compute_map(sq_emb, sq_pids, sq_cids, sg_emb, sg_pids, sg_cids)
        mAPs.append(ap_val)
        if (draw_i+1) % 10 == 0:
            print(f"    draw {draw_i+1}/{n_draws} mAP={ap_val:.2f}%", flush=True)
    arr = np.array(mAPs)
    return float(arr.mean()), float(arr.std()), arr.tolist()

print("\nRunning 50-draw matched sampling...", flush=True)

print("  A' (no-REA Market → Market, matched):", flush=True)
aprime_mean, aprime_std, aprime_draws = sample_matched(
    q_emb_mkt, mq_pids, mq_cids, g_emb_mkt, mg_pids, mg_cids,
    N_GALLERY_TARGET, N_QUERY_TARGET, RNG, N_DRAWS)
print(f"  A'_matched = {aprime_mean:.2f}% ± {aprime_std:.2f}%", flush=True)

print("  B' (no-REA Market → MSMT17, matched):", flush=True)
bprime_mean, bprime_std, bprime_draws = sample_matched(
    q_emb_msmt, q_pids, q_cids, g_emb_msmt, g_pids, g_cids,
    N_GALLERY_TARGET, N_QUERY_TARGET, RNG, N_DRAWS)
print(f"  B'_matched = {bprime_mean:.2f}% ± {bprime_std:.2f}%", flush=True)

print("  C (no-REA MSMT17 → Market, matched):", flush=True)
c_mean, c_std, c_draws = sample_matched(
    cq_emb, cq_pids, cq_cids, cg_emb, cg_pids, cg_cids,
    N_GALLERY_TARGET, N_QUERY_TARGET, RNG, N_DRAWS)
print(f"  C_matched = {c_mean:.2f}% ± {c_std:.2f}%", flush=True)

print("  D (no-REA MSMT17 → MSMT17, matched):", flush=True)
d_mean, d_std, d_draws = sample_matched(
    dq_emb, dq_pids, dq_cids, dg_emb, dg_pids, dg_cids,
    N_GALLERY_TARGET, N_QUERY_TARGET, RNG, N_DRAWS)
print(f"  D_matched = {d_mean:.2f}% ± {d_std:.2f}%", flush=True)

# ── Recipe-matched normalised matched asymmetry ───────────────────────────────
# (C/A') / (B'/D)  — recipe-matched version
# Both models are no-REA; asymmetry = transfer penalty ratio
norm_asym_matched = (c_mean / aprime_mean) / (bprime_mean / d_mean)
print(f"\nRecipe-matched normalised matched asymmetry:", flush=True)
print(f"  (C={c_mean:.2f}% / A'={aprime_mean:.2f}%) / (B'={bprime_mean:.2f}% / D={d_mean:.2f}%)", flush=True)
print(f"  = ({c_mean/aprime_mean:.4f}) / ({bprime_mean/d_mean:.4f})", flush=True)
print(f"  = {norm_asym_matched:.4f}×", flush=True)

# Also compute with-REA vs no-REA comparison (from context):
# A=85.84%, B=2.5614%, C=26.66%, D=41.62% (canonical 2x2, mixed recipes)
# A'=aprime_full (no-REA Market → Market)
print(f"\nRecipe crossings:", flush=True)
print(f"  Mixed-recipe B/D = {2.5614/41.62:.4f}  A/C = {85.84/26.66:.4f}", flush=True)
print(f"  No-REA B'/D = {bprime_full/41.62:.4f}  A'/C = {aprime_full/26.66:.4f}", flush=True)
print(f"  REA inflation on B: {2.5614/bprime_full:.3f}× (should be ~1.97×)", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    'aprime_full_pct': aprime_full,
    'bprime_full_pct': bprime_full,
    'aprime_matched_mean_pct': aprime_mean,
    'aprime_matched_std_pct':  aprime_std,
    'bprime_matched_mean_pct': bprime_mean,
    'bprime_matched_std_pct':  bprime_std,
    'c_matched_mean_pct': c_mean,
    'c_matched_std_pct':  c_std,
    'd_matched_mean_pct': d_mean,
    'd_matched_std_pct':  d_std,
    'norm_asym_matched': norm_asym_matched,
    'formula': '(C_matched/Aprime_matched) / (Bprime_matched/D_matched)',
    'aprime_draws': aprime_draws,
    'bprime_draws': bprime_draws,
    'c_draws': c_draws,
    'd_draws': d_draws,
    'checkpoint': str(CKPT),
    'n_draws': N_DRAWS,
    'rng_seed': 42,
}
OUT_JSON.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {OUT_JSON}", flush=True)
print("Done.", flush=True)
