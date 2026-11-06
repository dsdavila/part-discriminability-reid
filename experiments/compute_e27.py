"""
E27: matched-difficulty asymmetry.
Subsample MSMT17 test set to Market's shape (750 IDs, ~13115 gallery, proportionate queries).
50 independent draws. Stratified by images-per-identity and cameras-per-identity.
Reports B_matched, D_matched, raw asymmetry C/B_matched, normalised (C/A)/(B_matched/D_matched).
"""
import sys; sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

CACHE = Path('/tmp/e12_feature_cache')
OUT   = Path('/data/papers/low_dimensional_reid/results')
N_DRAWS = 50
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# Market shape (targets)
MKT_N_IDS  = 750
MKT_N_GAL  = 13115
MKT_N_QUE  = 3368

def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)

def compute_map(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=256):
    q_emb = l2(q_emb.astype(np.float32))
    g_emb = l2(g_emb.astype(np.float32))
    N_Q, N_G = len(q_emb), len(g_emb)
    ap_list = []
    r1_list = []
    for s in range(0, N_Q, batch):
        e = min(s + batch, N_Q)
        sim = q_emb[s:e] @ g_emb.T
        for bi in range(e - s):
            qi = s + bi
            qpid = int(q_pids[qi])
            qcid = int(q_cids[qi])
            junk = (g_pids == qpid) & (g_cids == qcid)
            correct = (g_pids == qpid) & ~junk
            G = int(correct.sum())
            if G == 0: continue
            sv = sim[bi].copy()
            sv[junk] = -np.inf
            order = np.argsort(-sv)
            ranked = g_pids[order]
            valid = sv[order] > -1e30
            c_pos = np.where((ranked == qpid) & valid)[0]
            if len(c_pos) == 0: continue
            r = c_pos + 1
            ap = float(np.sum(np.arange(1, G+1) / r)) / G
            ap_list.append(ap)
            r1_list.append(1.0 if c_pos[0] == 0 else 0.0)
    return float(np.mean(ap_list) * 100), float(np.mean(r1_list) * 100)

# ── Load full MSMT17 test embeddings (gallery + queries) ──────────────────
print("Loading MSMT17 test embeddings...", flush=True)
msmt_full = np.load(CACHE / 'msmt17_test.npz')
msmt_emb  = msmt_full['emb'].astype(np.float32)
msmt_pids = msmt_full['pids']
msmt_cids = msmt_full['cids']
N_QUE_FULL = 11659   # first N rows are queries
msmt_q_emb  = msmt_emb[:N_QUE_FULL]
msmt_q_pids = msmt_pids[:N_QUE_FULL]
msmt_q_cids = msmt_cids[:N_QUE_FULL]
msmt_g_emb  = msmt_emb[N_QUE_FULL:]
msmt_g_pids = msmt_pids[N_QUE_FULL:]
msmt_g_cids = msmt_cids[N_QUE_FULL:]
print(f"  Queries: {len(msmt_q_pids)}, Gallery: {len(msmt_g_pids)}", flush=True)

# ── Load Market test (A and C cells) ─────────────────────────────────────
print("Loading Market test embeddings...", flush=True)
mkt_a = np.load(CACHE / 'market_test.npz')
mkt_q_emb  = mkt_a['emb'][:MKT_N_QUE].astype(np.float32)
mkt_q_pids = mkt_a['pids'][:MKT_N_QUE]
mkt_q_cids = mkt_a['cids'][:MKT_N_QUE]
mkt_g_emb  = mkt_a['emb'][MKT_N_QUE:].astype(np.float32)
mkt_g_pids = mkt_a['pids'][MKT_N_QUE:]
mkt_g_cids = mkt_a['cids'][MKT_N_QUE:]
mkt_b = np.load(CACHE / 'bot_msmt_seed42_market.npz')  # BoT-MSMT → Market (Cell C)
mkt_c_q_emb  = mkt_b['emb'][:MKT_N_QUE].astype(np.float32)
mkt_c_q_pids = mkt_b['pids'][:MKT_N_QUE]
mkt_c_q_cids = mkt_b['cids'][:MKT_N_QUE]
mkt_c_g_emb  = mkt_b['emb'][MKT_N_QUE:].astype(np.float32)
mkt_c_g_pids = mkt_b['pids'][MKT_N_QUE:]
mkt_c_g_cids = mkt_b['cids'][MKT_N_QUE:]

# ── Cell A and C (unchanged) ──────────────────────────────────────────────
print("Computing Cell A (BoT-Mkt → Market)...", flush=True)
A_map, A_r1 = compute_map(mkt_q_emb, mkt_q_pids, mkt_q_cids,
                           mkt_g_emb, mkt_g_pids, mkt_g_cids)
print(f"  A: mAP={A_map:.2f}% R1={A_r1:.2f}%", flush=True)

print("Computing Cell C (BoT-MSMT → Market)...", flush=True)
C_map, C_r1 = compute_map(mkt_c_q_emb, mkt_c_q_pids, mkt_c_q_cids,
                           mkt_c_g_emb, mkt_c_g_pids, mkt_c_g_cids)
print(f"  C: mAP={C_map:.2f}% R1={C_r1:.2f}%", flush=True)

# ── Load BoT-MSMT17 embeddings for Cell D gallery ─────────────────────────
print("Loading BoT-MSMT17 MSMT17 embeddings (Cell D)...", flush=True)
msmt_d = np.load(CACHE / 'bot_msmt_seed42_msmt.npz')
msmt_d_q_emb  = msmt_d['emb'][:N_QUE_FULL].astype(np.float32)
msmt_d_q_pids = msmt_d['pids'][:N_QUE_FULL]
msmt_d_q_cids = msmt_d['cids'][:N_QUE_FULL]
msmt_d_g_emb  = msmt_d['emb'][N_QUE_FULL:].astype(np.float32)
msmt_d_g_pids = msmt_d['pids'][N_QUE_FULL:]
msmt_d_g_cids = msmt_d['cids'][N_QUE_FULL:]

# ── Stratified subsampling of MSMT17 test ────────────────────────────────
# Gallery: choose MKT_N_IDS identities, ~MKT_N_GAL gallery images
# Stratify by: images-per-identity and cameras-per-identity distributions

# Compute per-identity stats in full gallery
gal_pid_to_gidxs = defaultdict(list)
for i, pid in enumerate(msmt_g_pids):
    gal_pid_to_gidxs[int(pid)].append(i)

pid_n_images = {pid: len(idxs) for pid, idxs in gal_pid_to_gidxs.items()}
pid_n_cams   = {pid: len(set(msmt_g_cids[idx] for idx in idxs))
                for pid, idxs in gal_pid_to_gidxs.items()}

all_gal_pids = sorted(gal_pid_to_gidxs.keys())
N_GAL_IDS = len(all_gal_pids)
print(f"MSMT17 gallery: {len(msmt_g_pids)} images, {N_GAL_IDS} identities", flush=True)

# Compute images-per-identity and cameras-per-identity distributions
img_counts = np.array([pid_n_images[p] for p in all_gal_pids])
cam_counts = np.array([pid_n_cams[p] for p in all_gal_pids])
# Stratify into bins: img_bins × cam_bins
img_bins = np.percentile(img_counts, [0, 33, 66, 100])
cam_bins = np.array([1, 2, 4, 10])  # camera count breakpoints

def get_strat_bin(img_c, cam_c):
    ib = np.searchsorted(img_bins[1:-1], img_c)
    cb = np.searchsorted(cam_bins[1:], cam_c)
    return (ib, cb)

strat_groups = defaultdict(list)
for i, pid in enumerate(all_gal_pids):
    b = get_strat_bin(img_counts[i], cam_counts[i])
    strat_groups[b].append(pid)

print(f"Stratification bins: {len(strat_groups)} groups", flush=True)
for k, v in sorted(strat_groups.items()):
    print(f"  bin {k}: {len(v)} IDs", flush=True)

B_maps, B_r1s = [], []
D_maps, D_r1s = [], []

for draw in range(N_DRAWS):
    # Sample MKT_N_IDS identities proportionally from strat groups
    chosen_pids = []
    total_groups = sum(len(v) for v in strat_groups.values())
    for b, pids in strat_groups.items():
        n_take = max(1, round(MKT_N_IDS * len(pids) / total_groups))
        take = list(rng.choice(pids, min(n_take, len(pids)), replace=False))
        chosen_pids.extend(take)
    # Adjust to exactly MKT_N_IDS
    chosen_pids = list(rng.choice(chosen_pids, min(MKT_N_IDS, len(chosen_pids)), replace=False))
    chosen_set = set(chosen_pids)

    # Gallery subset: all gallery images for chosen IDs
    gal_mask = np.array([int(p) in chosen_set for p in msmt_g_pids])
    g_candidate_idx = np.where(gal_mask)[0]
    # Subsample gallery to ~MKT_N_GAL images (proportional per identity)
    if len(g_candidate_idx) > MKT_N_GAL:
        chosen_g_idx = rng.choice(g_candidate_idx, MKT_N_GAL, replace=False)
        chosen_g_idx = np.sort(chosen_g_idx)
    else:
        chosen_g_idx = g_candidate_idx

    # Query subset: queries for chosen IDs (junk-excluded as usual)
    q_mask = np.array([int(p) in chosen_set for p in msmt_q_pids])
    chosen_q_idx = np.where(q_mask)[0]

    # Cell B (BoT-Market → MSMT17_subset)
    sg_emb  = msmt_g_emb[chosen_g_idx]
    sg_pids = msmt_g_pids[chosen_g_idx]
    sg_cids = msmt_g_cids[chosen_g_idx]
    sq_emb  = msmt_q_emb[chosen_q_idx]
    sq_pids = msmt_q_pids[chosen_q_idx]
    sq_cids = msmt_q_cids[chosen_q_idx]
    b_map, b_r1 = compute_map(sq_emb, sq_pids, sq_cids, sg_emb, sg_pids, sg_cids)
    B_maps.append(b_map); B_r1s.append(b_r1)

    # Cell D (BoT-MSMT17 → MSMT17_subset)
    sg_d_emb  = msmt_d_g_emb[chosen_g_idx]
    sg_d_pids = msmt_d_g_pids[chosen_g_idx]
    sg_d_cids = msmt_d_g_cids[chosen_g_idx]
    sq_d_emb  = msmt_d_q_emb[chosen_q_idx]
    sq_d_pids = msmt_d_q_pids[chosen_q_idx]
    sq_d_cids = msmt_d_q_cids[chosen_q_idx]
    d_map, d_r1 = compute_map(sq_d_emb, sq_d_pids, sq_d_cids, sg_d_emb, sg_d_pids, sg_d_cids)
    D_maps.append(d_map); D_r1s.append(d_r1)

    n_q = int(q_mask.sum()); n_g = len(chosen_g_idx)
    if draw % 10 == 0:
        print(f"  Draw {draw+1}/{N_DRAWS}: n_q={n_q} n_g={n_g} B={b_map:.2f}% D={d_map:.2f}%", flush=True)

B_maps = np.array(B_maps); D_maps = np.array(D_maps)
B_r1s  = np.array(B_r1s);  D_r1s  = np.array(D_r1s)

B_mean = float(B_maps.mean()); B_ci = [float(np.percentile(B_maps, 2.5)), float(np.percentile(B_maps, 97.5))]
D_mean = float(D_maps.mean()); D_ci = [float(np.percentile(D_maps, 2.5)), float(np.percentile(D_maps, 97.5))]

# Asymmetries
raw_asym = C_map / B_mean
normed_asym = (C_map / A_map) / (B_mean / D_mean)

# Per-draw asymmetries
per_draw_raw    = C_map / B_maps
per_draw_normed = (C_map / A_map) / (B_maps / D_maps)
raw_ci    = [float(np.percentile(per_draw_raw, 2.5)),    float(np.percentile(per_draw_raw, 97.5))]
normed_ci = [float(np.percentile(per_draw_normed, 2.5)), float(np.percentile(per_draw_normed, 97.5))]

print(f"\n=== E27 Results ===")
print(f"A (BoT-Mkt→Mkt):        {A_map:.2f}%")
print(f"C (BoT-MSMT→Mkt):       {C_map:.2f}%")
print(f"B_matched mean:         {B_mean:.2f}% [{B_ci[0]:.2f}, {B_ci[1]:.2f}]")
print(f"D_matched mean:         {D_mean:.2f}% [{D_ci[0]:.2f}, {D_ci[1]:.2f}]")
print(f"Raw asymmetry C/B:      {raw_asym:.3f}x (95% CI [{raw_ci[0]:.3f}, {raw_ci[1]:.3f}])")
print(f"Normed (C/A)/(B/D):     {normed_asym:.3f}x (95% CI [{normed_ci[0]:.3f}, {normed_ci[1]:.3f}])")

# Compare with original (unmatched) asymmetry
# Original B = 2.72% (from prior runs), original D = 82.4%
# Original raw: C/B = 26.66/2.72 = ~9.8x
# Original normed: (C/A)/(B/D) = (26.66/85.84)/(2.72/82.4) = 0.311/0.033 = ~9.4x
print(f"\nComparison:")
print(f"  Original B (full MSMT17):  2.72%  D: 82.40%")
print(f"  Matched  B (MSMT17→Mkt shape): {B_mean:.2f}%  D: {D_mean:.2f}%")
print(f"  Original normalised asymmetry: {(C_map/A_map)/(2.72/82.4):.2f}x")
print(f"  Matched  normalised asymmetry: {normed_asym:.2f}x")

results = {
    'market_shape': {'n_ids': MKT_N_IDS, 'n_gallery': MKT_N_GAL, 'n_query': MKT_N_QUE},
    'n_draws': N_DRAWS,
    'cell_A': {'map': A_map, 'r1': A_r1},
    'cell_C': {'map': C_map, 'r1': C_r1},
    'cell_B_matched': {'mean_map': B_mean, 'ci95_map': B_ci,
                       'mean_r1': float(B_r1s.mean()), 'per_draw': B_maps.tolist()},
    'cell_D_matched': {'mean_map': D_mean, 'ci95_map': D_ci,
                       'mean_r1': float(D_r1s.mean()), 'per_draw': D_maps.tolist()},
    'raw_asymmetry_C_over_B': {'mean': raw_asym, 'ci95': raw_ci},
    'normed_asymmetry': {'mean': normed_asym, 'ci95': normed_ci},
    'original_unmatched': {
        'B': 2.72, 'D': 82.40, 'C': C_map, 'A': A_map,
        'normed_asym': (C_map/A_map)/(2.72/82.40),
    },
}
out = OUT / 'e27_matched_asymmetry.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}")
