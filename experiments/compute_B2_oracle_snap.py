"""
B2 oracle snap ceiling (rev 305). For each item (query + gallery) replace
its component along B_cam with the item's TRUE camera centroid's component
along B_cam (oracle labels); then L2-normalize and remeasure mAP.

Snap formula:
  x_snap = x - P_cam @ x + P_cam @ cam_c_mean   (c = item's true camera)
with P_cam = B_cam @ B_cam.T.

B_cam built from gallery, rank = n_cams - 1 (same as B1). Report delta
mAP vs the untouched baseline with identity-clustered bootstrap CI.
Pre-registered B2 floors against the canonical per-target gaps:
  MSMT as target (D-B'): gap 36.221 pp
  Market as target (A'-C): gap 53.737 pp
  <1 pp = dead; 1-5 pp = beats all; >5 pp = real.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
RNG_SEED = 42

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)",  'target': 'Market'},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)",   'target': 'MSMT'},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)",   'target': 'Market'},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)",     'target': 'MSMT'},
}

CANONICAL_GAP_PP = {'MSMT': 36.221, 'Market': 53.737}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def camera_offsets(X, cids):
    mu_g = X.mean(0)
    uc = np.unique(cids)
    return np.stack([X[cids == c].mean(0) - mu_g for c in uc]), uc


def top_r_basis(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:r].T


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, chunk=200):
    n_q = len(q_pids)
    aps = np.full(n_q, np.nan, dtype=np.float64)
    for i0 in range(0, n_q, chunk):
        i1 = min(i0 + chunk, n_q)
        sims = q_emb[i0:i1] @ g_emb.T
        for j, i in enumerate(range(i0, i1)):
            qid, qcam = q_pids[i], q_cids[i]
            junk = (g_pids == qid) & (g_cids == qcam)
            order = np.argsort(-sims[j])
            order = order[~junk[order]]
            good = g_pids[order] == qid
            if not good.any():
                continue
            pos = np.where(good)[0]
            aps[i] = float(np.mean((np.arange(1, good.sum() + 1) / (pos + 1))))
    return aps


def boot_delta(aps_base, aps_snap, q_pids, n_boot=N_BOOT, rng_seed=RNG_SEED):
    rng = np.random.default_rng(rng_seed)
    valid = ~(np.isnan(aps_base) | np.isnan(aps_snap))
    b = aps_base[valid]; s = aps_snap[valid]; pid = q_pids[valid]
    uids = np.unique(pid); idx_by_pid = {u: np.where(pid == u)[0] for u in uids}
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(uids, size=len(uids), replace=True)
        idxs = np.concatenate([idx_by_pid[u] for u in sampled])
        diffs.append((s[idxs].mean() - b[idxs].mean()) * 100)
    diffs = np.array(diffs)
    return {
        'base_pct': float(np.nanmean(b)) * 100,
        'snap_pct': float(np.nanmean(s)) * 100,
        'delta_pp_point': float(np.nanmean(s) - np.nanmean(b)) * 100,
        'delta_pp_median_boot': float(np.median(diffs)),
        'delta_pp_ci': [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        'n_valid_queries': int(valid.sum()),
    }


def oracle_snap(X, cids, cam_ids, cam_means, B_cam):
    """Replace each row's component along B_cam with its cam's mean's
    component along B_cam. Then L2-normalize.
    cam_ids: 1D array of unique cam ids; cam_means: (n_cams, d) matrix in same order.
    """
    # Map cid -> row in cam_means
    cid_to_row = {c: i for i, c in enumerate(cam_ids)}
    # For each item, its cam mean
    cam_rows = np.array([cid_to_row[c] for c in cids])
    cam_mean_per_item = cam_means[cam_rows]  # (n, d)
    # Projection operators
    #   P_cam x  = (x @ B) @ B.T
    #   x_snap = x - P_cam x + P_cam c_mean
    X_coords = X @ B_cam        # (n, r)
    cam_coords = cam_mean_per_item @ B_cam  # (n, r)
    X_snap = X - X_coords @ B_cam.T + cam_coords @ B_cam.T
    return l2(X_snap)


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    n_cams = len(np.unique(g_cids))
    r = n_cams - 1

    # Baseline
    print(f"  baseline mAP...", flush=True)
    aps_base = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    base = float(np.nanmean(aps_base)) * 100
    print(f"  baseline = {base:.4f}%", flush=True)

    # B_cam from gallery (not centered subtraction — take raw cam means and the span of their centered offsets)
    uc = np.unique(g_cids)
    cam_means = np.stack([g_emb[g_cids == c].mean(0) for c in uc])
    offsets = cam_means - g_emb.mean(0)
    B_cam = top_r_basis(offsets, r)

    # Oracle snap on query and gallery (using each one's TRUE camera label)
    print(f"  oracle-snapping each item's B_cam component to its true cam lane...", flush=True)
    q_snap = oracle_snap(q_emb, q_cids, uc, cam_means, B_cam)
    g_snap = oracle_snap(g_emb, g_cids, uc, cam_means, B_cam)
    aps_snap = per_query_ap(q_snap, q_pids, q_cids, g_snap, g_pids, g_cids)
    snap = float(np.nanmean(aps_snap)) * 100
    print(f"  snapped mAP = {snap:.4f}%  delta = {snap - base:+.4f} pp", flush=True)

    boot = boot_delta(aps_base, aps_snap, q_pids)
    print(f"  delta = {boot['delta_pp_point']:+.4f} pp  CI [{boot['delta_pp_ci'][0]:+.4f}, {boot['delta_pp_ci'][1]:+.4f}]", flush=True)

    tgt = info['target']
    gap = CANONICAL_GAP_PP[tgt]
    frac = boot['delta_pp_point'] / gap * 100
    print(f"  as pct of {tgt}-target canonical gap {gap:.3f} pp: {frac:+.2f}%", flush=True)

    return {
        'cell': tag,
        'label': info['label'],
        'target': tgt,
        'n_cams': n_cams,
        'r_projected': r,
        **boot,
        'canonical_gap_pp': gap,
        'delta_pct_of_canonical_gap': frac,
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)
    out_path = RESULTS / 'e43_B2_oracle_snap.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary vs floors
    print("\n\n=== B2 oracle snap ceiling summary ===")
    for tag in CELLS:
        r = out[tag]
        d_pp = r['delta_pp_point']
        floor_verdict = "dead (<1 pp)" if abs(d_pp) < 1 else ("beats all (1-5 pp)" if abs(d_pp) < 5 else "real (>5 pp)")
        print(f"  {tag} ({r['label']}): base={r['base_pct']:.3f}%  snap={r['snap_pct']:.3f}%  delta={d_pp:+.3f} pp  CI [{r['delta_pp_ci'][0]:+.3f}, {r['delta_pp_ci'][1]:+.3f}]  ({floor_verdict}; {r['delta_pct_of_canonical_gap']:+.2f}% of gap)")


if __name__ == '__main__':
    main()
