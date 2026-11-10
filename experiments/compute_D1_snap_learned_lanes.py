"""
D1 - Snap to LEARNED (training-side) camera lanes (rev 332 pre-reg).

Build training-side (b) identity-controlled B_cam from the MODEL's own
training set. Then for each test item (query + gallery) on the ood arm,
replace its component along the training-side B_cam with the nearest
training lane's (b) offset projection. L2-normalize and remeasure mAP.

Random-lane control: snap to a UNIFORMLY RANDOM training lane instead
of nearest. Required, not optional.

For each of the four cells:
  Training source:
    - Market model cells (A', B'): /tmp/e18_train_cache/bot_market_market_train.npz (6 cams)
    - MSMT model cells (C, D):     /tmp/e18_train_cache/bot_msmt_msmt_train.npz  (15 cams)
  Build training B_cam (b) at rank = n_cams - 1.
  Test-side arm: same cell as before (query + gallery from e12_feature_cache).

Report delta mAP (snap - baseline) with identity-clustered bootstrap
n=500 on the test arm. Against canonical per-target gaps:
  MSMT as target: 36.221 pp
  Market as target: 53.737 pp
Bands: <1 pp dead, 1-5 pp beats-all, >5 pp real.

claude-chat pre-reg: <1 pp.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
TRAIN_CACHE = Path('/tmp/e18_train_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
RNG_SEED = 42

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)",  'train_file': 'bot_market_market_train.npz', 'target': 'Market'},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)",   'train_file': 'bot_market_market_train.npz', 'target': 'MSMT'},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)",   'train_file': 'bot_msmt_msmt_train.npz',   'target': 'Market'},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)",     'train_file': 'bot_msmt_msmt_train.npz',   'target': 'MSMT'},
}

CANONICAL_GAP_PP = {'MSMT': 36.221, 'Market': 53.737}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def camera_offsets_b(X, pids, cids):
    unique_pids = np.unique(pids)
    uc = np.unique(cids)
    per_cam = {c: [] for c in uc}
    for p in unique_pids:
        mask_p = (pids == p)
        items_p = X[mask_p]
        if len(items_p) < 2:
            continue
        mu_i = items_p.mean(0)
        cams_p = cids[mask_p]
        for c in np.unique(cams_p):
            mu_ic = items_p[cams_p == c].mean(0)
            per_cam[c].append(mu_ic - mu_i)
    nu_c = []
    for c in uc:
        if per_cam[c]:
            nu_c.append(np.mean(np.stack(per_cam[c]), axis=0))
        else:
            nu_c.append(np.zeros(X.shape[1], dtype=X.dtype))
    nu_c = np.stack(nu_c)
    return nu_c - nu_c.mean(0), uc


def top_r_basis(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:r].T, s


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


def boot_delta(aps_base, aps_mod, q_pids, n_boot=N_BOOT, rng_seed=RNG_SEED):
    rng = np.random.default_rng(rng_seed)
    valid = ~(np.isnan(aps_base) | np.isnan(aps_mod))
    b = aps_base[valid]; m = aps_mod[valid]; pid = q_pids[valid]
    uids = np.unique(pid); idx_by_pid = {u: np.where(pid == u)[0] for u in uids}
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(uids, size=len(uids), replace=True)
        idxs = np.concatenate([idx_by_pid[u] for u in sampled])
        diffs.append((m[idxs].mean() - b[idxs].mean()) * 100)
    diffs = np.array(diffs)
    return {
        'base_pct': float(np.nanmean(b)) * 100,
        'mod_pct': float(np.nanmean(m)) * 100,
        'delta_pp_point': float(np.nanmean(m) - np.nanmean(b)) * 100,
        'delta_pp_ci': [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
    }


def snap_to_nearest_lane(X, B_cam_train, nu_c_train):
    """For each row x, snap its B_cam_train component to nearest training lane.
    Returns L2-normalized result.
    """
    x_coords = X @ B_cam_train              # (n, r)
    lane_coords = nu_c_train @ B_cam_train   # (n_cams_train, r)
    # Nearest lane by Euclidean in B_cam_train coordinates
    d2 = ((x_coords[:, None, :] - lane_coords[None, :, :]) ** 2).sum(axis=-1)  # (n, n_cams_train)
    nearest = d2.argmin(axis=1)
    target_coords = lane_coords[nearest]
    X_snap = X - x_coords @ B_cam_train.T + target_coords @ B_cam_train.T
    return l2(X_snap), nearest


def snap_to_random_lane(X, B_cam_train, nu_c_train, rng):
    """Snap each row to a uniformly random training lane."""
    lane_coords = nu_c_train @ B_cam_train
    n_cams_train = lane_coords.shape[0]
    random_assign = rng.integers(0, n_cams_train, size=len(X))
    target_coords = lane_coords[random_assign]
    x_coords = X @ B_cam_train
    X_snap = X - x_coords @ B_cam_train.T + target_coords @ B_cam_train.T
    return l2(X_snap), random_assign


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    # Load test arm
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    # Load training arm for training-side B_cam
    dt = np.load(TRAIN_CACHE / info['train_file'])
    t_emb = l2(dt['emb'].astype(np.float32)); t_pids = dt['pids']; t_cids = dt['cids']
    n_cams_train = len(np.unique(t_cids))
    r_train = n_cams_train - 1
    print(f"  training source: {info['train_file']}  n_items={len(t_emb)}  n_cams_train={n_cams_train}  r_train={r_train}", flush=True)
    # Build (b) B_cam from training
    nu_c_train, uc_train = camera_offsets_b(t_emb, t_pids, t_cids)
    B_cam_train, s_train = top_r_basis(nu_c_train, r_train)
    print(f"  training B_cam (b) sing vals: {np.round(s_train, 4)}", flush=True)

    # Baseline mAP on test arm
    print(f"  baseline mAP...", flush=True)
    aps_base = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    base = float(np.nanmean(aps_base)) * 100
    print(f"  baseline = {base:.4f}%", flush=True)

    # Snap to nearest training lane
    print(f"  snap to NEAREST training lane...", flush=True)
    q_snap_n, q_assign_n = snap_to_nearest_lane(q_emb, B_cam_train, nu_c_train)
    g_snap_n, g_assign_n = snap_to_nearest_lane(g_emb, B_cam_train, nu_c_train)
    # Report distribution of chosen lanes
    _, counts_n = np.unique(np.concatenate([q_assign_n, g_assign_n]), return_counts=True)
    print(f"  nearest-lane assignment counts across all items: {counts_n.tolist()}  (uniform would be {(len(q_assign_n) + len(g_assign_n)) // n_cams_train})", flush=True)
    aps_near = per_query_ap(q_snap_n, q_pids, q_cids, g_snap_n, g_pids, g_cids)
    near_res = boot_delta(aps_base, aps_near, q_pids)
    print(f"  NEAREST snap: snap={near_res['mod_pct']:.4f}%  delta={near_res['delta_pp_point']:+.4f} pp  CI [{near_res['delta_pp_ci'][0]:+.4f}, {near_res['delta_pp_ci'][1]:+.4f}]", flush=True)

    # Snap to RANDOM training lane (control)
    print(f"  snap to RANDOM training lane (control)...", flush=True)
    rng_ctl = np.random.default_rng(RNG_SEED + 1)
    q_snap_r, _ = snap_to_random_lane(q_emb, B_cam_train, nu_c_train, rng_ctl)
    g_snap_r, _ = snap_to_random_lane(g_emb, B_cam_train, nu_c_train, rng_ctl)
    aps_rand = per_query_ap(q_snap_r, q_pids, q_cids, g_snap_r, g_pids, g_cids)
    rand_res = boot_delta(aps_base, aps_rand, q_pids)
    print(f"  RANDOM snap: snap={rand_res['mod_pct']:.4f}%  delta={rand_res['delta_pp_point']:+.4f} pp  CI [{rand_res['delta_pp_ci'][0]:+.4f}, {rand_res['delta_pp_ci'][1]:+.4f}]", flush=True)

    tgt = info['target']; gap = CANONICAL_GAP_PP[tgt]
    near_frac = near_res['delta_pp_point'] / gap * 100
    rand_frac = rand_res['delta_pp_point'] / gap * 100
    excess_pp = near_res['delta_pp_point'] - rand_res['delta_pp_point']
    excess_frac = excess_pp / gap * 100
    print(f"  Fraction of {tgt}-target canonical gap ({gap} pp): nearest={near_frac:+.2f}%  random={rand_frac:+.2f}%  excess(near-rand)={excess_frac:+.2f}%", flush=True)

    def band(d_pp):
        if abs(d_pp) < 1: return "dead (<1pp)"
        if abs(d_pp) < 5: return "beats-all (1-5pp)"
        return "real (>5pp)"

    return {
        'cell': tag,
        'label': info['label'],
        'target': tgt,
        'train_file': info['train_file'],
        'n_cams_train': n_cams_train,
        'r_train': r_train,
        'canonical_gap_pp': gap,
        'nearest_snap': {**near_res, 'band': band(near_res['delta_pp_point']), 'frac_of_gap_pct': near_frac},
        'random_snap_control': {**rand_res, 'band': band(rand_res['delta_pp_point']), 'frac_of_gap_pct': rand_frac},
        'excess_near_over_random_pp': excess_pp,
        'excess_frac_of_gap_pct': excess_frac,
        'nearest_assignment_counts': counts_n.tolist(),
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)
    out_path = RESULTS / 'e43_D1_snap_learned_lanes.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= D1 SNAP-TO-LEARNED-LANES SUMMARY =========")
    for tag in CELLS:
        r = out[tag]
        print(f"\n  {tag} ({r['label']}) — {r['target']}-target gap {r['canonical_gap_pp']:.3f} pp:")
        print(f"    NEAREST:  delta={r['nearest_snap']['delta_pp_point']:+.4f} pp CI [{r['nearest_snap']['delta_pp_ci'][0]:+.3f}, {r['nearest_snap']['delta_pp_ci'][1]:+.3f}]  ({r['nearest_snap']['band']}, {r['nearest_snap']['frac_of_gap_pct']:+.2f}% of gap)")
        print(f"    RANDOM:   delta={r['random_snap_control']['delta_pp_point']:+.4f} pp CI [{r['random_snap_control']['delta_pp_ci'][0]:+.3f}, {r['random_snap_control']['delta_pp_ci'][1]:+.3f}]  ({r['random_snap_control']['band']}, {r['random_snap_control']['frac_of_gap_pct']:+.2f}% of gap)")
        print(f"    EXCESS (near-rand): {r['excess_near_over_random_pp']:+.4f} pp ({r['excess_frac_of_gap_pct']:+.2f}% of gap)")


if __name__ == '__main__':
    main()
