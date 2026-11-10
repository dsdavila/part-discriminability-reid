"""
A2-train (rev 335): re-run A2 with id_top_r built from source-domain
training split and held FIXED across both arms, so the shift measures
B_cam moving relative to a stationary identity reference.

For each model:
  id_top_r_TRAIN = top-r SVD of identity centroids (centered) built from
    /tmp/e18_train_cache/bot_<model>_<model>_train.npz. Held FIXED.
  B_cam_id = top-r SVD of (b) nu_c on source TEST gallery (id arm).
  B_cam_ood = top-r SVD of (b) nu_c on target TEST gallery (ood arm).
  theta_id = median principal angles(B_cam_id, id_top_r_TRAIN)
  theta_ood = median principal angles(B_cam_ood, id_top_r_TRAIN)
  shift = theta_ood - theta_id

Null: (b) identity-controlled structure-matched null on each test
gallery, angle measured vs id_top_r_TRAIN (NOT resampled). Shift null
is Cartesian pair of id and ood null arms. N=1000 per arm, N_PAIR=50000.

Rank sweep: r in {3, 5, 8, 14} for MSMT model, {2, 3, 5} for Market.

Pre-reg (claude-chat): Market->MSMT excess survives at >= half current
magnitude (<= -9.2 degrees) on r=5. Falsified if vanishes or reverses.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
TRAIN_CACHE = Path('/tmp/e18_train_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_NULL = 1000
N_PAIR = 50000

MODELS = {
    'Market_model': {
        'train_file': 'bot_market_market_train.npz',
        'id_file':    'market_test_noREA.npz',  'id_n_q':  3368,
        'ood_file':   'msmt17_test_noREA.npz',  'ood_n_q': 11659,
        'r_sweep':    [2, 3, 5],
        'label':      'Market model: A′(id) → B′(ood)',
    },
    'MSMT_model': {
        'train_file': 'bot_msmt_msmt_train.npz',
        'id_file':    'bot_msmt_seed42_msmt.npz',   'id_n_q':  11659,
        'ood_file':   'bot_msmt_seed42_market.npz', 'ood_n_q': 3368,
        'r_sweep':    [3, 5, 8, 14],
        'label':      'MSMT model: D(id) → C(ood)',
    },
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def identity_centroids_centered(X, pids):
    mu_g = X.mean(0)
    unique_p = np.unique(pids)
    return np.stack([X[pids == p].mean(0) - mu_g for p in unique_p])


def top_K_basis(mat, K):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:K].T


def principal_angles(B1, B2):
    s = np.linalg.svd(B1.T @ B2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


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
    return nu_c - nu_c.mean(0)


def identity_mean_dict(X, pids):
    unique_pids = np.unique(pids)
    out = {}
    for p in unique_pids:
        mask = (pids == p)
        out[int(p)] = X[mask].mean(0) if mask.sum() >= 2 else None
    return out


def partition_b_full_basis(X, pids, group_id, n_groups, id_mean_dict):
    nu = []
    for k in range(n_groups):
        mask_k = (group_id == k)
        if mask_k.sum() == 0:
            nu.append(np.zeros(X.shape[1], dtype=X.dtype))
            continue
        pids_k = pids[mask_k]
        X_k = X[mask_k]
        unique_p_in_k, inv_idx = np.unique(pids_k, return_inverse=True)
        d = X.shape[1]
        sums = np.zeros((len(unique_p_in_k), d), dtype=X.dtype)
        counts = np.zeros(len(unique_p_in_k), dtype=np.int64)
        np.add.at(sums, inv_idx, X_k)
        np.add.at(counts, inv_idx, 1)
        mu_ik = sums / counts[:, None]
        deltas = []
        for j, p in enumerate(unique_p_in_k):
            mu_i = id_mean_dict.get(int(p))
            if mu_i is None:
                continue
            deltas.append(mu_ik[j] - mu_i)
        if deltas:
            nu.append(np.mean(np.stack(deltas), axis=0))
        else:
            nu.append(np.zeros(X.shape[1], dtype=X.dtype))
    nu = np.stack(nu)
    nu = nu - nu.mean(0)
    _, s, Vt = np.linalg.svd(nu, full_matrices=False)
    return Vt.T, s


def load_arm(info, key):
    d = np.load(CACHE_DIR / info[f'{key}_file'])
    emb = d['emb'].astype(np.float32); n_q = info[f'{key}_n_q']
    pids = d['pids']; cids = d['cids']
    return l2(emb[n_q:]), pids[n_q:], cids[n_q:]


def process_model(model, info):
    print(f"\n{'='*60}\n=== A2-train: {info['label']}\n{'='*60}", flush=True)
    # Training id_top_r (held fixed)
    print(f"  loading training cache {info['train_file']}...", flush=True)
    dt = np.load(TRAIN_CACHE / info['train_file'])
    t_emb = l2(dt['emb'].astype(np.float32)); t_pids = dt['pids']
    max_r = max(info['r_sweep'])
    id_full_train = top_K_basis(identity_centroids_centered(t_emb, t_pids), max_r)
    print(f"  id_top_r_TRAIN computed from {len(t_emb)} train items, {len(np.unique(t_pids))} identities, max_r={max_r}", flush=True)
    # Load both arms
    arm = {}
    for key in ['id', 'ood']:
        X, pids, cids = load_arm(info, key)
        arm[key] = {'X': X, 'pids': pids, 'cids': cids}
        n_cams = len(np.unique(cids))
        nu_c = camera_offsets_b(X, pids, cids)
        _, s_all, Vt = np.linalg.svd(nu_c, full_matrices=False)
        B_full = Vt[:min(max_r, n_cams - 1)].T  # (d, r) up to n_cams-1
        arm[key]['B_full'] = B_full
        arm[key]['n_cams'] = n_cams
        arm[key]['id_mean'] = identity_mean_dict(X, pids)
        arm[key]['cam_sizes'] = [int((cids == c).sum()) for c in np.unique(cids)]
        print(f"  {key} arm: n_cams={n_cams}  (b) B_cam svals={np.round(s_all[:5], 5)}", flush=True)

    # Observed angles per r
    obs_by_r = {}
    for r in info['r_sweep']:
        obs = {}
        for key in ['id', 'ood']:
            r_eff = min(r, arm[key]['n_cams'] - 1)
            ang = float(np.median(principal_angles(arm[key]['B_full'][:, :r_eff], id_full_train[:, :r])))
            obs[key] = ang
        shift = obs['ood'] - obs['id']
        obs_by_r[r] = {**obs, 'shift': shift}
        print(f"  r={r:2d}: obs id={obs['id']:.3f}  ood={obs['ood']:.3f}  shift={shift:+.3f}", flush=True)

    # Null draws per arm (keep id_top_r_TRAIN fixed)
    print(f"\n  Drawing (b)-matched null, {N_NULL} per arm (reference FIXED to train id_top_r)...", flush=True)
    rngs = {'id': np.random.default_rng(42), 'ood': np.random.default_rng(1337)}
    null_bases = {}
    for key in ['id', 'ood']:
        print(f"    -- {key} arm (n_cams={arm[key]['n_cams']}) --", flush=True)
        rng = rngs[key]
        X = arm[key]['X']; pid = arm[key]['pids']
        sizes = arm[key]['cam_sizes']; n_groups = arm[key]['n_cams']
        id_mean = arm[key]['id_mean']
        bases = []
        for di in range(N_NULL):
            perm = rng.permutation(len(X))
            gid = np.empty(len(X), dtype=np.int64); idx = 0
            for k, nk in enumerate(sizes):
                gid[perm[idx:idx+nk]] = k; idx += nk
            basis, _ = partition_b_full_basis(X, pid, gid, n_groups, id_mean)
            bases.append(basis)
            if di < 3 or di == N_NULL - 1 or (di + 1) % 250 == 0:
                print(f"      draw {di+1}/{N_NULL}", flush=True)
        null_bases[key] = bases

    # Angles per draw per r (vs fixed id_top_r_TRAIN)
    null_angles = {r: {'id': [], 'ood': []} for r in info['r_sweep']}
    for key in ['id', 'ood']:
        for basis in null_bases[key]:
            for r in info['r_sweep']:
                r_eff = min(r, arm[key]['n_cams'] - 1)
                ang = float(np.median(principal_angles(basis[:, :r_eff], id_full_train[:, :r])))
                null_angles[r][key].append(ang)

    # Shift null per r
    rng_pair = np.random.default_rng(2026)
    out = {'model': model, 'label': info['label'], 'per_r': {}}
    for r in info['r_sweep']:
        id_arr = np.array(null_angles[r]['id'])
        ood_arr = np.array(null_angles[r]['ood'])
        idx_i = rng_pair.integers(0, N_NULL, N_PAIR)
        idx_j = rng_pair.integers(0, N_NULL, N_PAIR)
        shift_null = ood_arr[idx_j] - id_arr[idx_i]
        obs_shift = obs_by_r[r]['shift']
        pct = float((shift_null < obs_shift).mean())
        excess = obs_shift - shift_null.mean()
        z = excess / shift_null.std(ddof=1)
        out['per_r'][r] = {
            'obs_id': obs_by_r[r]['id'], 'obs_ood': obs_by_r[r]['ood'], 'obs_shift': obs_shift,
            'null_id_mean': float(id_arr.mean()), 'null_id_sd': float(id_arr.std(ddof=1)),
            'null_ood_mean': float(ood_arr.mean()), 'null_ood_sd': float(ood_arr.std(ddof=1)),
            'null_shift_mean': float(shift_null.mean()), 'null_shift_sd': float(shift_null.std(ddof=1)),
            'null_shift_min': float(shift_null.min()), 'null_shift_max': float(shift_null.max()),
            'excess': float(excess), 'z': float(z),
            'empirical_p': pct, 'in_support': bool(shift_null.min() <= obs_shift <= shift_null.max()),
            'n_pair_draws': N_PAIR,
        }
        in_sup = "in-support" if (shift_null.min() <= obs_shift <= shift_null.max()) else "OUTSIDE support"
        print(f"  r={r:2d}: null shift mean={shift_null.mean():+.3f}  sd={shift_null.std(ddof=1):.3f}  range=[{shift_null.min():+.3f}, {shift_null.max():+.3f}]  obs={obs_shift:+.3f} ({in_sup})  excess={excess:+.3f}  z={z:+.3f}  p={pct:.5f}", flush=True)
    return out


def main():
    out = {}
    for model, info in MODELS.items():
        out[model] = process_model(model, info)
    out_path = RESULTS / 'e43_A2_train.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========== A2-train (fixed training id_top_r) ==========")
    for model in MODELS:
        print(f"\n  {out[model]['label']}:")
        print(f"    {'r':>4s}  {'obs id':>8s}  {'obs ood':>8s}  {'obs shift':>10s}  {'null shift':>11s}  {'excess':>8s}  {'z':>7s}  {'p':>10s}")
        for r, d in out[model]['per_r'].items():
            print(f"    {r:4d}  {d['obs_id']:8.3f}  {d['obs_ood']:8.3f}  {d['obs_shift']:+10.3f}  {d['null_shift_mean']:+11.3f}  {d['excess']:+8.3f}  {d['z']:+7.3f}  {d['empirical_p']:10.5f}")


if __name__ == '__main__':
    main()
