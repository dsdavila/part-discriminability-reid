"""
A2 (b) with matched identity-controlled null, both directions, r-sweep (rev 323).

MSMT direction (A2 original): r in {3, 5, 8, 14}
  id arm:  bot_msmt_seed42_msmt.npz (BoT-MSMT on MSMT test)
  ood arm: msmt17_test_noREA.npz    (BoT-Market on MSMT test)

Market direction (new per claude-chat): r in {2, 3, 5}
  id arm:  market_test_noREA.npz     (BoT-Market on Market test)
  ood arm: bot_msmt_seed42_market.npz (BoT-MSMT on Market test)

For each (direction, r):
  - (b) B_cam per arm (top-r SVD of centred identity-controlled nu_c)
  - id_top_r per arm (top-r SVD of identity centroids centred at global mean)
  - observed angle = median principal angles
  - 1000 identity-controlled structure-matched null draws per arm
    (random partition, per-group nu_{k,s} = mean_i (mu_{i,k,s} - mu_i))
  - For each draw, SVD basis of centred {nu_{k,s}} precomputed to max r,
    then truncated per r in the sweep
  - 50000 pair draws for shift null, excess + z + empirical p per r
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_NULL = 1000
N_PAIR = 50000

DIRECTIONS = {
    'MSMT': {
        'id_file':  'bot_msmt_seed42_msmt.npz',  'id_n_q':  11659,
        'ood_file': 'msmt17_test_noREA.npz',     'ood_n_q': 11659,
        'r_sweep':  [3, 5, 8, 14],
        'id_cams':  15,  'ood_cams': 15,
        'max_r':    14,
    },
    'Market': {
        'id_file':  'market_test_noREA.npz',     'id_n_q':  3368,
        'ood_file': 'bot_msmt_seed42_market.npz','ood_n_q': 3368,
        'r_sweep':  [2, 3, 5],
        'id_cams':  6,  'ood_cams': 6,
        'max_r':    5,
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
    return Vt[:K].T, s


def principal_angles(B1, B2):
    M = B1.T @ B2
    s = np.linalg.svd(M, compute_uv=False)
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
        if mask.sum() < 2:
            out[int(p)] = None
        else:
            out[int(p)] = X[mask].mean(0)
    return out


def partition_b_full_basis(X, pids, group_id, n_groups, id_mean_dict):
    """Return full SVD basis (d, n_groups) of centred identity-controlled
    nu_{k,s}, so it can be truncated to any r <= n_groups - 1."""
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
    return Vt.T, s  # (d, n_groups)


def run_direction(direction, cfg):
    print(f"\n{'='*60}\n=== A2 (b) r-sweep: {direction} direction\n{'='*60}", flush=True)
    arm = {}
    for tag, (fname, n_q, n_cams_key) in [
        ('id', (cfg['id_file'], cfg['id_n_q'], 'id_cams')),
        ('ood', (cfg['ood_file'], cfg['ood_n_q'], 'ood_cams')),
    ]:
        d = np.load(CACHE_DIR / fname)
        emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
        X = l2(emb[n_q:]); pid = pids[n_q:]; cid = cids[n_q:]
        n_cams = len(np.unique(cid))
        assert n_cams == cfg[n_cams_key], f"{direction}/{tag}: expected {cfg[n_cams_key]} cams, got {n_cams}"
        # Full identity SVD
        id_cents = identity_centroids_centered(X, pid)
        id_full, id_svals = top_K_basis(id_cents, cfg['max_r'])
        # Full (b) B_cam SVD
        nu_c = camera_offsets_b(X, pid, cid)
        B_full, B_svals = top_K_basis(nu_c, min(cfg['max_r'], n_cams - 1))
        # id mean dict for null draws
        id_mean = identity_mean_dict(X, pid)
        arm[tag] = {
            'X': X, 'pids': pid, 'cids': cid,
            'n_cams': n_cams,
            'id_full': id_full, 'B_full': B_full,
            'id_mean': id_mean,
            'cam_sizes': [int((cid == c).sum()) for c in np.unique(cid)],
        }
        print(f"  {tag}: n_items={len(X)}  n_cams={n_cams}  B_full svals={np.round(B_svals, 4)}", flush=True)

    # Observed angles per r
    obs_by_r = {}
    for r in cfg['r_sweep']:
        obs_id  = float(np.median(principal_angles(arm['id']['B_full'][:, :min(r, arm['id']['n_cams']-1)],
                                                    arm['id']['id_full'][:, :r])))
        obs_ood = float(np.median(principal_angles(arm['ood']['B_full'][:, :min(r, arm['ood']['n_cams']-1)],
                                                    arm['ood']['id_full'][:, :r])))
        obs_by_r[r] = {'id': obs_id, 'ood': obs_ood, 'shift': obs_ood - obs_id}
        print(f"  r={r:2d}: obs id={obs_id:.3f}  ood={obs_ood:.3f}  shift={obs_by_r[r]['shift']:+.3f}", flush=True)

    # Null draws per arm, SVD stored up to max r
    print(f"\n  Drawing (b)-matched null, {N_NULL} per arm...", flush=True)
    rngs = {'id': np.random.default_rng(42), 'ood': np.random.default_rng(1337)}
    null_bases = {}  # arm -> list of basis (d, n_cams)
    for tag in ['id', 'ood']:
        print(f"    -- {tag} arm (n_cams={arm[tag]['n_cams']}) --", flush=True)
        rng = rngs[tag]
        X = arm[tag]['X']; pid = arm[tag]['pids']; cid = arm[tag]['cids']
        sizes = arm[tag]['cam_sizes']; n_groups = arm[tag]['n_cams']
        id_mean = arm[tag]['id_mean']
        n = len(X)
        bases = []
        for di in range(N_NULL):
            perm = rng.permutation(n)
            gid = np.empty(n, dtype=np.int64); idx = 0
            for k, nk in enumerate(sizes):
                gid[perm[idx:idx+nk]] = k; idx += nk
            basis, _ = partition_b_full_basis(X, pid, gid, n_groups, id_mean)
            bases.append(basis)
            if di < 3 or di == N_NULL - 1 or (di + 1) % 250 == 0:
                print(f"      draw {di+1}/{N_NULL}", flush=True)
        null_bases[tag] = bases

    # Angle per draw per r
    null_angles = {r: {'id': [], 'ood': []} for r in cfg['r_sweep']}
    for tag in ['id', 'ood']:
        for basis in null_bases[tag]:
            for r in cfg['r_sweep']:
                r_eff = min(r, arm[tag]['n_cams'] - 1)
                ang = float(np.median(principal_angles(basis[:, :r_eff], arm[tag]['id_full'][:, :r])))
                null_angles[r][tag].append(ang)

    # Shift distribution per r
    rng_pair = np.random.default_rng(2026)
    out = {'direction': direction, 'per_r': {}}
    for r in cfg['r_sweep']:
        id_arr  = np.array(null_angles[r]['id'])
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
            'null_id_mean': float(id_arr.mean()),  'null_id_sd':  float(id_arr.std(ddof=1)),
            'null_id_min': float(id_arr.min()), 'null_id_max': float(id_arr.max()),
            'null_ood_mean': float(ood_arr.mean()),'null_ood_sd': float(ood_arr.std(ddof=1)),
            'null_ood_min': float(ood_arr.min()), 'null_ood_max': float(ood_arr.max()),
            'null_shift_mean': float(shift_null.mean()), 'null_shift_sd': float(shift_null.std(ddof=1)),
            'null_shift_min': float(shift_null.min()), 'null_shift_max': float(shift_null.max()),
            'obs_inside_support': bool(shift_null.min() <= obs_shift <= shift_null.max()),
            'obs_below_min_by_deg': float(shift_null.min() - obs_shift) if obs_shift < shift_null.min() else 0.0,
            'excess': float(excess), 'z': float(z), 'empirical_p': pct, 'n_pair_draws': N_PAIR,
        }
        in_sup = "in-support" if (shift_null.min() <= obs_shift <= shift_null.max()) else "OUTSIDE support"
        print(f"  r={r:2d}: null shift [{shift_null.min():+.3f}, {shift_null.max():+.3f}]  mean={shift_null.mean():+.3f}  sd={shift_null.std(ddof=1):.3f}  obs={obs_shift:+.3f} ({in_sup})  excess={excess:+.3f}  z={z:+.3f}  p={pct:.5f}", flush=True)
    return out


def main():
    out = {}
    for direction, cfg in DIRECTIONS.items():
        out[direction] = run_direction(direction, cfg)
    out_path = RESULTS / 'e43_A2_b_rsweep_minmax.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n=========== A2 (b)-matched r-sweep SUMMARY ===========")
    for direction, result in out.items():
        print(f"\n  Direction: {direction}")
        print(f"    {'r':>4s}  {'obs shift':>10s}  {'null shift':>11s}  {'excess':>8s}  {'z':>6s}  {'p':>10s}")
        for r, d in result['per_r'].items():
            print(f"    {r:4d}  {d['obs_shift']:+10.3f}  {d['null_shift_mean']:+11.3f}  {d['excess']:+8.3f}  {d['z']:+6.3f}  {d['empirical_p']:10.5f}")


if __name__ == '__main__':
    main()
