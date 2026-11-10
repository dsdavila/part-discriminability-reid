"""
A2 (b) with identity-controlled structure-matched null (rev 321).

Per claude-chat's rev 321: the (b) rebuild observed statistic uses
identity-controlled offsets, but the previous null still drew raw
partition group means. Rebuild the null to match: for each surrogate
partition s and each group k in it, compute
  nu_k_s = mean_i(mu_{i,k,s} - mu_i) over identities present in (k,s)
then centre the {nu_k_s} across k and take top-r SVD -> B_fake_s.

MSMT only, r = 5, 1000 draws per arm, 50000 independent pair draws
for the shift distribution.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_NULL = 1000
N_PAIR = 50000
R = 5

CELLS = {
    'id':  ('bot_msmt_seed42_msmt.npz',  11659),
    'ood': ('msmt17_test_noREA.npz',     11659),
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def identity_centroids_centered(X, pids):
    mu_g = X.mean(0)
    unique_p = np.unique(pids)
    return np.stack([X[pids == p].mean(0) - mu_g for p in unique_p])


def top_r_basis(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:r].T


def principal_angles(B1, B2):
    M = B1.T @ B2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def camera_offsets_b(X, pids, cids):
    """Observed (b) B_cam: identity-controlled."""
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
    nu_c = nu_c - nu_c.mean(0)
    return nu_c


def identity_mean_table(X, pids):
    """Return dict pid -> mu_i (item-weighted identity centroid) for pids
    with at least 2 items; None otherwise."""
    unique_pids = np.unique(pids)
    out = {}
    for p in unique_pids:
        mask = (pids == p)
        if mask.sum() < 2:
            out[p] = None
        else:
            out[p] = X[mask].mean(0)
    return out


def partition_identity_controlled_span(X, pids, group_id, n_groups, id_mean_dict, r):
    """Compute (b)-style span from a surrogate partition.
    group_id: (n,) integer array in [0, n_groups).
    Returns top-r SVD basis of centred {nu_k}."""
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
                continue  # identity with <2 total items globally: skip
            deltas.append(mu_ik[j] - mu_i)
        if deltas:
            nu.append(np.mean(np.stack(deltas), axis=0))
        else:
            nu.append(np.zeros(X.shape[1], dtype=X.dtype))
    nu = np.stack(nu)
    nu = nu - nu.mean(0)
    return top_r_basis(nu, r)


def struct_matched_b_draws(X, pids, cids, id_top_r_basis, n_draws, rng):
    """Draw n_draws identity-controlled structure-matched surrogates and
    return list of median principal angles vs id_top_r_basis."""
    real_sizes = [int((cids == c).sum()) for c in np.unique(cids)]
    n_groups = len(real_sizes)
    id_mean_dict = identity_mean_table(X, pids)
    n = len(X)
    angs = []
    for d in range(n_draws):
        perm = rng.permutation(n)
        group_id = np.empty(n, dtype=np.int64)
        idx = 0
        for k, n_imgs in enumerate(real_sizes):
            group_id[perm[idx:idx+n_imgs]] = k
            idx += n_imgs
        B_fake = partition_identity_controlled_span(X, pids, group_id, n_groups, id_mean_dict, R)
        angs.append(float(np.median(principal_angles(B_fake, id_top_r_basis))))
        if d < 3 or d == n_draws - 1 or (d+1) % 200 == 0:
            print(f"    draw {d+1}/{n_draws}: ang={angs[-1]:.3f}", flush=True)
    return np.array(angs)


def main():
    arm_data = {}
    for arm, (fname, n_q) in CELLS.items():
        print(f"\n=== Loading arm={arm} ===", flush=True)
        d = np.load(CACHE_DIR / fname)
        emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
        X = l2(emb[n_q:]); pid = pids[n_q:]; cid = cids[n_q:]
        id_top_r = top_r_basis(identity_centroids_centered(X, pid), R)
        nu_c = camera_offsets_b(X, pid, cid)
        B_b = top_r_basis(nu_c, R)
        obs = float(np.median(principal_angles(B_b, id_top_r)))
        arm_data[arm] = {
            'X': X, 'pids': pid, 'cids': cid, 'id_top_r': id_top_r, 'obs': obs
        }
        print(f"  arm={arm}: (b) B_cam vs id_top_r median angle = {obs:.3f}", flush=True)

    shift = arm_data['ood']['obs'] - arm_data['id']['obs']
    print(f"\nObserved (b) shift (ood - id) = {shift:+.3f}", flush=True)

    # (b)-matched null per arm
    print(f"\nDrawing identity-controlled structure-matched null (n={N_NULL} per arm)...", flush=True)
    rng_id = np.random.default_rng(42)
    rng_ood = np.random.default_rng(1337)
    print("\n-- id arm --", flush=True)
    null_id = struct_matched_b_draws(arm_data['id']['X'], arm_data['id']['pids'], arm_data['id']['cids'],
                                     arm_data['id']['id_top_r'], N_NULL, rng_id)
    print("\n-- ood arm --", flush=True)
    null_ood = struct_matched_b_draws(arm_data['ood']['X'], arm_data['ood']['pids'], arm_data['ood']['cids'],
                                      arm_data['ood']['id_top_r'], N_NULL, rng_ood)
    print(f"\n  id null: median={np.median(null_id):.3f}  mean={null_id.mean():.3f}  sd={null_id.std(ddof=1):.3f}  CI=[{np.quantile(null_id, 0.025):.3f}, {np.quantile(null_id, 0.975):.3f}]")
    print(f"  ood null: median={np.median(null_ood):.3f}  mean={null_ood.mean():.3f}  sd={null_ood.std(ddof=1):.3f}  CI=[{np.quantile(null_ood, 0.025):.3f}, {np.quantile(null_ood, 0.975):.3f}]")

    # Shift distribution over independent pair draws
    rng_pair = np.random.default_rng(2026)
    idx_i = rng_pair.integers(0, N_NULL, N_PAIR)
    idx_j = rng_pair.integers(0, N_NULL, N_PAIR)
    shift_null = null_ood[idx_j] - null_id[idx_i]
    pct_below = float((shift_null < shift).mean())
    excess = shift - shift_null.mean()
    z = excess / shift_null.std(ddof=1)
    print(f"\n  (b)-matched shift null: mean={shift_null.mean():+.3f}  sd={shift_null.std(ddof=1):.3f}  CI=[{np.quantile(shift_null, 0.025):+.3f}, {np.quantile(shift_null, 0.975):+.3f}]")
    print(f"  observed shift = {shift:+.3f}")
    print(f"  excess = {excess:+.3f}  z = {z:+.3f}  empirical p = {pct_below:.4f}  (N_PAIR={N_PAIR})")

    # Also the (a)-null comparison used in rev 320, for context
    # ood_a_null_mean used in rev 320 was -13.71, sd 5.86 - approximate from paired draws of the (a) null
    print(f"\nFor context (rev 320 used (a)-null): mean=-13.706 sd=5.862 -> excess=-17.322 z=-2.955 p=0.0008")

    out = {
        'observed': {
            'id_angle_b': arm_data['id']['obs'],
            'ood_angle_b': arm_data['ood']['obs'],
            'shift_b': shift,
        },
        'null_id_arm_b_matched': {
            'median': float(np.median(null_id)), 'mean': float(null_id.mean()), 'sd': float(null_id.std(ddof=1)),
            'ci': [float(np.quantile(null_id, 0.025)), float(np.quantile(null_id, 0.975))],
        },
        'null_ood_arm_b_matched': {
            'median': float(np.median(null_ood)), 'mean': float(null_ood.mean()), 'sd': float(null_ood.std(ddof=1)),
            'ci': [float(np.quantile(null_ood, 0.025)), float(np.quantile(null_ood, 0.975))],
        },
        'null_shift_b_matched': {
            'mean': float(shift_null.mean()), 'median': float(np.median(shift_null)), 'sd': float(shift_null.std(ddof=1)),
            'ci': [float(np.quantile(shift_null, 0.025)), float(np.quantile(shift_null, 0.975))],
            'excess_vs_obs': float(excess),
            'z_vs_obs': float(z),
            'empirical_p_one_tail': pct_below,
            'n_pair_draws': N_PAIR,
            'n_null_per_arm': N_NULL,
        },
    }
    out_path = RESULTS / 'e43_A2_b_matched_null.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
