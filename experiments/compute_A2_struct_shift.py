"""
Rev-301 follow-up: compute A2 shift p-value from structure-matched draws,
not from the covariance-matched nulls that have been superseded.

For each of N draws on each arm independently:
  - sample a size-matched random 15-partition of items
  - form fake camera offsets, take top-r basis B_fake
  - measure median principal angle vs that arm's id_top_r

Form the empirical distribution of (ood median − id median) over independent
pair draws (Cartesian pairing). Report the observed shift's percentile and z.

MSMT only. r = 5.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_NULL = 1000
R = 5

CELLS = {
    'id':  ('bot_msmt_seed42_msmt.npz',  11659),
    'ood': ('msmt17_test_noREA.npz',     11659),
}


def load_gallery(fname, n_q):
    d = np.load(CACHE_DIR / fname); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


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


def struct_matched_draws(X, cids, id_top_r_basis, n_draws, rng):
    """Draw n_draws structure-matched surrogates for this arm and return
    the list of median principal angles vs id_top_r_basis."""
    real_cam_sizes = [int((cids == c).sum()) for c in np.unique(cids)]
    mu_g = X.mean(0)
    angles = []
    for _ in range(n_draws):
        perm = rng.permutation(len(X))
        idx = 0
        fake_offsets = []
        for n_imgs in real_cam_sizes:
            group = perm[idx:idx+n_imgs]
            idx += n_imgs
            fake_offsets.append(X[group].mean(0) - mu_g)
        fake_offsets = np.stack(fake_offsets)
        B_fake = top_r_basis(fake_offsets, R)
        angles.append(float(np.median(principal_angles(B_fake, id_top_r_basis))))
    return np.array(angles)


def main():
    out = {}
    arm_data = {}
    for arm, (fname, n_q) in CELLS.items():
        X, pids, cids = load_gallery(fname, n_q)
        X = l2(X)
        arm_data[arm] = {
            'X': X, 'pids': pids, 'cids': cids,
            'id_top_r': top_r_basis(identity_centroids_centered(X, pids), R),
        }

    # Observed angles (same as rev 300)
    obs = {}
    for arm in ['id', 'ood']:
        # Camera offsets
        X = arm_data[arm]['X']; cids = arm_data[arm]['cids']
        mu_g = X.mean(0)
        cam_offsets = np.stack([X[cids == c].mean(0) - mu_g for c in np.unique(cids)])
        B_cam = top_r_basis(cam_offsets, R)
        obs[arm] = float(np.median(principal_angles(B_cam, arm_data[arm]['id_top_r'])))
    obs_shift = obs['ood'] - obs['id']
    print(f"Observed: id={obs['id']:.3f}  ood={obs['ood']:.3f}  shift={obs_shift:+.3f}", flush=True)

    # Structure-matched null draws per arm, SEPARATE RNG streams
    print(f"\nComputing structure-matched draws ({N_NULL} per arm)...", flush=True)
    rng_id = np.random.default_rng(42)
    rng_ood = np.random.default_rng(1337)
    null_id = struct_matched_draws(arm_data['id']['X'], arm_data['id']['cids'],
                                   arm_data['id']['id_top_r'], N_NULL, rng_id)
    null_ood = struct_matched_draws(arm_data['ood']['X'], arm_data['ood']['cids'],
                                    arm_data['ood']['id_top_r'], N_NULL, rng_ood)
    print(f"  id null: median={np.median(null_id):.3f}  mean={null_id.mean():.3f}  sd={null_id.std(ddof=1):.3f}  CI=[{np.quantile(null_id, 0.025):.3f}, {np.quantile(null_id, 0.975):.3f}]")
    print(f"  ood null: median={np.median(null_ood):.3f}  mean={null_ood.mean():.3f}  sd={null_ood.std(ddof=1):.3f}  CI=[{np.quantile(null_ood, 0.025):.3f}, {np.quantile(null_ood, 0.975):.3f}]")

    # Empirical shift distribution: Cartesian pair → convolution.
    # Sample 50000 independent pair draws.
    rng_pair = np.random.default_rng(2026)
    N_PAIR = 50000
    idx_i = rng_pair.integers(0, N_NULL, N_PAIR)
    idx_j = rng_pair.integers(0, N_NULL, N_PAIR)
    shift_null = null_ood[idx_j] - null_id[idx_i]
    print(f"\nEmpirical shift null (ood − id):")
    print(f"  mean={shift_null.mean():+.3f}  median={np.median(shift_null):+.3f}  sd={shift_null.std(ddof=1):.3f}")
    print(f"  CI=[{np.quantile(shift_null, 0.025):+.3f}, {np.quantile(shift_null, 0.975):+.3f}]")
    pct_below = float((shift_null < obs_shift).mean())
    z = (obs_shift - shift_null.mean()) / shift_null.std(ddof=1)
    excess = obs_shift - shift_null.mean()
    print(f"\n  observed shift = {obs_shift:+.3f}")
    print(f"  null-own shift = {shift_null.mean():+.3f}")
    print(f"  excess        = {excess:+.3f}")
    print(f"  pct of null draws below observed: {pct_below:.4f}")
    print(f"  z vs null: {z:+.3f}")

    # Compare to rev-300's covariance-matched shift p = 0.039 and claude-chat's
    # normal approximation p ≈ 0.10.
    print(f"\n  (rev-300 covariance-matched shift p = 0.039 one-tail)")
    print(f"  (claude-chat normal approximation under struct-matched: p ≈ 0.10)")

    out = {
        'observed': {'id_angle': obs['id'], 'ood_angle': obs['ood'], 'shift': obs_shift},
        'null_id_arm':  {'median': float(np.median(null_id)),  'mean': float(null_id.mean()),  'sd': float(null_id.std(ddof=1)),
                         'ci': [float(np.quantile(null_id, 0.025)),  float(np.quantile(null_id, 0.975))]},
        'null_ood_arm': {'median': float(np.median(null_ood)), 'mean': float(null_ood.mean()), 'sd': float(null_ood.std(ddof=1)),
                         'ci': [float(np.quantile(null_ood, 0.025)), float(np.quantile(null_ood, 0.975))]},
        'null_shift': {
            'mean': float(shift_null.mean()), 'median': float(np.median(shift_null)), 'sd': float(shift_null.std(ddof=1)),
            'ci': [float(np.quantile(shift_null, 0.025)), float(np.quantile(shift_null, 0.975))],
            'excess': float(excess),
            'pct_below_observed_one_tail_p': pct_below,
            'z': float(z),
        },
    }
    out_path = RESULTS / 'e43_A2_struct_shift.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
