"""
D1 power check + training-side containment share (rev 364).

Part 1: Per-item ||Δx||/||x|| magnitude of the D1 snap.
  Δx = x_snap - x where x_snap = x - P_cam_train(x) + P_cam_train(nu_c*)
  and c* is the nearest training lane (and the uniformly random lane
  for the control). Report mean and distribution per cell.

Part 2: Training-side containment share.
  For each model, build the TRAINING-side (b) identity-controlled camera
  span. Project the OOD arm's cam offsets onto it. Report share of OOD
  cam energy captured.
    Market model: train span from /tmp/e18_train_cache/bot_market_market_train.npz
                  probe = B' (msmt17_test_noREA.npz) cam offsets
    MSMT model:   train span from /tmp/e18_train_cache/bot_msmt_msmt_train.npz
                  probe = C (bot_msmt_seed42_market.npz) cam offsets
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
TRAIN_CACHE = Path('/tmp/e18_train_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
RNG_SEED = 42

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)",  'train_file': 'bot_market_market_train.npz'},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)",   'train_file': 'bot_market_market_train.npz'},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)",   'train_file': 'bot_msmt_msmt_train.npz'},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)",     'train_file': 'bot_msmt_msmt_train.npz'},
}


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


def snap_to_nearest_lane(X, B_cam_train, nu_c_train):
    """Return (X_snap unnormalized, nearest-lane assignment)."""
    x_coords = X @ B_cam_train
    lane_coords = nu_c_train @ B_cam_train
    d2 = ((x_coords[:, None, :] - lane_coords[None, :, :]) ** 2).sum(axis=-1)
    nearest = d2.argmin(axis=1)
    target_coords = lane_coords[nearest]
    X_snap = X - x_coords @ B_cam_train.T + target_coords @ B_cam_train.T
    return X_snap, nearest


def snap_to_random_lane(X, B_cam_train, nu_c_train, rng):
    lane_coords = nu_c_train @ B_cam_train
    n_cams_train = lane_coords.shape[0]
    rand_assign = rng.integers(0, n_cams_train, size=len(X))
    target_coords = lane_coords[rand_assign]
    x_coords = X @ B_cam_train
    X_snap = X - x_coords @ B_cam_train.T + target_coords @ B_cam_train.T
    return X_snap, rand_assign


def part1_power_check(tag, info):
    """D1 power check: per-item ||Δx||/||x|| for nearest and random snap."""
    print(f"\n### Part 1: D1 power check — {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); n_q = info['n_q']
    X = l2(emb)  # all items (q + g), L2-normalized
    dt = np.load(TRAIN_CACHE / info['train_file'])
    t_emb = l2(dt['emb'].astype(np.float32)); t_pids = dt['pids']; t_cids = dt['cids']
    n_cams_train = len(np.unique(t_cids))
    r_train = n_cams_train - 1
    nu_c_train, _ = camera_offsets_b(t_emb, t_pids, t_cids)
    B_cam_train, _ = top_r_basis(nu_c_train, r_train)

    # Nearest snap
    X_snap_n, _ = snap_to_nearest_lane(X, B_cam_train, nu_c_train)
    delta_n = X_snap_n - X
    norm_delta_n = np.linalg.norm(delta_n, axis=1)
    # ||x|| is 1 since L2-normalized, but compute anyway for clarity
    norm_x = np.linalg.norm(X, axis=1)
    ratio_n = norm_delta_n / np.maximum(norm_x, 1e-12)
    # Random snap
    rng = np.random.default_rng(RNG_SEED)
    X_snap_r, _ = snap_to_random_lane(X, B_cam_train, nu_c_train, rng)
    delta_r = X_snap_r - X
    norm_delta_r = np.linalg.norm(delta_r, axis=1)
    ratio_r = norm_delta_r / np.maximum(norm_x, 1e-12)

    print(f"  n_items={len(X)}  n_cams_train={n_cams_train}  r_train={r_train}", flush=True)
    print(f"  NEAREST snap ||Δx||/||x||:  mean={ratio_n.mean():.4f}  median={np.median(ratio_n):.4f}  max={ratio_n.max():.4f}  p95={np.quantile(ratio_n, 0.95):.4f}", flush=True)
    print(f"  RANDOM  snap ||Δx||/||x||:  mean={ratio_r.mean():.4f}  median={np.median(ratio_r):.4f}  max={ratio_r.max():.4f}  p95={np.quantile(ratio_r, 0.95):.4f}", flush=True)

    return {
        'cell': tag,
        'label': info['label'],
        'n_items': int(len(X)),
        'nearest_ratio': {
            'mean': float(ratio_n.mean()), 'median': float(np.median(ratio_n)),
            'max': float(ratio_n.max()), 'p95': float(np.quantile(ratio_n, 0.95)),
        },
        'random_ratio': {
            'mean': float(ratio_r.mean()), 'median': float(np.median(ratio_r)),
            'max': float(ratio_r.max()), 'p95': float(np.quantile(ratio_r, 0.95)),
        },
    }


def part2_training_containment():
    """Training-side containment share: each model's TRAINING camera span
    captures what fraction of the OOD-arm camera-offset energy.
    """
    print(f"\n### Part 2: Training-side containment share ###", flush=True)
    results = {}
    MODELS = {
        'Market_model': {
            'train_file': 'bot_market_market_train.npz',
            'probe_cell': 'B_prime',
            'probe_file': 'msmt17_test_noREA.npz',
            'probe_n_q': 11659,
            'label': 'Market model: train span vs B′ (MSMT) cam offsets',
        },
        'MSMT_model': {
            'train_file': 'bot_msmt_msmt_train.npz',
            'probe_cell': 'C',
            'probe_file': 'bot_msmt_seed42_market.npz',
            'probe_n_q': 3368,
            'label': 'MSMT model: train span vs C (Market) cam offsets',
        },
    }
    for model, cfg in MODELS.items():
        print(f"\n  --- {model}: {cfg['label']} ---", flush=True)
        # Training (b) B_cam span
        dt = np.load(TRAIN_CACHE / cfg['train_file'])
        t_emb = l2(dt['emb'].astype(np.float32))
        t_pids = dt['pids']; t_cids = dt['cids']
        n_cams_train = len(np.unique(t_cids))
        nu_c_train, _ = camera_offsets_b(t_emb, t_pids, t_cids)
        # Use full rank (n_cams - 1) for the span
        r_train = n_cams_train - 1
        B_train, s_train = top_r_basis(nu_c_train, r_train)
        print(f"    train: n_items={len(t_emb)}  n_cams={n_cams_train}  r_train={r_train}  sval[:5]={np.round(s_train[:5], 4)}", flush=True)
        # Probe (b) cam offsets
        dp = np.load(CACHE_DIR / cfg['probe_file'])
        p_emb = l2(dp['emb'].astype(np.float32)); p_pids = dp['pids']; p_cids = dp['cids']
        n_q = cfg['probe_n_q']
        X_probe = p_emb[n_q:]; p_pids_g = p_pids[n_q:]; p_cids_g = p_cids[n_q:]
        nu_c_probe, _ = camera_offsets_b(X_probe, p_pids_g, p_cids_g)
        print(f"    probe: n_items={len(X_probe)}  n_cams={len(np.unique(p_cids_g))}  nu_probe.shape={nu_c_probe.shape}", flush=True)
        # Capture share: ||proj_Btrain(nu_c_probe)||^2 / ||nu_c_probe||^2 per probe cam
        per_cam_capture = []
        for i, nu in enumerate(nu_c_probe):
            if np.linalg.norm(nu) < 1e-12:
                per_cam_capture.append(0.0)
                continue
            coef = B_train.T @ nu
            cap = float((coef ** 2).sum() / (nu ** 2).sum())
            per_cam_capture.append(cap)
        per_cam_capture = np.array(per_cam_capture)
        mean_cap = float(per_cam_capture.mean())
        print(f"    per-cam capture: mean={mean_cap:.4f}  min={per_cam_capture.min():.4f}  max={per_cam_capture.max():.4f}", flush=True)
        print(f"    values: {np.round(per_cam_capture, 4).tolist()}", flush=True)
        # Compare to Item 3 (test-side) capture, re-derived from existing JSON
        results[model] = {
            'label': cfg['label'],
            'n_train': int(len(t_emb)),
            'n_cams_train': int(n_cams_train),
            'r_train': int(r_train),
            'n_probe_cams': int(len(np.unique(p_cids_g))),
            'per_cam_capture': per_cam_capture.tolist(),
            'mean_capture': mean_cap,
        }
    return results


def main():
    out = {'power_check': {}, 'training_containment': {}}
    for tag, info in CELLS.items():
        out['power_check'][tag] = part1_power_check(tag, info)
    out['training_containment'] = part2_training_containment()
    out_path = RESULTS / 'e43_D1_power_and_train_containment.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n========= Part 1 SUMMARY: ||Δx||/||x|| =========")
    for tag in CELLS:
        r = out['power_check'][tag]
        print(f"  {tag}: NEAREST mean={r['nearest_ratio']['mean']:.4f}  RANDOM mean={r['random_ratio']['mean']:.4f}")

    print("\n========= Part 2 SUMMARY: training-side containment =========")
    for model in out['training_containment']:
        r = out['training_containment'][model]
        print(f"  {model}: mean capture = {r['mean_capture']:.4f}  ({r['r_train']} dims of train span, {r['n_probe_cams']} probe cams)")


if __name__ == '__main__':
    main()
