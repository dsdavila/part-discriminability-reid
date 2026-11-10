"""
B1 power check: per-item ||Δx||/||x|| for B1's own-arm (b) B_cam projection.
Compare to D1's snap move magnitudes.

For each cell: Δx = projection move = -(x @ B_cam) @ B_cam.T.
Both the raw move (pre-L2-renormalisation) and the renormalised move.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)"},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)"},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)"},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)"},
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
    return nu_c - nu_c.mean(0)


def top_r_basis(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:r].T


def process(tag, info):
    print(f"\n### {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32)
    pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    X_all = l2(emb)
    # Build (b) B_cam from gallery as in B1
    g_emb = X_all[n_q:]; g_pids = pids[n_q:]; g_cids = cids[n_q:]
    n_cams = len(np.unique(g_cids))
    r = n_cams - 1
    nu_c = camera_offsets_b(g_emb, g_pids, g_cids)
    B_cam = top_r_basis(nu_c, r)
    # Per-item projection move (pre-L2)
    X_coords = X_all @ B_cam  # (n, r)
    delta_pre = -(X_coords @ B_cam.T)  # delta before renormalisation: x_proj_raw - x
    norm_delta_pre = np.linalg.norm(delta_pre, axis=1)
    norm_x = np.linalg.norm(X_all, axis=1)
    ratio_pre = norm_delta_pre / np.maximum(norm_x, 1e-12)
    # After L2 renorm
    X_proj_raw = X_all + delta_pre
    X_proj = l2(X_proj_raw)
    delta_post = X_proj - X_all
    norm_delta_post = np.linalg.norm(delta_post, axis=1)
    ratio_post = norm_delta_post / np.maximum(norm_x, 1e-12)
    print(f"  n_items={len(X_all)}  n_cams={n_cams}  r={r}", flush=True)
    print(f"  ||Δx||/||x|| PRE-renorm:  mean={ratio_pre.mean():.4f}  median={np.median(ratio_pre):.4f}  max={ratio_pre.max():.4f}  p95={np.quantile(ratio_pre, 0.95):.4f}", flush=True)
    print(f"  ||Δx||/||x|| POST-renorm: mean={ratio_post.mean():.4f}  median={np.median(ratio_post):.4f}  max={ratio_post.max():.4f}  p95={np.quantile(ratio_post, 0.95):.4f}", flush=True)
    return {
        'cell': tag, 'label': info['label'], 'r': r,
        'pre_renorm': {
            'mean': float(ratio_pre.mean()), 'median': float(np.median(ratio_pre)),
            'max': float(ratio_pre.max()), 'p95': float(np.quantile(ratio_pre, 0.95)),
        },
        'post_renorm': {
            'mean': float(ratio_post.mean()), 'median': float(np.median(ratio_post)),
            'max': float(ratio_post.max()), 'p95': float(np.quantile(ratio_post, 0.95)),
        },
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process(tag, info)
    out_path = RESULTS / 'e43_B1_power_check.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= B1 power check vs D1 =========")
    # D1 nearest ratios (from committed e43_D1_power_and_train_containment.json)
    try:
        with open(RESULTS / 'e43_D1_power_and_train_containment.json') as f:
            d1 = json.load(f)
    except Exception:
        d1 = None
    for tag in CELLS:
        r = out[tag]
        b1_pre = r['pre_renorm']['mean']
        b1_post = r['post_renorm']['mean']
        d1_near = d1['power_check'][tag]['nearest_ratio']['mean'] if d1 else float('nan')
        print(f"  {tag}: B1 pre={b1_pre:.4f}  B1 post={b1_post:.4f}  D1 nearest={d1_near:.4f}")


if __name__ == '__main__':
    main()
