"""
Mean-of-9 null for C2 row 2 (rev 353 close).

Per-arm covariance-matched null, same construction as the 10k run, but
store ALL 10 principal angles per pair draw. Then compute:
  - mean-of-10 (should match prior)
  - min-of-10
  - mean-of-9 (leave min out)

Observed mean-of-9 (from rev 338/343 angle lists):
  Market: angles [57.78, 63.21, 70.05, 75.25, 76.64, 79.77, 81.78, 84.77, 87.03, 88.22]
          min excluded -> mean of 9 = 78.52
  MSMT:   angles [55.85, 68.26, 73.17, 76.52, 80.11, 82.47, 84.75, 85.46, 88.00, 88.87]
          min excluded -> mean of 9 = 80.86

For each arm's test gallery feature covariance, draw 10,000 pairs of
10-dim subspaces independently per arm. Compute all 10 principal angles,
sort, take mean-of-9 (indices 1..9). Report null mean-of-9 distribution
and p for the observed mean-of-9.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_NULL = 10000
K = 10

MODELS = {
    'Market_model': {
        'id_file':  'market_test_noREA.npz',  'id_n_q':  3368,
        'ood_file': 'msmt17_test_noREA.npz',  'ood_n_q': 11659,
        'obs_angles': [57.78, 63.21, 70.05, 75.25, 76.64, 79.77, 81.78, 84.77, 87.03, 88.22],
        'label': 'Market model (A′ vs B′)',
    },
    'MSMT_model': {
        'id_file':  'bot_msmt_seed42_msmt.npz',    'id_n_q':  11659,
        'ood_file': 'bot_msmt_seed42_market.npz',  'ood_n_q': 3368,
        'obs_angles': [55.85, 68.26, 73.17, 76.52, 80.11, 82.47, 84.75, 85.46, 88.00, 88.87],
        'label': 'MSMT model (D vs C)',
    },
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def principal_angles(B1, B2):
    s = np.linalg.svd(B1.T @ B2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def pca_basis(X, K_max=None):
    mu = X.mean(0); Xc = X - mu
    n, d = Xc.shape
    if d <= n:
        C = Xc.T @ Xc / max(n - 1, 1)
        w, v = np.linalg.eigh(C.astype(np.float64))
        order = np.argsort(-w)
        return v[:, order[:K_max]].astype(np.float32), w[order[:K_max]].astype(np.float32)
    G = Xc @ Xc.T / max(n - 1, 1)
    w, u = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(-w)
    w_sel = w[order[:K_max]]; u_sel = u[:, order[:K_max]]
    v = (Xc.T @ u_sel) / np.sqrt(np.maximum(w_sel, 1e-12) * (n - 1))
    v = v / np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-12)
    return v.astype(np.float32), w_sel.astype(np.float32)


def sample_cov_matched_subspace(V, lam, K, rng):
    n_pcs = V.shape[1]
    coef = rng.standard_normal((n_pcs, K)) * np.sqrt(np.maximum(lam, 0))[:, None]
    Q, _ = np.linalg.qr(V @ coef)
    return Q


def load_g(info_key, info):
    d = np.load(CACHE_DIR / info[f'{info_key}_file'])
    emb = d['emb'].astype(np.float32); n_q = info[f'{info_key}_n_q']
    return l2(emb[n_q:])


def process_model(model, info):
    print(f"\n### {info['label']} ###", flush=True)
    obs_angles = np.array(info['obs_angles'])
    obs_mean = obs_angles.mean()
    obs_min = obs_angles.min()
    obs_mean_9 = np.sort(obs_angles)[1:].mean()
    print(f"  obs: mean-10={obs_mean:.3f}  min={obs_min:.3f}  mean-9 (min excluded)={obs_mean_9:.3f}", flush=True)

    g_id = load_g('id', info)
    g_ood = load_g('ood', info)
    V_id, lam_id = pca_basis(g_id, K_max=min(500, g_id.shape[1], g_id.shape[0] - 1))
    V_ood, lam_ood = pca_basis(g_ood, K_max=min(500, g_ood.shape[1], g_ood.shape[0] - 1))
    print(f"  drawing {N_NULL} null pairs...", flush=True)
    rng = np.random.default_rng(42)
    null_mean10 = np.empty(N_NULL)
    null_min = np.empty(N_NULL)
    null_mean9 = np.empty(N_NULL)
    for i in range(N_NULL):
        Q1 = sample_cov_matched_subspace(V_id, lam_id, K, rng)
        Q2 = sample_cov_matched_subspace(V_ood, lam_ood, K, rng)
        angs = np.sort(principal_angles(Q1, Q2))
        null_mean10[i] = angs.mean()
        null_min[i] = angs[0]
        null_mean9[i] = angs[1:].mean()
        if i < 3 or (i+1) % 2000 == 0:
            print(f"    draw {i+1}/{N_NULL}: mean10={null_mean10[i]:.3f}  min={null_min[i]:.3f}  mean9={null_mean9[i]:.3f}", flush=True)

    def summ(arr, obs, label):
        return {
            'label': label,
            'mean': float(arr.mean()), 'sd': float(arr.std(ddof=1)),
            'ci': [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
            'range': [float(arr.min()), float(arr.max())],
            'obs': float(obs),
            'excess': float(obs - arr.mean()),
            'pct_below_obs': float((arr < obs).mean()),
            'in_support': bool(arr.min() <= obs <= arr.max()),
            'z': float((obs - arr.mean()) / arr.std(ddof=1)),
        }
    return {
        'model': model, 'label': info['label'],
        'obs_angles': obs_angles.tolist(),
        'mean_10': summ(null_mean10, obs_mean, 'mean-of-10'),
        'min': summ(null_min, obs_min, 'min-of-10'),
        'mean_9': summ(null_mean9, obs_mean_9, 'mean-of-9'),
    }


def main():
    out = {}
    for model, info in MODELS.items():
        out[model] = process_model(model, info)
    out_path = RESULTS / 'e43_C2_mean_of_9_null.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= C2 mean-of-9 null summary =========")
    for model in MODELS:
        r = out[model]
        print(f"\n  {r['label']}:")
        for key in ['mean_10', 'min', 'mean_9']:
            s = r[key]
            print(f"    {s['label']:>14s}: obs={s['obs']:.3f}  null={s['mean']:.3f}±{s['sd']:.3f} range=[{s['range'][0]:.3f}, {s['range'][1]:.3f}]  excess={s['excess']:+.3f}  z={s['z']:+.2f}  in-support={s['in_support']}  p(below)={s['pct_below_obs']:.5f}")


if __name__ == '__main__':
    main()
