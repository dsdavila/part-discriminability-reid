"""
C2 structure-matched angle null, PER-ARM gallery covariance (rev 344 fix).

Previous null drew both subspaces from source in-domain gallery. The
two subspaces being compared live in DIFFERENT galleries whose
covariances differ (Sum Var 0.98 vs 0.82 on Market model; mean norms
0.141 vs 0.424). Correct null: draw subspace_1 from the id arm's own
covariance, subspace_2 from the ood arm's own covariance.

Report mean and minimum principal angles against this per-arm null.
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
        'observed_mean_angle': 76.45, 'observed_min_angle': 57.78,
        'label': 'Market model (A′ vs B′)',
    },
    'MSMT_model': {
        'id_file':  'bot_msmt_seed42_msmt.npz',    'id_n_q':  11659,
        'ood_file': 'bot_msmt_seed42_market.npz',  'ood_n_q': 3368,
        'observed_mean_angle': 78.35, 'observed_min_angle': 55.85,
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
        w = w[order]; v = v[:, order]
    else:
        G = Xc @ Xc.T / max(n - 1, 1)
        w, u = np.linalg.eigh(G.astype(np.float64))
        order = np.argsort(-w)
        w = w[order]; u = u[:, order]
        v = (Xc.T @ u) / np.sqrt(np.maximum(w, 1e-12) * (n - 1))
        v = v / np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-12)
    if K_max is not None:
        return v[:, :K_max], w[:K_max]
    return v, w


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
    print(f"\n### {info['label']} (PER-ARM null) ###", flush=True)
    g_id = load_g('id', info)
    g_ood = load_g('ood', info)
    print(f"  id gallery: {g_id.shape}  ood gallery: {g_ood.shape}", flush=True)
    V_id, lam_id = pca_basis(g_id, K_max=min(500, g_id.shape[1], g_id.shape[0] - 1))
    V_ood, lam_ood = pca_basis(g_ood, K_max=min(500, g_ood.shape[1], g_ood.shape[0] - 1))
    print(f"  id top PCs: {np.round(lam_id[:5], 5)}  ...  ood top PCs: {np.round(lam_ood[:5], 5)}", flush=True)
    rng = np.random.default_rng(42)
    mean_angles = []; min_angles = []
    for i in range(N_NULL):
        Q1 = sample_cov_matched_subspace(V_id, lam_id, K, rng)
        Q2 = sample_cov_matched_subspace(V_ood, lam_ood, K, rng)
        angs = principal_angles(Q1, Q2)
        mean_angles.append(angs.mean())
        min_angles.append(angs.min())
        if i < 3 or (i+1) % 100 == 0:
            print(f"    draw {i+1}/{N_NULL}: mean={mean_angles[-1]:.3f}  min={min_angles[-1]:.3f}", flush=True)
    mean_angles = np.array(mean_angles)
    min_angles = np.array(min_angles)
    obs_m = info['observed_mean_angle']; obs_min = info['observed_min_angle']
    def summ(arr, obs):
        in_sup = (arr.min() <= obs <= arr.max())
        return {
            'mean': float(arr.mean()), 'sd': float(arr.std(ddof=1)),
            'ci': [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
            'range': [float(arr.min()), float(arr.max())],
            'excess_obs_minus_null_mean': float(obs - arr.mean()),
            'in_support': bool(in_sup),
            'pct_below_obs': float((arr < obs).mean()),
        }
    mean_null = summ(mean_angles, obs_m)
    min_null = summ(min_angles, obs_min)
    print(f"\n  MEAN angle null: mean={mean_null['mean']:.3f}  sd={mean_null['sd']:.3f}  range=[{mean_null['range'][0]:.3f}, {mean_null['range'][1]:.3f}]  obs={obs_m:.3f}  excess={mean_null['excess_obs_minus_null_mean']:+.3f}  in-support={mean_null['in_support']}  p(below)={mean_null['pct_below_obs']:.4f}")
    print(f"  MIN  angle null: mean={min_null['mean']:.3f}  sd={min_null['sd']:.3f}  range=[{min_null['range'][0]:.3f}, {min_null['range'][1]:.3f}]  obs={obs_min:.3f}  excess={min_null['excess_obs_minus_null_mean']:+.3f}  in-support={min_null['in_support']}  p(below)={min_null['pct_below_obs']:.4f}")
    return {
        'model': model, 'label': info['label'],
        'observed_mean_angle': obs_m, 'observed_min_angle': obs_min,
        'null_mean_angle_perarm': mean_null,
        'null_min_angle_perarm': min_null,
    }


def main():
    out = {}
    for model, info in MODELS.items():
        out[model] = process_model(model, info)
    out_path = RESULTS / 'e43_C2_angle_perarm_null_10k.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n========= C2 per-arm null summary =========")
    for model in MODELS:
        r = out[model]
        m = r['null_mean_angle_perarm']; mi = r['null_min_angle_perarm']
        print(f"  {r['label']}:")
        print(f"    mean: obs={r['observed_mean_angle']:.3f}  null={m['mean']:.3f}  excess={m['excess_obs_minus_null_mean']:+.3f}  in-support={m['in_support']}  p(below)={m['pct_below_obs']:.4f}")
        print(f"    min:  obs={r['observed_min_angle']:.3f}  null={mi['mean']:.3f}  excess={mi['excess_obs_minus_null_mean']:+.3f}  in-support={mi['in_support']}  p(below)={mi['pct_below_obs']:.4f}")


if __name__ == '__main__':
    main()
