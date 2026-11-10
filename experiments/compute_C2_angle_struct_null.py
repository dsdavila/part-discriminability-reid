"""
C2 structure-matched angle null (rev 341).

Uniform-null principal angle in 2048D is 86.6 degrees. Observed 76.5
and 78.4 degrees are MORE aligned than chance. The correct null for
principal angles in an anisotropic embedding is subspaces drawn to
match the gallery feature covariance (where real features live), not
uniformly on the sphere.

For each arm (A', B', C, D):
  Compute gallery feature covariance eigendecomposition.
  Draw N pairs of 10-dim subspaces from N(0, Sigma_gallery): for each,
    (i) K=10 Gaussian vectors in the PC basis with per-PC std sqrt(lam_k),
   (ii) QR orthonormalize -> 10-dim subspace.
  Compute mean principal angle between the two subspaces per draw.
  Report null distribution mean/sd/range.

Observed mean principal angles (from C2 eigen rebuild):
  Market (A' vs B'): 76.45  (uniform-null 86.62)
  MSMT   (D vs C):   78.35  (uniform-null 86.62)

Per model, use the gallery covariance that is the arm-combined feature
space the two M matrices live in. Since each arm has its own gallery,
compute the null under the SOURCE in-domain gallery (A' for Market
model; D for MSMT model) - this is the ambient geometry the model was
trained to produce.

Report excess = observed - null_mean, and whether observed is outside
the realised null range.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_NULL = 500
K = 10

ANCHORS = {
    'Market_model': {'file': 'market_test_noREA.npz', 'n_q': 3368,
                     'observed_mean_angle': 76.45,
                     'observed_min_angle': 57.78,
                     'label': 'Market model (A′ vs B′)'},
    'MSMT_model':   {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659,
                     'observed_mean_angle': 78.35,
                     'observed_min_angle': 55.85,
                     'label': 'MSMT model (D vs C)'},
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
    """Draw K orthonormal vectors from N(0, V diag(lam) V^T).
    Returns (d, K) orthonormal basis.
    """
    d = V.shape[0]
    n_pcs = V.shape[1]
    coef = rng.standard_normal((n_pcs, K)) * np.sqrt(np.maximum(lam, 0))[:, None]
    B_in_full = V @ coef
    Q, _ = np.linalg.qr(B_in_full)
    return Q


def process_model(model, info):
    print(f"\n### {info['label']} ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); n_q = info['n_q']
    g_emb = l2(emb[n_q:])
    print(f"  computing gallery feature covariance eigendecomp ({g_emb.shape[0]} x {g_emb.shape[1]})...", flush=True)
    V, lam = pca_basis(g_emb, K_max=min(500, g_emb.shape[1], g_emb.shape[0] - 1))
    # Use all 500 PCs for the null
    print(f"  top PC eigs: {np.round(lam[:5], 5)}  ...  tail {np.round(lam[-5:], 5)}", flush=True)
    print(f"  drawing {N_NULL} null pairs...", flush=True)
    rng = np.random.default_rng(42)
    mean_angles = []; min_angles = []
    for i in range(N_NULL):
        Q1 = sample_cov_matched_subspace(V, lam, K, rng)
        Q2 = sample_cov_matched_subspace(V, lam, K, rng)
        angs = principal_angles(Q1, Q2)
        mean_angles.append(angs.mean())
        min_angles.append(angs.min())
        if i < 3 or (i+1) % 100 == 0:
            print(f"    draw {i+1}/{N_NULL}: mean_ang={mean_angles[-1]:.3f}  min_ang={min_angles[-1]:.3f}", flush=True)
    mean_angles = np.array(mean_angles)
    min_angles = np.array(min_angles)

    obs_mean = info['observed_mean_angle']
    obs_min = info['observed_min_angle']
    mean_z = (obs_mean - mean_angles.mean()) / mean_angles.std(ddof=1)
    min_z = (obs_min - min_angles.mean()) / min_angles.std(ddof=1)
    mean_in = (mean_angles.min() <= obs_mean <= mean_angles.max())
    min_in = (min_angles.min() <= obs_min <= min_angles.max())
    pct_mean_below = float((mean_angles < obs_mean).mean())
    pct_min_below = float((min_angles < obs_min).mean())
    print(f"\n  MEAN angle null: mean={mean_angles.mean():.3f}  sd={mean_angles.std(ddof=1):.3f}  CI=[{np.quantile(mean_angles, 0.025):.3f}, {np.quantile(mean_angles, 0.975):.3f}]  range=[{mean_angles.min():.3f}, {mean_angles.max():.3f}]")
    print(f"  obs mean angle = {obs_mean:.3f}  z={mean_z:+.3f}  in-support={mean_in}  pct below obs={pct_mean_below:.4f}")
    print(f"  MIN  angle null: mean={min_angles.mean():.3f}  sd={min_angles.std(ddof=1):.3f}  CI=[{np.quantile(min_angles, 0.025):.3f}, {np.quantile(min_angles, 0.975):.3f}]  range=[{min_angles.min():.3f}, {min_angles.max():.3f}]")
    print(f"  obs min angle = {obs_min:.3f}  z={min_z:+.3f}  in-support={min_in}  pct below obs={pct_min_below:.4f}")

    return {
        'model': model, 'label': info['label'],
        'observed_mean_angle': obs_mean, 'observed_min_angle': obs_min,
        'null_mean_angle': {
            'mean': float(mean_angles.mean()), 'sd': float(mean_angles.std(ddof=1)),
            'ci': [float(np.quantile(mean_angles, 0.025)), float(np.quantile(mean_angles, 0.975))],
            'range': [float(mean_angles.min()), float(mean_angles.max())],
            'z': float(mean_z), 'in_support': bool(mean_in), 'pct_below_obs': pct_mean_below,
        },
        'null_min_angle': {
            'mean': float(min_angles.mean()), 'sd': float(min_angles.std(ddof=1)),
            'ci': [float(np.quantile(min_angles, 0.025)), float(np.quantile(min_angles, 0.975))],
            'range': [float(min_angles.min()), float(min_angles.max())],
            'z': float(min_z), 'in_support': bool(min_in), 'pct_below_obs': pct_min_below,
        },
    }


def main():
    out = {}
    for model, info in ANCHORS.items():
        out[model] = process_model(model, info)
    out_path = RESULTS / 'e43_C2_angle_struct_null.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= C2 angle structure-matched null =========")
    for model in ANCHORS:
        r = out[model]
        m = r['null_mean_angle']; mi = r['null_min_angle']
        print(f"\n  {r['label']}:")
        print(f"    mean angle: obs={r['observed_mean_angle']:.2f}  null={m['mean']:.2f}  excess={r['observed_mean_angle'] - m['mean']:+.3f}  z={m['z']:+.2f}  in-support={m['in_support']}")
        print(f"    min angle:  obs={r['observed_min_angle']:.2f}  null={mi['mean']:.2f}  excess={r['observed_min_angle'] - mi['mean']:+.3f}  z={mi['z']:+.2f}  in-support={mi['in_support']}")


if __name__ == '__main__':
    main()
