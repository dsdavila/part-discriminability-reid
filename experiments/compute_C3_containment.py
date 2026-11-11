"""
C3 whole-embedding containment (pre-reg at block .1178580 / rev 358).

For each model:
  Build training top-k subspace by PCA on training gallery (all items).
  For each k in {10, 50, 203, 500}:
    id_share  = variance share of IN-DOMAIN TEST gallery in top-k subspace
    ood_share = variance share of OOD gallery in top-k subspace
    ratio = ood_share / id_share
  Structure-matched null: draw random k-dim subspaces from TRAINING feature
  covariance (anisotropic baseline). For each draw, compute ood_share and
  id_share. 100 draws per k.
  Pre-reg at k=203: ratio ∈ [0.5, 0.9); falsified below 0.3 or at/above 0.9.
  Also report x-chance for each share.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
TRAIN_CACHE = Path('/tmp/e18_train_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
D_EMB = 2048
K_SWEEP = [10, 50, 203, 500]
N_NULL = 100
RNG_SEED = 42

MODELS = {
    'Market_model': {
        'train_file': 'bot_market_market_train.npz',
        'id_file':    'market_test_noREA.npz',   'id_n_q':  3368,
        'ood_file':   'msmt17_test_noREA.npz',   'ood_n_q': 11659,
        'label': 'Market model: train top-k vs A′ (id) and B′ (ood)',
    },
    'MSMT_model': {
        'train_file': 'bot_msmt_msmt_train.npz',
        'id_file':    'bot_msmt_seed42_msmt.npz',   'id_n_q':  11659,
        'ood_file':   'bot_msmt_seed42_market.npz', 'ood_n_q': 3368,
        'label': 'MSMT model: train top-k vs D (id) and C (ood)',
    },
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def pca_basis_full(X, K_max=None):
    mu = X.mean(0); Xc = X - mu
    n, d = Xc.shape
    if d <= n:
        C = Xc.T @ Xc / max(n - 1, 1)
        w, v = np.linalg.eigh(C.astype(np.float64))
        order = np.argsort(-w)
        return mu, v[:, order[:K_max]].astype(np.float32), w[order[:K_max]].astype(np.float32)
    G = Xc @ Xc.T / max(n - 1, 1)
    w, u = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(-w)
    w_sel = w[order[:K_max]]; u_sel = u[:, order[:K_max]]
    v = (Xc.T @ u_sel) / np.sqrt(np.maximum(w_sel, 1e-12) * (n - 1))
    v = v / np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-12)
    return mu, v.astype(np.float32), w_sel.astype(np.float32)


def variance_share(X, mu_train, B):
    """Share of variance of X (after centering at mu_train) captured by
    subspace B. = ||(X - mu_train) @ B||_F^2 / ||X - mu_train||_F^2."""
    Xc = X - mu_train
    num = ((Xc @ B) ** 2).sum()
    den = (Xc ** 2).sum()
    return float(num / max(den, 1e-12))


def sample_cov_matched_subspace(V, lam, K, rng):
    n_pcs = V.shape[1]
    coef = rng.standard_normal((n_pcs, K)) * np.sqrt(np.maximum(lam, 0))[:, None]
    Q, _ = np.linalg.qr(V @ coef)
    return Q


def process_model(model, info):
    print(f"\n### {info['label']} ###", flush=True)
    dt = np.load(TRAIN_CACHE / info['train_file'])
    t_emb = l2(dt['emb'].astype(np.float32))
    print(f"  loading train: n={len(t_emb)}", flush=True)
    # Full PCA basis (up to max K_sweep for observed; up to 500 for null too)
    K_total = min(500, t_emb.shape[1], t_emb.shape[0] - 1)
    mu_train, V_all, eigs_all = pca_basis_full(t_emb, K_max=K_total)
    print(f"  train top-500 PCs: lam[:5]={np.round(eigs_all[:5], 5)}  lam[-1]={eigs_all[-1]:.5f}", flush=True)

    # Load id and ood galleries
    d_id = np.load(CACHE_DIR / info['id_file'])
    X_id = l2(d_id['emb'].astype(np.float32)[info['id_n_q']:])
    d_ood = np.load(CACHE_DIR / info['ood_file'])
    X_ood = l2(d_ood['emb'].astype(np.float32)[info['ood_n_q']:])
    print(f"  id: n={len(X_id)}  ood: n={len(X_ood)}", flush=True)

    out_per_k = {}
    rng = np.random.default_rng(RNG_SEED)
    for k in K_SWEEP:
        if k > K_total:
            print(f"  k={k} exceeds K_total={K_total}, skip", flush=True)
            continue
        B = V_all[:, :k]
        id_share = variance_share(X_id, mu_train, B)
        ood_share = variance_share(X_ood, mu_train, B)
        ratio = ood_share / max(id_share, 1e-12)
        x_chance = (id_share + ood_share) / 2 / (k / D_EMB)
        print(f"  k={k}: id_share={id_share:.4f}  ood_share={ood_share:.4f}  ratio={ratio:.4f}  x-chance avg={x_chance:.2f}", flush=True)

        # Null: random cov-matched subspaces
        print(f"    structure-matched null ({N_NULL} draws)...", flush=True)
        null_id = []; null_ood = []; null_ratio = []
        for di in range(N_NULL):
            Q = sample_cov_matched_subspace(V_all, eigs_all, k, rng)
            is_ = variance_share(X_id, mu_train, Q)
            os_ = variance_share(X_ood, mu_train, Q)
            null_id.append(is_); null_ood.append(os_); null_ratio.append(os_ / max(is_, 1e-12))
            if di < 2 or (di + 1) % 25 == 0:
                print(f"      draw {di+1}: id={is_:.4f}  ood={os_:.4f}  ratio={null_ratio[-1]:.4f}", flush=True)
        null_id = np.array(null_id); null_ood = np.array(null_ood); null_ratio = np.array(null_ratio)
        print(f"    null id: mean={null_id.mean():.4f}  sd={null_id.std(ddof=1):.5f}", flush=True)
        print(f"    null ood: mean={null_ood.mean():.4f}  sd={null_ood.std(ddof=1):.5f}", flush=True)
        print(f"    null ratio: mean={null_ratio.mean():.4f}  sd={null_ratio.std(ddof=1):.4f}  range=[{null_ratio.min():.4f}, {null_ratio.max():.4f}]", flush=True)
        out_per_k[k] = {
            'id_share': id_share, 'ood_share': ood_share, 'ratio': ratio,
            'x_chance_id': id_share / (k / D_EMB),
            'x_chance_ood': ood_share / (k / D_EMB),
            'null_id_mean': float(null_id.mean()), 'null_id_sd': float(null_id.std(ddof=1)),
            'null_ood_mean': float(null_ood.mean()), 'null_ood_sd': float(null_ood.std(ddof=1)),
            'null_ratio_mean': float(null_ratio.mean()), 'null_ratio_sd': float(null_ratio.std(ddof=1)),
            'null_ratio_range': [float(null_ratio.min()), float(null_ratio.max())],
            'excess_id': float(id_share - null_id.mean()),
            'excess_ood': float(ood_share - null_ood.mean()),
            'z_id': float((id_share - null_id.mean()) / null_id.std(ddof=1)),
            'z_ood': float((ood_share - null_ood.mean()) / null_ood.std(ddof=1)),
        }
    return out_per_k


def main():
    out = {}
    for model, info in MODELS.items():
        out[model] = process_model(model, info)
    out_path = RESULTS / 'e43_C3_containment.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= C3 containment summary =========")
    for model in MODELS:
        print(f"\n  {model}:")
        print(f"    {'k':>4s}  {'id_share':>9s}  {'ood_share':>9s}  {'ratio':>7s}  {'null_ratio':>10s}  {'x-chance id':>11s}  {'x-chance ood':>12s}  {'z_id':>6s}  {'z_ood':>6s}")
        for k in K_SWEEP:
            if k not in out[model]: continue
            r = out[model][k]
            print(f"    {k:>4d}  {r['id_share']:9.4f}  {r['ood_share']:9.4f}  {r['ratio']:7.4f}  {r['null_ratio_mean']:10.4f}  {r['x_chance_id']:11.2f}  {r['x_chance_ood']:12.2f}  {r['z_id']:+6.2f}  {r['z_ood']:+6.2f}")
    # Pre-reg verdict at k=203
    print("\n  Pre-reg at k=203 (ratio ood/id in [0.5, 0.9), falsified <0.3 or >=0.9):")
    for model in MODELS:
        if 203 in out[model]:
            r = out[model][203]['ratio']
            if r < 0.3: verdict = "FALSIFIED (below 0.3)"
            elif r >= 0.9: verdict = "FALSIFIED (>=0.9)"
            elif 0.5 <= r < 0.9: verdict = "CONFIRMED (in [0.5, 0.9))"
            else: verdict = f"MARGINAL (in [0.3, 0.5))"
            print(f"    {model}: ratio = {r:.4f}  → {verdict}")


if __name__ == '__main__':
    main()
