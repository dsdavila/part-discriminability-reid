"""
C3 foreign-training-subspace control (rev 371).

C3 showed OOD gallery captures ~same share as id in own-model's training
top-k subspace. The null ratio is also ~1.0, so the statistic may have
no power. Required control: project each model's OOD gallery onto the
OTHER model's training subspace.

- Market model's OOD gallery = B' (msmt17_test_noREA.npz)
- MSMT model's OOD gallery = C (bot_msmt_seed42_market.npz)

For each foreign pair:
  Build foreign model's training top-k PC subspace at k in {10, 50, 203, 500}
  Project OOD gallery onto it; report capture.
  Compare to the own-model capture from the C3 run.

If foreign ≈ own, "learned" is doing no work (C3 weakens). If foreign <
own, C3 is specific.

Important: features are model-specific, so "foreign" means:
  Market-OOD gallery embedded by Market model vs MSMT training-set embedded by MSMT model
  — these live in different embedding spaces (both R^2048 but produced by
  different networks).
This is a CROSS-MODEL projection test. The projection formula still
operates mathematically on R^2048 vectors, but the question is whether
MSMT-model directions capture Market-model's OOD variance.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
TRAIN_CACHE = Path('/tmp/e18_train_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
D_EMB = 2048
K_SWEEP = [10, 50, 203, 500]
N_NULL = 50
RNG_SEED = 42

# Each setup: (foreign_train_cache, own_ood_cache)
FOREIGN_SETUPS = {
    'Bprime_vs_foreign_MSMTtrain_in_Marketspace': {
        # B' = Market model on MSMT test (Market embedding space)
        # Foreign subspace = MSMT train images embedded by MARKET model (same space)
        # -> isolates "which images defined the directions"
        'foreign_train_file': 'bot_market_msmt_train.npz',
        'own_ood_file': 'msmt17_test_noREA.npz',
        'own_ood_n_q': 11659,
        'label': "B' (Market-model space): proj onto Market-model-embedded MSMT-train subspace",
    },
    'C_vs_foreign_Markettrain_in_MSMTspace': {
        # C = MSMT model on Market test (MSMT embedding space)
        # Foreign subspace = Market train images embedded by MSMT model (same space)
        'foreign_train_file': 'bot_msmt_market_train.npz',
        'own_ood_file': 'bot_msmt_seed42_market.npz',
        'own_ood_n_q': 3368,
        'label': "C (MSMT-model space): proj onto MSMT-model-embedded Market-train subspace",
    },
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def pca_basis(X, K_max=None):
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
    Xc = X - mu_train
    num = ((Xc @ B) ** 2).sum()
    den = (Xc ** 2).sum()
    return float(num / max(den, 1e-12))


def sample_cov_matched_subspace(V, lam, K, rng):
    n_pcs = V.shape[1]
    coef = rng.standard_normal((n_pcs, K)) * np.sqrt(np.maximum(lam, 0))[:, None]
    Q, _ = np.linalg.qr(V @ coef)
    return Q


def process(name, info):
    print(f"\n### {name}: {info['label']} ###", flush=True)
    dt = np.load(TRAIN_CACHE / info['foreign_train_file'])
    t_emb = l2(dt['emb'].astype(np.float32))
    K_total = min(500, t_emb.shape[1], t_emb.shape[0] - 1)
    mu_train, V_all, eigs_all = pca_basis(t_emb, K_max=K_total)
    print(f"  foreign train: n={len(t_emb)}  top-5 lam={np.round(eigs_all[:5], 5)}", flush=True)

    d_ood = np.load(CACHE_DIR / info['own_ood_file'])
    X_ood = l2(d_ood['emb'].astype(np.float32)[info['own_ood_n_q']:])
    print(f"  ood gallery: n={len(X_ood)}", flush=True)

    out_per_k = {}
    rng = np.random.default_rng(RNG_SEED)
    for k in K_SWEEP:
        if k > K_total: continue
        B = V_all[:, :k]
        obs = variance_share(X_ood, mu_train, B)
        x_chance = obs / (k / D_EMB)
        # Null: cov-matched random subspaces from foreign-train cov
        null_caps = []
        for di in range(N_NULL):
            Q = sample_cov_matched_subspace(V_all, eigs_all, k, rng)
            null_caps.append(variance_share(X_ood, mu_train, Q))
        null_caps = np.array(null_caps)
        excess = obs - null_caps.mean()
        z = (obs - null_caps.mean()) / null_caps.std(ddof=1) if null_caps.std(ddof=1) > 0 else 0.0
        x_null = obs / null_caps.mean() if null_caps.mean() > 0 else float('inf')
        print(f"  k={k}: obs={obs:.4f}  x-chance={x_chance:.2f}  null={null_caps.mean():.4f}+-{null_caps.std(ddof=1):.5f}  excess={excess:+.4f}  z={z:+.2f}  x-null={x_null:.2f}", flush=True)
        out_per_k[k] = {
            'obs': obs, 'x_chance': x_chance,
            'null_mean': float(null_caps.mean()), 'null_sd': float(null_caps.std(ddof=1)),
            'excess': float(excess), 'z': float(z), 'x_null': float(x_null),
        }
    return out_per_k


def main():
    out = {}
    for name, info in FOREIGN_SETUPS.items():
        out[name] = process(name, info)
    out_path = RESULTS / 'e43_C3_foreign_control_v2.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= C3 foreign-training-subspace control =========")
    print("Compare 'obs' here to C3 own-model obs: MSMT k=203 was 0.782 for MSMT's ood;")
    print("Market's own-model ood at k=203 was 0.488.")
    for name in FOREIGN_SETUPS:
        print(f"\n  {name}:")
        for k in K_SWEEP:
            if k not in out[name]: continue
            r = out[name][k]
            print(f"    k={k}: obs={r['obs']:.4f}  x-chance={r['x_chance']:.2f}  x-null={r['x_null']:.2f}")


if __name__ == '__main__':
    main()
