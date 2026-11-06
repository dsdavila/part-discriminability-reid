"""E24a/b/c with all-test convention (query+gallery), consistent with E17c reference eff_ranks."""
import sys
sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import json
from pathlib import Path
from scipy.sparse.linalg import svds

CACHE  = Path('/tmp/e12_feature_cache')
TCACHE = Path('/tmp/e18_train_cache')
OUT    = Path('/data/papers/low_dimensional_reid/results')

# All test embeddings (query first, then gallery) in each file
# market files: 16483 total (3368 Q + 13115 G)
# msmt files:   93787 total (11659 Q + 82128 G)
N_Q_MKT  = 3368
N_Q_MSMT = 11659

FILES = {
    'A': ('market_test.npz',         N_Q_MKT),
    'B': ('msmt17_test.npz',          N_Q_MSMT),
    'C': ('bot_msmt_seed42_market.npz', N_Q_MKT),
    'D': ('bot_msmt_seed42_msmt.npz',   N_Q_MSMT),
}

def load_all(cell):
    fname, nq = FILES[cell]
    d = np.load(CACHE / fname)
    emb  = d['emb'].astype(np.float64)
    pids = d['pids']
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)
    return emb, pids, nq  # all rows, L2-normalised

def load_gallery(cell):
    emb, pids, nq = load_all(cell)
    return emb[nq:], pids[nq:]

def load_train(pair):
    fname = 'bot_market_market_train.npz' if pair == 'market' else 'bot_msmt_msmt_train.npz'
    d = np.load(TCACHE / fname)
    emb = d['emb'].astype(np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)
    return emb, d['pids']

def effective_rank(X):
    if X.shape[0] < X.shape[1]:
        C = X @ X.T / X.shape[0]
    else:
        C = X.T @ X / X.shape[0]
    eigs = np.abs(np.linalg.eigvalsh(C))
    eigs = eigs[eigs > 1e-12]
    p = eigs / eigs.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

def project_out_dirs(X, dirs):
    X = X.copy()
    for v in dirs:
        v = v / (np.linalg.norm(v) + 1e-12)
        X = X - (X @ v[:, None]) * v[None, :]
    return X

print("=== E24a (all-test convention, consistent with E17c) ===\n")
results = {}
results['e24a'] = {}

for cell in ['A', 'C', 'D', 'B']:
    X, pids, nq = load_all(cell)
    N, d = X.shape

    er_raw = effective_rank(X)

    mu = X.mean(axis=0)
    mu_hat = mu / (np.linalg.norm(mu) + 1e-12)

    # Top-8 PCs via SVD (from mean-centred X, using sparse SVD)
    Xc = X - mu
    try:
        U2, s2, Vt2 = svds(Xc, k=8)
        idx = np.argsort(s2)[::-1]
        top8 = Vt2[idx]   # (8, d) descending
    except Exception as e:
        print(f"SVD error {cell}: {e}")
        top8 = None

    er_rm_mu   = effective_rank(project_out_dirs(X, [mu_hat]))
    er_rm_top2 = effective_rank(project_out_dirs(X, top8[:2])) if top8 is not None else None
    er_rm_top4 = effective_rank(project_out_dirs(X, top8[:4])) if top8 is not None else None
    er_rm_top8 = effective_rank(project_out_dirs(X, top8[:8])) if top8 is not None else None
    cos_mu_v1  = float(abs(mu_hat @ top8[0])) if top8 is not None else None

    print(f"Cell {cell}: N={N}")
    print(f"  eff_rank raw:          {er_raw:.2f}   [ref: A=261.20 B=133.30 C=208.86 D=359.91]")
    print(f"  eff_rank after -μ̂:    {er_rm_mu:.2f}  (+{er_rm_mu-er_raw:.2f})")
    if er_rm_top2: print(f"  eff_rank after -top2:  {er_rm_top2:.2f}  (+{er_rm_top2-er_raw:.2f})")
    if er_rm_top4: print(f"  eff_rank after -top4:  {er_rm_top4:.2f}  (+{er_rm_top4-er_raw:.2f})")
    if er_rm_top8: print(f"  eff_rank after -top8:  {er_rm_top8:.2f}  (+{er_rm_top8-er_raw:.2f})")
    if cos_mu_v1 is not None: print(f"  cos(μ̂, v1):           {cos_mu_v1:.4f}")
    print()

    results['e24a'][cell] = {
        'N': N, 'd': d, 'n_q': nq,
        'er_raw': er_raw, 'er_rm_mu': er_rm_mu,
        'er_rm_top2': er_rm_top2, 'er_rm_top4': er_rm_top4, 'er_rm_top8': er_rm_top8,
        'cos_mu_v1': cos_mu_v1,
    }

print("\n=== E24b (gallery only, size-weighted scatter) ===\n")
results['e24b'] = {}

for cell in ['A', 'C', 'D', 'B']:
    X, pids = load_gallery(cell)
    N = len(X)
    global_mu = X.mean(axis=0)
    unique_ids = np.unique(pids)
    n_cls = len(unique_ids)

    # Between-class scatter matrix M_B: rows = sqrt(n_c/N) * (mu_c - global_mu)
    M_B = np.zeros((n_cls, X.shape[1]))
    for i, pid in enumerate(unique_ids):
        mask = pids == pid
        n_c = mask.sum()
        mu_c = X[mask].mean(axis=0)
        M_B[i] = np.sqrt(n_c / N) * (mu_c - global_mu)

    # Within-class scatter: rows = (x - mu_c) / sqrt(N)
    M_W = np.zeros_like(X)
    for i, pid in enumerate(unique_ids):
        mask = pids == pid
        mu_c = X[mask].mean(axis=0)
        M_W[mask] = (X[mask] - mu_c) / np.sqrt(N)

    er_SB = effective_rank(M_B)
    er_SW = effective_rank(M_W)

    print(f"Cell {cell}: n_ids={n_cls}, N={N}")
    print(f"  eff_rank(S_B) = {er_SB:.2f}")
    print(f"  eff_rank(S_W) = {er_SW:.2f}")
    print(f"  ratio S_B/S_W = {er_SB/er_SW:.3f}")
    print()

    results['e24b'][cell] = {'n_ids': int(n_cls), 'N': int(N), 'er_SB': er_SB, 'er_SW': er_SW}

print("\n=== E24c: OOD variance inside training subspace ===\n")
results['e24c'] = {}
K_VALUES = [32, 64, 128, 256]

for pair, cells_for_pair in [('market', ['A', 'B']), ('msmt', ['C', 'D'])]:
    train_emb, train_pids = load_train(pair)
    T = train_emb
    T_mu = T.mean(axis=0)
    Tc = T - T_mu
    max_k = max(K_VALUES)
    try:
        _, sv, Vt = svds(Tc, k=max_k)
        idx = np.argsort(sv)[::-1]
        Vt = Vt[idx]
    except Exception as e:
        print(f"SVD error {pair}: {e}"); continue

    total_train_var = float(np.var(Tc, axis=0).sum())
    print(f"Train({pair}): N={len(T)}, total_var={total_train_var:.4f}")

    for cell in cells_for_pair:
        X, _, nq = load_all(cell)
        Xc = X - T_mu
        total_test_var = float(np.var(Xc, axis=0).sum())
        results['e24c'][cell] = {'N_test': int(len(X)), 'total_var': total_test_var}
        print(f"  Cell {cell}: N={len(X)}, total_var={total_test_var:.4f}")
        for k in K_VALUES:
            proj = Xc @ Vt[:k].T
            retained = float(np.var(proj, axis=0).sum())
            frac = retained / total_test_var
            print(f"    k={k:3d}: {frac*100:.1f}%")
            results['e24c'][cell][f'k{k}'] = frac
        print()

out_path = OUT / 'e24_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved → {out_path}")
