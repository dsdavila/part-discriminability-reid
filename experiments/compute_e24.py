"""
E24: Daniel's intersection hypothesis — three analyses on cached features.
Uses e12_feature_cache (test embeddings) and e18_train_cache (train embeddings).
All four cells: A (BoT-Market→Market), B (BoT-Market→MSMT17),
                C (BoT-MSMT17→Market), D (BoT-MSMT17→MSMT17)
"""
import sys
sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import json
from pathlib import Path
from scipy.sparse.linalg import svds

CACHE  = Path('/tmp/e12_feature_cache')
TCACHE = Path('/tmp/e18_train_cache')
OUT    = Path('/data/papers/low_dimensional_reid/results')

N_Q_MKT  = 3368
N_Q_MSMT = 11659   # actual size in this cache

def load_cell(cell):
    """Load L2-normalized test embeddings (gallery + query) for a cell."""
    if cell == 'A':
        d = np.load(CACHE / 'market_test.npz')
        n_q = N_Q_MKT
    elif cell == 'C':
        d = np.load(CACHE / 'bot_msmt_seed42_market.npz')
        n_q = N_Q_MKT
    elif cell == 'D':
        d = np.load(CACHE / 'bot_msmt_seed42_msmt.npz')
        n_q = N_Q_MSMT
    elif cell == 'B':
        d = np.load(CACHE / 'msmt17_test.npz')
        n_q = N_Q_MSMT
    emb  = d['emb'].astype(np.float32)
    pids = d['pids']
    cams = d.get('cams', d.get('cam_ids', np.zeros(len(pids), dtype=int)))
    # gallery only (skip query rows, which come first)
    gal_emb  = emb[n_q:]
    gal_pids = pids[n_q:]
    qry_emb  = emb[:n_q]
    qry_pids = pids[:n_q]
    return gal_emb, gal_pids, qry_emb, qry_pids

def load_train(pair):
    """Load training embeddings for a model-dataset pair."""
    # pair: 'market' = BoT-Market model, Market train
    #       'msmt'   = BoT-MSMT17 model, MSMT17 train
    if pair == 'market':
        d = np.load(TCACHE / 'bot_market_market_train.npz')
    else:
        d = np.load(TCACHE / 'bot_msmt_msmt_train.npz')
    return d['emb'].astype(np.float32), d['pids']

def effective_rank(X):
    """
    Effective rank of X (N × d) via singular values.
    eff_rank = exp(-sum(p_i log p_i)) where p_i = σ_i^2 / sum(σ_j^2).
    """
    if X.shape[0] < X.shape[1]:
        # fewer samples than dims: work in sample space
        C = (X.astype(np.float64) @ X.astype(np.float64).T) / X.shape[0]
    else:
        C = (X.astype(np.float64).T @ X.astype(np.float64)) / X.shape[0]
    eigs = np.linalg.eigvalsh(C)
    eigs = np.abs(eigs)
    eigs = eigs[eigs > 1e-12]
    p = eigs / eigs.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

def project_out(X, directions):
    """Remove top-k directions from X (each row of directions is a unit vector)."""
    X = X.copy()
    for v in directions:
        v = v / (np.linalg.norm(v) + 1e-12)
        X = X - (X @ v[:, None]) * v[None, :]
    return X

print("=== E24: Daniel's intersection hypothesis ===\n")

results = {}

# ── E24a: Mean removal and effective rank ─────────────────────────────────────
print("--- E24a: Effective rank after removing top directions ---\n")
results['e24a'] = {}

for cell in ['A', 'C', 'D', 'B']:
    gal_emb, gal_pids, qry_emb, qry_pids = load_cell(cell)
    # Use gallery embeddings (already L2-normalized from cache)
    X = gal_emb.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)  # ensure L2-normalized

    er_raw = effective_rank(X)

    # Compute top-8 singular directions via SVD of centred X
    mu = X.mean(axis=0)
    mu_hat = mu / (np.linalg.norm(mu) + 1e-12)

    # Top 8 PCs of X
    try:
        U, s, Vt = svds(X.T, k=8)  # Vt: (8, N) — these are left singular vectors of X.T
        top8 = Vt  # shape (8, N) — no, wrong convention for what we want
        # Actually for removing directions from embedding space:
        # We want top-8 RIGHT singular vectors of X (shape d)
        # svds(X, k=8): U(N,8), s(8,), Vt(8,d)
        U2, s2, Vt2 = svds(X, k=8)
        # Vt2[i] is the i-th top right singular vector (in d-space)
        # Sort by singular value descending
        idx = np.argsort(s2)[::-1]
        top8_dirs = Vt2[idx]  # (8, d), descending
    except Exception as e:
        print(f"  SVD error Cell {cell}: {e}")
        top8_dirs = None

    er_rm_mu   = effective_rank(project_out(X, [mu_hat]))
    er_rm_top2 = effective_rank(project_out(X, top8_dirs[:2])) if top8_dirs is not None else None
    er_rm_top4 = effective_rank(project_out(X, top8_dirs[:4])) if top8_dirs is not None else None
    er_rm_top8 = effective_rank(project_out(X, top8_dirs[:8])) if top8_dirs is not None else None

    # cos(mu_hat, v1) — alignment of mean with top PC
    cos_mu_v1 = float(abs(mu_hat @ top8_dirs[0])) if top8_dirs is not None else None

    print(f"Cell {cell} (N={len(X)}, d={X.shape[1]}):")
    print(f"  eff_rank raw:          {er_raw:.2f}")
    print(f"  eff_rank after -μ̂:    {er_rm_mu:.2f}  (delta={er_rm_mu-er_raw:+.2f})")
    if er_rm_top2: print(f"  eff_rank after -top2:  {er_rm_top2:.2f}  (delta={er_rm_top2-er_raw:+.2f})")
    if er_rm_top4: print(f"  eff_rank after -top4:  {er_rm_top4:.2f}  (delta={er_rm_top4-er_raw:+.2f})")
    if er_rm_top8: print(f"  eff_rank after -top8:  {er_rm_top8:.2f}  (delta={er_rm_top8-er_raw:+.2f})")
    if cos_mu_v1 is not None: print(f"  cos(μ̂, v1):           {cos_mu_v1:.4f}")
    print()

    results['e24a'][cell] = {
        'N': int(len(X)), 'd': int(X.shape[1]),
        'er_raw': er_raw, 'er_rm_mu': er_rm_mu,
        'er_rm_top2': er_rm_top2, 'er_rm_top4': er_rm_top4, 'er_rm_top8': er_rm_top8,
        'cos_mu_v1': cos_mu_v1,
    }


# ── E24b: Between/within scatter effective rank ───────────────────────────────
print("--- E24b: eff_rank(S_B) and eff_rank(S_W) ---\n")
results['e24b'] = {}

for cell in ['A', 'C', 'D', 'B']:
    gal_emb, gal_pids, _, _ = load_cell(cell)
    X = gal_emb.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)

    N = len(X)
    global_mu = X.mean(axis=0)
    unique_pids = np.unique(gal_pids)
    n_cls = len(unique_pids)

    # S_B: between-class scatter (size-weighted)
    # = sum_c (n_c / N) * (mu_c - global_mu) @ (mu_c - global_mu).T
    # Equivalent: build matrix M of shape (n_cls, d) with rows = sqrt(n_c/N) * (mu_c - global_mu)
    M_B = np.zeros((n_cls, X.shape[1]), dtype=np.float64)
    for i, pid in enumerate(unique_pids):
        mask = gal_pids == pid
        n_c = mask.sum()
        mu_c = X[mask].mean(axis=0)
        M_B[i] = np.sqrt(n_c / N) * (mu_c - global_mu)

    # S_W: within-class scatter
    # Build matrix M of shape (N, d) with rows = (x - mu_c) / sqrt(N)
    M_W = np.zeros_like(X)
    for i, pid in enumerate(unique_pids):
        mask = gal_pids == pid
        mu_c = X[mask].mean(axis=0)
        M_W[mask] = (X[mask] - mu_c) / np.sqrt(N)

    er_SB = effective_rank(M_B)
    er_SW = effective_rank(M_W)

    print(f"Cell {cell}: n_ids={n_cls}, N={N}")
    print(f"  eff_rank(S_B) = {er_SB:.2f}")
    print(f"  eff_rank(S_W) = {er_SW:.2f}")
    print(f"  ratio S_B/S_W = {er_SB/er_SW:.3f}")
    print()

    results['e24b'][cell] = {'n_ids': int(n_cls), 'N': int(N),
                              'er_SB': er_SB, 'er_SW': er_SW}


# ── E24c: Variance retained in training subspace ──────────────────────────────
print("--- E24c: OOD variance inside training-domain subspace ---\n")
results['e24c'] = {}

K_VALUES = [32, 64, 128, 256]

for pair, cells_for_pair in [('market', ['A', 'B']), ('msmt', ['C', 'D'])]:
    print(f"Train PCs from: BoT-{pair.upper()} model, {pair.upper()} training set")
    train_emb, train_pids = load_train(pair)
    T = train_emb.astype(np.float64)
    norms = np.linalg.norm(T, axis=1, keepdims=True)
    T = T / np.maximum(norms, 1e-12)
    T_mu = T.mean(axis=0)
    T_centered = T - T_mu

    # Compute top-256 PCs via SVD
    max_k = max(K_VALUES)
    try:
        _, _, Vt_train = svds(T_centered, k=max_k)
        idx = np.argsort(np.linalg.norm(T_centered @ Vt_train.T, axis=0))[::-1]
        Vt_train = Vt_train[idx]  # sorted descending — actually svds doesn't sort, sort by variance
    except Exception as e:
        print(f"  SVD error: {e}")
        continue

    # Sort PCs by variance explained in training set
    variances = np.array([(T_centered @ Vt_train[i]) .var() for i in range(max_k)])
    sort_idx = np.argsort(variances)[::-1]
    Vt_train = Vt_train[sort_idx]

    total_train_var = np.var(T_centered, axis=0).sum()
    print(f"  Train N={len(T)}, total variance={total_train_var:.4f}")

    for cell in cells_for_pair:
        gal_emb, gal_pids, _, _ = load_cell(cell)
        X = gal_emb.astype(np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)
        X_centered = X - T_mu  # centre by TRAINING mean

        total_test_var = np.var(X_centered, axis=0).sum()
        results['e24c'][cell] = {'N_test': int(len(X)), 'total_var': float(total_test_var)}

        print(f"  Cell {cell} (N={len(X)}, total_var={total_test_var:.4f}):")
        for k in K_VALUES:
            # Project onto top-k training PCs
            proj = X_centered @ Vt_train[:k].T  # (N_test, k)
            retained_var = np.var(proj, axis=0).sum()
            frac = retained_var / total_test_var
            print(f"    k={k:3d}: retained {frac*100:.1f}% of test variance")
            results['e24c'][cell][f'k{k}'] = float(frac)
        print()
    print()

out_path = OUT / 'e24_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved → {out_path}")
