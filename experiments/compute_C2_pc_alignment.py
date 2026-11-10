"""
PC-alignment diagnostic (rev 346): is M's top-10 eigenspace concentrated
in each arm's gallery leading PCs? Decides whether the per-arm null is
well-specified (high alignment -> row 2 stands) or mis-specified (low
alignment -> source-matched 'at chance' governs and row 2 dies).

For each cell:
  Compute M_rank = C_rank - C_pos (rank-matched negatives at ranks 1-20
  among non-same-pid), take top-10 eigenvectors by |lam|.
  Compute gallery top-10 and top-50 PCs.
  Report:
    - Principal angles between M_rank top-10 and gallery top-10
    - Energy fraction of M_rank top-10 in gallery top-10 and top-50
      = (1/10) * sum ||proj_PCs(v_M_k)||^2 for v_M_k in top-10 M eigvecs
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
RNG_SEED = 42
RANK_BAND = 20
K_EIG = 10

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)"},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)"},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)"},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)"},
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def principal_angles(B1, B2):
    s = np.linalg.svd(B1.T @ B2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def participation(B, PCs):
    """||proj_PCs(B)||^2_F / ||B||^2_F for orthonormal B (K cols)."""
    coef = B.T @ PCs
    return float((coef ** 2).sum() / B.shape[1])


def pca_basis(X, K_max):
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


def accumulate_M_rank(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """Accumulate C_pos (pos pairs) and C_rank (rank-matched neg pairs)."""
    d = q_emb.shape[1]
    sum_pos = np.zeros((d, d), dtype=np.float64); n_pos = 0
    sum_rank = np.zeros((d, d), dtype=np.float64); n_rank = 0
    for i in range(len(q_pids)):
        if i % 2000 == 0:
            print(f"    query {i}/{len(q_pids)}  (n_pos={n_pos}, n_rank={n_rank})", flush=True)
        qp = q_pids[i]; qc = q_cids[i]; q = q_emb[i]
        junk = (g_pids == qp) & (g_cids == qc)
        pos_mask = (g_pids == qp) & ~junk
        if pos_mask.any():
            pos_items = g_emb[pos_mask]
            diffs = q - pos_items
            sum_pos += diffs.T @ diffs
            n_pos += len(pos_items)
        # Rank-matched negatives
        sims = q @ g_emb.T
        sims[junk] = -np.inf
        order = np.argsort(-sims)
        is_pos = g_pids[order] == qp
        neg_mask = ~is_pos & (sims[order] > -np.inf)
        neg_idx = order[neg_mask][:RANK_BAND]
        if len(neg_idx) > 0:
            rn_items = g_emb[neg_idx]
            diffs = q - rn_items
            sum_rank += diffs.T @ diffs
            n_rank += len(neg_idx)
    return sum_rank / n_rank, sum_pos / n_pos, n_pos, n_rank


def eigvecs_top_K(M, K):
    w, V = np.linalg.eigh(M.astype(np.float64))
    order = np.argsort(-np.abs(w))
    return V[:, order[:K]], w[order[:K]]


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    print(f"  accumulating C_pos and C_rank...", flush=True)
    C_rank, C_pos, n_pos, n_rank = accumulate_M_rank(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    M = C_rank - C_pos
    V_M, _ = eigvecs_top_K(M, K_EIG)
    V_M = V_M.astype(np.float32)
    print(f"  computing gallery PCs (top-50)...", flush=True)
    PCs_50, _ = pca_basis(g_emb, 50)
    PCs_10 = PCs_50[:, :10]
    # Principal angles between M top-10 and gallery top-10
    angles_top10 = principal_angles(V_M, PCs_10)
    energy_top10 = participation(V_M, PCs_10)
    energy_top50 = participation(V_M, PCs_50)
    print(f"  M top-10 vs gallery top-10 principal angles: {np.round(angles_top10, 2).tolist()}", flush=True)
    print(f"  M top-10 vs gallery top-10 mean angle = {angles_top10.mean():.3f}  median = {np.median(angles_top10):.3f}", flush=True)
    print(f"  M top-10 energy in gallery top-10 PCs = {energy_top10:.4f}", flush=True)
    print(f"  M top-10 energy in gallery top-50 PCs = {energy_top50:.4f}", flush=True)
    return {
        'cell': tag, 'label': info['label'],
        'angles_M_top10_vs_PC_top10_deg': angles_top10.tolist(),
        'angles_mean_deg': float(angles_top10.mean()),
        'angles_median_deg': float(np.median(angles_top10)),
        'energy_M_top10_in_PC_top10': energy_top10,
        'energy_M_top10_in_PC_top50': energy_top50,
        'n_pos_pairs': n_pos, 'n_rank_pairs': n_rank,
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)
    out_path = RESULTS / 'e43_C2_pc_alignment.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= PC-alignment diagnostic =========")
    print(f"{'cell':9s}  {'mean ang':>10s}  {'median':>10s}  {'E in top-10':>12s}  {'E in top-50':>12s}")
    for tag in CELLS:
        r = out[tag]
        print(f"{tag:9s}  {r['angles_mean_deg']:9.3f}  {r['angles_median_deg']:9.3f}  {r['energy_M_top10_in_PC_top10']:11.4f}  {r['energy_M_top10_in_PC_top50']:11.4f}")


if __name__ == '__main__':
    main()
