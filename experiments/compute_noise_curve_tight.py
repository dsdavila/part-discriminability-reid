"""
Rerun 4 matched-mAP curve points per eval set with the same null precision
as the anchors (N_REWIRE=20, factor=15), plus bootstrap CIs on cc.

Uses the σ/k values from noise_curve_v2.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_RECIP   = 50
N_REWIRE  = 20
REWIRE_FACTOR = 15
N_BOOT    = 200

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659},
}

# From noise_curve_v2.json: σ values that hit target mAPs
MSMT_SIGMAS = {30.0: 0.0352, 20.0: 0.0449, 10.0: 0.0557, 5.4: 0.0625}
MSMT_K_PROJ = {30.0: 89, 20.0: 45, 10.0: 24, 5.4: 17}
MKT_SIGMAS = {60.0: 0.0439, 45.0: 0.0522, 35.0: 0.0576, 26.66: 0.0620}
MKT_K_PROJ = {60.0: 47, 45.0: 28, 35.0: 21, 26.66: 17}


def load_cell(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(G, Q, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    q_eps = rng.normal(0, sigma, Q.shape).astype(np.float32)
    return l2(G + g_eps), l2(Q + q_eps)


def rand_proj_degrade(G, Q, k_target, seed=42):
    D = G.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Qmat, _ = np.linalg.qr(R)
    return l2(G @ Qmat), l2(Q @ Qmat)


def mAP_fast(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    sims = q_emb @ g_emb.T
    aps = []
    for i in range(len(q_emb)):
        qp, qc = q_pids[i], q_cids[i]
        valid = ~((g_pids == qp) & (g_cids == qc))
        rel = (g_pids == qp) & valid
        if rel.sum() == 0: continue
        order = np.argsort(-sims[i][valid])
        rel_v = rel[valid][order]
        if rel_v.sum() == 0: continue
        pos = np.where(rel_v)[0]
        aps.append(float(((np.arange(len(pos))+1) / (pos+1)).mean()))
    return float(np.mean(aps)) * 100


def topk_neighbours(G, g_pids, g_cids, k_max, chunk=2048):
    N = len(G)
    out = np.full((N, k_max), -1, dtype=np.int64)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            top = np.argpartition(-sims[local], k_max)[:k_max]
            out[gi] = top[np.argsort(-sims[local, top])]
    return out


def build_graph(topk_idx, g_pids, k):
    uniq = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    adj = {int(u): set() for u in uniq}
    for i, u in enumerate(g_pids):
        for jj in nbr[i][im[i]]:
            adj[int(u)].add(int(jj))
    return adj


def reciprocity_of(adj):
    edges = set()
    for i, outs in adj.items():
        for j in outs: edges.add((i, j))
    if not edges: return 0.0, 0
    recip = sum(1 for e in edges if (e[1], e[0]) in edges)
    return recip / len(edges), len(edges)


def config_rewire(adj, rng, factor=REWIRE_FACTOR):
    edges = [(i, j) for i, outs in adj.items() for j in outs]
    n = len(edges)
    adj_sets = {k: set(v) for k, v in adj.items()}
    for _ in range(factor * n):
        if n < 2: break
        a, b = rng.integers(0, n), rng.integers(0, n)
        if a == b: continue
        i1, j1 = edges[a]; i2, j2 = edges[b]
        if j2 in adj_sets[i1] or j1 in adj_sets[i2]: continue
        if j2 == i1 or j1 == i2: continue
        adj_sets[i1].remove(j1); adj_sets[i1].add(j2)
        adj_sets[i2].remove(j2); adj_sets[i2].add(j1)
        edges[a] = (i1, j2); edges[b] = (i2, j1)
    return adj_sets


def measure(G, Q, q_pids, q_cids, g_pids, g_cids, label):
    print(f"  {label}  computing...", flush=True)
    t0 = time.time()
    m_val = mAP_fast(Q, q_pids, q_cids, G, g_pids, g_cids)
    topk = topk_neighbours(G, g_pids, g_cids, K_RECIP)
    adj = build_graph(topk, g_pids, K_RECIP)
    meas, n_edges = reciprocity_of(adj)
    rng = np.random.default_rng(42)
    null_vals = [reciprocity_of(config_rewire(adj, rng))[0] for _ in range(N_REWIRE)]
    null_mean = float(np.mean(null_vals))
    null_std  = float(np.std(null_vals))
    cc = (meas - null_mean) / max(1 - null_mean, 1e-12)
    # Identity-level bootstrap of cc (null fixed)
    unique_ids = np.array(sorted(adj.keys()))
    edges = [(i, j) for i, outs in adj.items() for j in outs]
    edges_set = set(edges)
    rng_b = np.random.default_rng(42)
    cc_boots = []
    n_id = len(unique_ids)
    for _ in range(N_BOOT):
        sampled = set(int(x) for x in rng_b.choice(unique_ids, n_id, replace=True))
        sub = [e for e in edges if e[0] in sampled and e[1] in sampled]
        if not sub: continue
        rec_c = sum(1 for e in sub if (e[1], e[0]) in edges_set)
        m_b = rec_c / len(sub)
        cc_boots.append((m_b - null_mean) / max(1 - null_mean, 1e-12))
    cc_ci = (float(np.percentile(cc_boots, 2.5)), float(np.percentile(cc_boots, 97.5))) if cc_boots else (0, 0)
    density = n_edges / max(n_id * (n_id - 1), 1)
    print(f"    mAP={m_val:.2f}  density={density:.4f}  cc={cc:.4f} CI=[{cc_ci[0]:.4f},{cc_ci[1]:.4f}]  "
          f"null={null_mean:.4f}±{null_std:.4f}  t={(time.time()-t0)/60:.1f}m", flush=True)
    return {'mAP_pct': m_val, 'density': density,
            'reciprocity_measured': meas, 'reciprocity_null_mean': null_mean,
            'reciprocity_null_std': null_std,
            'reciprocity_chance_corrected': cc, 'cc_ci95': list(cc_ci)}


def main():
    out = {'MSMT': {}, 'Market': {}}

    # MSMT: degrade D at the 4 chosen σ/k values
    print("\n=== MSMT: degraded D ===", flush=True)
    q_emb_D, q_pids_D, q_cids_D, G_D, g_pids_D, g_cids_D = load_cell('D')
    for tgt, sigma in MSMT_SIGMAS.items():
        G_d, Q_d = noise_degrade(G_D, q_emb_D, sigma)
        out['MSMT'][f'noise_tgt{tgt}'] = measure(G_d, Q_d, q_pids_D, q_cids_D, g_pids_D, g_cids_D,
                                                  f"noise σ={sigma} target mAP={tgt}")
    for tgt, k_t in MSMT_K_PROJ.items():
        G_d, Q_d = rand_proj_degrade(G_D, q_emb_D, k_t)
        out['MSMT'][f'randproj_tgt{tgt}'] = measure(G_d, Q_d, q_pids_D, q_cids_D, g_pids_D, g_cids_D,
                                                     f"randproj k={k_t} target mAP={tgt}")

    # Market: degrade A' at the 4 chosen σ/k values
    print("\n=== Market: degraded A' ===", flush=True)
    q_emb_A, q_pids_A, q_cids_A, G_A, g_pids_A, g_cids_A = load_cell("A'")
    for tgt, sigma in MKT_SIGMAS.items():
        G_d, Q_d = noise_degrade(G_A, q_emb_A, sigma)
        out['Market'][f'noise_tgt{tgt}'] = measure(G_d, Q_d, q_pids_A, q_cids_A, g_pids_A, g_cids_A,
                                                    f"noise σ={sigma} target mAP={tgt}")
    for tgt, k_t in MKT_K_PROJ.items():
        G_d, Q_d = rand_proj_degrade(G_A, q_emb_A, k_t)
        out['Market'][f'randproj_tgt{tgt}'] = measure(G_d, Q_d, q_pids_A, q_cids_A, g_pids_A, g_cids_A,
                                                       f"randproj k={k_t} target mAP={tgt}")

    out_path = RESULTS / 'noise_curve_tight.json'
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
