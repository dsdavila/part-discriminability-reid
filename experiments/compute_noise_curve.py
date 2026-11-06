"""
Noise-degradation curve: reciprocity ~ mAP under structure-free degradation.

Two routes per eval set:
  (a) Isotropic Gaussian noise added to gallery features, L2-renormalised.
  (b) Random rotation of gallery features, then truncate to k dims, L2-renormalised.

Points span the undegraded mAP down past the OOD cell's mAP.

Also compute, at each σ: ceiling-corrected per-identity overlap between the
ORIGINAL (undegraded) top-50 intruder sets and the degraded top-50 sets.

Output: /data/papers/low_dimensional_reid/results/noise_curve.json
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_RECIP   = 50
N_REWIRE  = 10   # per-config null rewires; fewer than before to save time

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659},
}
# Degradation routes: start from ID cell, target OOD cell's mAP
RUNS = [
    {'name': 'MSMT',   'id_cell': 'D',  'ood_cell': "B'", 'target_mAP': 5.40,
     'noise_sigmas': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]},
    {'name': 'Market', 'id_cell': "A'", 'ood_cell': 'C',  'target_mAP': 26.66,
     'noise_sigmas': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]},
]
# Random-rotation route: pick k_target levels (fewer dims = more degradation)
RANDOM_DIM_LEVELS = [2048, 128, 32, 16, 8, 4]


def load_cell(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(X, sigma, rng):
    if sigma == 0.0: return X.copy()
    eps = rng.normal(0, sigma, X.shape).astype(np.float32)
    return l2(X + eps)


def random_proj_degrade(X, k_target, rng):
    D = X.shape[1]
    if k_target >= D:
        return X.copy()
    # random orthonormal basis (via QR of random Gaussian)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Q, _ = np.linalg.qr(R)
    return l2(X @ Q)


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


def mAP(q, q_pids, q_cids, g, g_pids, g_cids):
    sims = q @ g.T
    aps = []
    for i in range(len(q)):
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


def build_graph(topk_idx, g_pids, k):
    unique_ids = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    adj = {int(u): set() for u in unique_ids}
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


def config_rewire(adj, rng, factor=10):
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


def per_id_sets(topk_idx, g_pids, k):
    uniq = np.unique(g_pids)
    nbr = g_pids[topk_idx[:, :k]]
    im = (nbr != g_pids[:, None])
    out = {}
    for uid in uniq:
        mask = g_pids == uid
        intruders = nbr[mask][im[mask]]
        out[int(uid)] = set(int(x) for x in intruders)
    return out


def ceiling_overlap(sets1, sets2):
    vals = []
    for uid in sets1:
        a, b = sets1[uid], sets2[uid]
        if not a or not b: continue
        vals.append(len(a & b) / min(len(a), len(b)))
    return float(np.mean(vals)) if vals else 0.0


def run_cell(label, g_emb, q_emb, g_pids, g_cids, q_pids, q_cids, baseline_topk):
    """Measure mAP, density, reciprocity, overlap(vs baseline_topk) on g_emb."""
    N = len(g_emb)
    topk = topk_neighbours(g_emb, g_pids, g_cids, K_RECIP)
    m_val = mAP(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    adj = build_graph(topk, g_pids, K_RECIP)
    meas, n_edges = reciprocity_of(adj)
    rng = np.random.default_rng(42)
    null_vals = [reciprocity_of(config_rewire(adj, rng))[0] for _ in range(N_REWIRE)]
    null_mean = float(np.mean(null_vals))
    cc = (meas - null_mean) / max(1 - null_mean, 1e-12)
    max_edges = len(np.unique(g_pids)) * (len(np.unique(g_pids)) - 1)
    density = n_edges / max(max_edges, 1)
    # Overlap vs baseline (if provided)
    if baseline_topk is not None:
        base_sets = per_id_sets(baseline_topk, g_pids, K_RECIP)
        cur_sets  = per_id_sets(topk, g_pids, K_RECIP)
        overlap = ceiling_overlap(base_sets, cur_sets)
    else:
        overlap = 1.0
    print(f"  {label:<30}  mAP={m_val:5.2f}  density={density:.4f}  "
          f"recip={meas:.4f}  cc={cc:.4f}  overlap_vs_base={overlap:.4f}", flush=True)
    return {
        'mAP_pct': m_val, 'n_edges': int(n_edges), 'density': density,
        'reciprocity_measured': meas, 'reciprocity_null_mean': null_mean,
        'reciprocity_chance_corrected': cc,
        'overlap_vs_baseline': overlap,
    }, topk


def main():
    results = {}
    for run in RUNS:
        name = run['name']; c_id = run['id_cell']; c_ood = run['ood_cell']
        print(f"\n=== {name} degradation: starting from {c_id}, target OOD {c_ood} (mAP {run['target_mAP']}) ===", flush=True)
        _, _, _, G_id, g_pids, g_cids = load_cell(c_id)
        q_emb, q_pids, q_cids, _, _, _ = load_cell(c_id)
        _, _, _, G_ood, _, _ = load_cell(c_ood)
        q_emb_ood, q_pids_ood, q_cids_ood, _, _, _ = load_cell(c_ood)

        # baseline topk from undegraded G_id
        print(f"  baseline topk under {c_id}...", flush=True)
        baseline_topk = topk_neighbours(G_id, g_pids, g_cids, K_RECIP)

        # Noise route
        noise_points = []
        rng = np.random.default_rng(42)
        for sigma in run['noise_sigmas']:
            G_deg = noise_degrade(G_id, sigma, rng)
            q_deg = noise_degrade(q_emb, sigma, rng) if sigma > 0 else q_emb
            # If sigma=0 use undegraded; else degrade q with same scheme
            t0 = time.time()
            row, _ = run_cell(f"noise σ={sigma}", G_deg, q_deg, g_pids, g_cids, q_pids, q_cids, baseline_topk)
            row['sigma'] = sigma; row['route'] = 'gaussian_noise'
            print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
            noise_points.append(row)

        # Random-projection route
        rand_points = []
        rng = np.random.default_rng(43)
        for k_target in RANDOM_DIM_LEVELS:
            G_deg = random_proj_degrade(G_id, k_target, rng)
            q_deg = random_proj_degrade(q_emb, k_target, np.random.default_rng(43)) if k_target < G_id.shape[1] else q_emb
            # Use SAME random basis for query — recompute
            # Actually need to apply SAME projection; refactor:
            if k_target < G_id.shape[1]:
                D = G_id.shape[1]
                R = np.random.default_rng(43 + k_target).standard_normal((D, k_target)).astype(np.float32)
                Q_, _ = np.linalg.qr(R)
                G_deg = l2(G_id @ Q_)
                q_deg = l2(q_emb @ Q_)
            else:
                G_deg = G_id.copy(); q_deg = q_emb.copy()
            t0 = time.time()
            row, _ = run_cell(f"randproj k={k_target}", G_deg, q_deg, g_pids, g_cids, q_pids, q_cids, baseline_topk)
            row['k_target'] = k_target; row['route'] = 'random_projection'
            print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
            rand_points.append(row)

        # Record OOD cell's actual stats for anchor
        print(f"  OOD anchor {c_ood}...", flush=True)
        G_ood_e, q_ood_e = G_ood, q_emb_ood
        t0 = time.time()
        ood_row, _ = run_cell(f"OOD {c_ood} (anchor)", G_ood_e, q_ood_e, g_pids, g_cids,
                              q_pids_ood, q_cids_ood, baseline_topk)
        print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)
        results[name] = {
            'id_cell': c_id, 'ood_cell': c_ood,
            'target_mAP_ood': run['target_mAP'],
            'noise_route': noise_points,
            'random_proj_route': rand_points,
            'ood_anchor': ood_row,
        }

    out_path = RESULTS / 'noise_curve.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == '__main__':
    main()
