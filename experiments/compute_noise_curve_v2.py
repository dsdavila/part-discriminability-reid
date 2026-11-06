"""
Noise-degradation curve v2 — binary search for target mAPs, then measure stats.

Two routes per eval set:
  (a) Isotropic Gaussian noise
  (b) Random rotation + truncation

For each route, pick 4 target mAPs spanning the ID→OOD range, binary-search σ,
then measure mAP + k=50 reciprocity (chance-corrected) + overlap vs undegraded.

Also measure the untouched σ=0 anchor.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_RECIP   = 50
N_REWIRE  = 5
REWIRE_FACTOR = 3

CELLS = {
    "A'": {'file': 'market_test_noREA.npz',    'n_q': 3368},
    'C':  {'file': 'bot_msmt_seed42_market.npz','n_q': 3368},
    'D':  {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659},
    "B'": {'file': 'msmt17_test_noREA.npz',    'n_q': 11659},
}
# Each run: ID cell is the baseline; targets are mAPs to hit via degradation
RUNS = [
    {'name': 'MSMT',   'id_cell': 'D',  'ood_cell': "B'", 'target_mAPs': [30.0, 20.0, 10.0, 5.40]},
    {'name': 'Market', 'id_cell': "A'", 'ood_cell': 'C',  'target_mAPs': [60.0, 45.0, 35.0, 26.66]},
]


def load_cell(cell):
    cfg = CELLS[cell]
    d = np.load(CACHE_DIR / cfg['file']); n_q = cfg['n_q']
    emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


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


def noise_degrade(G, Q, sigma, seed=42):
    rng = np.random.default_rng(seed)
    if sigma == 0.0:
        return G.copy(), Q.copy()
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    q_eps = rng.normal(0, sigma, Q.shape).astype(np.float32)
    return l2(G + g_eps), l2(Q + q_eps)


def rand_proj_degrade(G, Q, k_target, seed=42):
    D = G.shape[1]
    if k_target >= D:
        return G.copy(), Q.copy()
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Qmat, _ = np.linalg.qr(R)
    return l2(G @ Qmat), l2(Q @ Qmat)


def binary_search_sigma(G, Q, q_pids, q_cids, g_pids, g_cids, target_mAP,
                         sigma_lo=0.0, sigma_hi=0.5, max_iter=10, seed=42):
    """Returns σ such that mAP(noise-degraded G,Q) ≈ target."""
    for _ in range(max_iter):
        s = (sigma_lo + sigma_hi) / 2
        G_d, Q_d = noise_degrade(G, Q, s, seed)
        m = mAP_fast(Q_d, q_pids, q_cids, G_d, g_pids, g_cids)
        if abs(m - target_mAP) < 0.5: return s, m
        if m > target_mAP: sigma_lo = s
        else:              sigma_hi = s
    return s, m


def binary_search_k(G, Q, q_pids, q_cids, g_pids, g_cids, target_mAP, max_iter=12, seed=42):
    """Returns k such that mAP(rand-proj) ≈ target."""
    D = G.shape[1]
    k_lo, k_hi = 2, D
    for _ in range(max_iter):
        k_mid = (k_lo + k_hi) // 2
        if k_mid <= 1 or k_mid >= D: break
        G_d, Q_d = rand_proj_degrade(G, Q, k_mid, seed)
        m = mAP_fast(Q_d, q_pids, q_cids, G_d, g_pids, g_cids)
        if abs(m - target_mAP) < 0.5: return k_mid, m
        if m > target_mAP: k_hi = k_mid
        else:              k_lo = k_mid
    return k_mid, m


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


def measure_point(G, Q, q_pids, q_cids, g_pids, g_cids, baseline_topk):
    """Compute mAP, density, reciprocity (point + null), overlap vs baseline."""
    m_val = mAP_fast(Q, q_pids, q_cids, G, g_pids, g_cids)
    topk = topk_neighbours(G, g_pids, g_cids, K_RECIP)
    adj = build_graph(topk, g_pids, K_RECIP)
    meas, n_edges = reciprocity_of(adj)
    rng = np.random.default_rng(42)
    null_vals = [reciprocity_of(config_rewire(adj, rng))[0] for _ in range(N_REWIRE)]
    null_mean = float(np.mean(null_vals))
    cc = (meas - null_mean) / max(1 - null_mean, 1e-12)
    n_ids = len(np.unique(g_pids))
    density = n_edges / max(n_ids * (n_ids - 1), 1)
    base_sets = per_id_sets(baseline_topk, g_pids, K_RECIP)
    cur_sets  = per_id_sets(topk, g_pids, K_RECIP)
    overlap = ceiling_overlap(base_sets, cur_sets)
    return {
        'mAP_pct': m_val, 'n_edges': int(n_edges), 'density': density,
        'reciprocity_measured': meas, 'reciprocity_null_mean': null_mean,
        'reciprocity_chance_corrected': cc,
        'overlap_vs_baseline': overlap,
    }


def main():
    results = {}
    for run in RUNS:
        name = run['name']; c_id = run['id_cell']; c_ood = run['ood_cell']
        print(f"\n=== {name} degradation: ID cell {c_id} → OOD target {c_ood} ===", flush=True)
        q_emb, q_pids, q_cids, G_id, g_pids, g_cids = load_cell(c_id)
        q_emb_ood, q_pids_ood, q_cids_ood, G_ood, _, _ = load_cell(c_ood)

        # Baseline topk and anchor
        print(f"  baseline (undegraded) topk + stats on {c_id}...", flush=True)
        baseline_topk = topk_neighbours(G_id, g_pids, g_cids, K_RECIP)
        t0 = time.time()
        base_row = measure_point(G_id, q_emb, q_pids, q_cids, g_pids, g_cids, baseline_topk)
        base_row['sigma'] = 0.0; base_row['route'] = 'baseline'
        print(f"    base: mAP={base_row['mAP_pct']:.2f} recip={base_row['reciprocity_measured']:.4f} cc={base_row['reciprocity_chance_corrected']:.4f}  t={(time.time()-t0)/60:.1f}m", flush=True)

        # OOD anchor
        print(f"  OOD anchor {c_ood} stats...", flush=True)
        t0 = time.time()
        ood_row = measure_point(G_ood, q_emb_ood, q_pids_ood, q_cids_ood, g_pids, g_cids, baseline_topk)
        print(f"    ood: mAP={ood_row['mAP_pct']:.2f} recip={ood_row['reciprocity_measured']:.4f} cc={ood_row['reciprocity_chance_corrected']:.4f}  t={(time.time()-t0)/60:.1f}m", flush=True)

        # Noise route: binary search to each target
        noise_pts = [base_row]
        for tgt in run['target_mAPs']:
            print(f"  [noise] searching σ for target mAP={tgt}...", flush=True)
            t0 = time.time()
            s, m = binary_search_sigma(G_id, q_emb, q_pids, q_cids, g_pids, g_cids, tgt)
            print(f"    σ={s:.4f}  achieved mAP={m:.2f}", flush=True)
            G_d, Q_d = noise_degrade(G_id, q_emb, s)
            t1 = time.time()
            row = measure_point(G_d, Q_d, q_pids, q_cids, g_pids, g_cids, baseline_topk)
            row['sigma'] = float(s); row['target_mAP'] = tgt; row['route'] = 'gaussian_noise'
            print(f"    stats: mAP={row['mAP_pct']:.2f} density={row['density']:.4f} recip={row['reciprocity_measured']:.4f} cc={row['reciprocity_chance_corrected']:.4f} overlap={row['overlap_vs_baseline']:.4f}  t={(time.time()-t1)/60:.1f}m (search {(t1-t0)/60:.1f}m)", flush=True)
            noise_pts.append(row)

        # Random-projection route: binary search for k
        rand_pts = [base_row]  # share base
        for tgt in run['target_mAPs']:
            print(f"  [randproj] searching k for target mAP={tgt}...", flush=True)
            t0 = time.time()
            k, m = binary_search_k(G_id, q_emb, q_pids, q_cids, g_pids, g_cids, tgt)
            print(f"    k={k}  achieved mAP={m:.2f}", flush=True)
            G_d, Q_d = rand_proj_degrade(G_id, q_emb, k)
            t1 = time.time()
            row = measure_point(G_d, Q_d, q_pids, q_cids, g_pids, g_cids, baseline_topk)
            row['k_target'] = int(k); row['target_mAP'] = tgt; row['route'] = 'random_projection'
            print(f"    stats: mAP={row['mAP_pct']:.2f} density={row['density']:.4f} recip={row['reciprocity_measured']:.4f} cc={row['reciprocity_chance_corrected']:.4f} overlap={row['overlap_vs_baseline']:.4f}  t={(time.time()-t1)/60:.1f}m (search {(t1-t0)/60:.1f}m)", flush=True)
            rand_pts.append(row)

        results[name] = {
            'id_cell': c_id, 'ood_cell': c_ood,
            'target_mAPs': run['target_mAPs'],
            'baseline': base_row,
            'ood_anchor': ood_row,
            'noise_route': noise_pts,
            'random_proj_route': rand_pts,
        }
    out_path = RESULTS / 'noise_curve_v2.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
