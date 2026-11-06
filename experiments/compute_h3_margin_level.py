"""
Follow-up to H3: pooled margin, centroid angle, and spread-sum per cell
with bootstrap CIs, plus cross-cell differences.

Addresses the primary framing from Channel rev 239:
  margin = centroid_angle / (spread_i + spread_j)
  Does real transfer degrade margin as sharply as matched-accuracy noise?
  Report numerator and denominator separately.
"""
import json, time
import numpy as np
from pathlib import Path
from collections import defaultdict

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K_CONF    = 10
NOISE_SIGMA_D = 0.0625


def load_cell(cell):
    files = {
        "B'": ('msmt17_test_noREA.npz', 11659),
        'D':  ('bot_msmt_seed42_msmt.npz', 11659),
    }
    fn, n_q = files[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
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


def per_id_centroid_spread(X, pids):
    ids = np.unique(pids)
    mu = np.zeros((len(ids), X.shape[1]), np.float32)
    spread = np.zeros(len(ids), np.float32)
    for k, pid in enumerate(ids):
        mask = pids == pid
        xi = X[mask]
        mu_k = xi.mean(axis=0)
        mu_k /= max(np.linalg.norm(mu_k), 1e-12)
        mu[k] = mu_k
        cos_x = np.clip(xi @ mu_k, -1.0, 1.0)
        spread[k] = np.arccos(cos_x).mean()
    return ids, mu, spread


def topk_confuser_pairs(Q, qp, qc, G, gp, gc, k=10):
    """Return set of (min(i,j), max(i,j)) confuser pairs."""
    sim = Q @ G.T
    edges = set()
    for qi in range(len(qp)):
        qid, qcid = qp[qi], qc[qi]
        valid = ~((gp == qid) & (gc == qcid))
        s = sim[qi].copy()
        s[~valid] = -np.inf
        top_idx = np.argpartition(-s, k)[:k]
        top_idx = top_idx[np.argsort(-s[top_idx])]
        for gi in top_idx:
            j = gp[gi]
            if j != qid:
                edges.add((int(qid), int(j)))
    pairs = set()
    for (i, j) in edges:
        pairs.add((min(i, j), max(i, j)))
    return pairs


def pair_arrays(pairs, id_to_idx, mu, spread):
    ang_list, marg_list, sum_list = [], [], []
    for (i, j) in pairs:
        if i not in id_to_idx or j not in id_to_idx:
            continue
        ki, kj = id_to_idx[i], id_to_idx[j]
        cos_ij = float(np.clip(mu[ki] @ mu[kj], -1.0, 1.0))
        ang = float(np.arccos(cos_ij))
        si, sj = float(spread[ki]), float(spread[kj])
        s_sum = si + sj
        marg = ang / max(s_sum, 1e-12)
        ang_list.append(ang)
        marg_list.append(marg)
        sum_list.append(s_sum)
    return np.array(ang_list), np.array(marg_list), np.array(sum_list)


def bootstrap_mean(vals, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = vals[idx].mean()
    return {'mean': float(vals.mean()),
            'CI95': [float(np.percentile(means, 2.5)),
                     float(np.percentile(means, 97.5))]}


def bootstrap_diff(a, b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    diffs = np.zeros(n_boot)
    for i in range(n_boot):
        ai = rng.integers(0, na, na)
        bi = rng.integers(0, nb, nb)
        diffs[i] = a[ai].mean() - b[bi].mean()
    return {'mean': float(a.mean() - b.mean()),
            'CI95': [float(np.percentile(diffs, 2.5)),
                     float(np.percentile(diffs, 97.5))]}


def run_cell(cell_name, Q, qp, qc, G, gp, gc):
    print(f"\n=== {cell_name} ===", flush=True)
    t0 = time.time()
    ids, mu, spread = per_id_centroid_spread(G, gp)
    id_to_idx = {int(p): k for k, p in enumerate(ids)}
    print(f"  n_ids={len(ids)}  t={(time.time()-t0)/60:.1f}m", flush=True)

    t1 = time.time()
    pairs = topk_confuser_pairs(Q, qp, qc, G, gp, gc, k=K_CONF)
    print(f"  pairs={len(pairs)}  t={(time.time()-t1)/60:.1f}m", flush=True)

    ang, marg, s_sum = pair_arrays(pairs, id_to_idx, mu, spread)
    return {'n_pairs': len(pairs),
            'ang_arr': ang, 'marg_arr': marg, 'sum_arr': s_sum,
            'ang':   bootstrap_mean(ang),
            'margin': bootstrap_mean(marg),
            'sum':   bootstrap_mean(s_sum)}


def main():
    cells = {}

    print("\n########## D (in-domain baseline) ##########", flush=True)
    q, qp, qc, G, gp, gc = load_cell('D')
    cells['D'] = run_cell('D', q, qp, qc, G, gp, gc)

    print("\n########## noise-D at σ=0.0625 (matched mAP to B') ##########", flush=True)
    Gn, Qn = noise_degrade(G, q, NOISE_SIGMA_D)
    cells['noise_D'] = run_cell('noise-D', Qn, qp, qc, Gn, gp, gc)
    del q, qp, qc, G, gp, gc, Gn, Qn

    print("\n########## B' (real transfer, Market→MSMT) ##########", flush=True)
    q, qp, qc, G, gp, gc = load_cell("B'")
    cells['B_prime'] = run_cell("B'", q, qp, qc, G, gp, gc)

    # Cross-cell differences
    print("\n=== Cross-cell differences (bootstrap n=2000) ===", flush=True)
    diffs = {}
    for a, b in [('B_prime', 'noise_D'), ('D', 'B_prime'), ('D', 'noise_D')]:
        diffs[f"{a}_minus_{b}"] = {
            'margin': bootstrap_diff(cells[a]['marg_arr'], cells[b]['marg_arr']),
            'ang':    bootstrap_diff(cells[a]['ang_arr'],  cells[b]['ang_arr']),
            'sum':    bootstrap_diff(cells[a]['sum_arr'],  cells[b]['sum_arr']),
        }

    # Report
    print("\n=== Pooled per-cell (mean ± 95% CI) ===")
    print(f"{'cell':<12} {'n_pairs':>8} {'margin':>22} {'angle':>22} {'spread_sum':>22}")
    for name in ('D', 'B_prime', 'noise_D'):
        c = cells[name]
        print(f"{name:<12} {c['n_pairs']:>8} "
              f"{c['margin']['mean']:.4f} [{c['margin']['CI95'][0]:.4f},{c['margin']['CI95'][1]:.4f}]  "
              f"{c['ang']['mean']:.4f} [{c['ang']['CI95'][0]:.4f},{c['ang']['CI95'][1]:.4f}]  "
              f"{c['sum']['mean']:.4f} [{c['sum']['CI95'][0]:.4f},{c['sum']['CI95'][1]:.4f}]")

    print("\n=== Cross-cell differences ===")
    for key, d in diffs.items():
        print(f"\n{key}:")
        for q in ('margin', 'ang', 'sum'):
            v = d[q]
            print(f"  Δ{q:<8} = {v['mean']:+.4f}  CI95 [{v['CI95'][0]:+.4f}, {v['CI95'][1]:+.4f}]")

    # Save (drop arrays from JSON)
    out = {name: {k: v for k, v in c.items() if not k.endswith('_arr')}
           for name, c in cells.items()}
    out['diffs'] = diffs
    with open(RESULTS / 'h3_margin_level.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {RESULTS / 'h3_margin_level.json'}")


if __name__ == '__main__':
    main()
