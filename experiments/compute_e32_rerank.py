"""
E32 — k-reciprocal re-ranking test on real transfer vs matched-accuracy noise.

Apply torchreid's standard k-reciprocal re-ranking (Zhong et al. CVPR 2017,
default params k1=20, k2=6, λ=0.3) to:
  - B' (Market→MSMT, 5.40%) and noise-degraded-D at σ=0.0625 (matched 5.40%)
  - C (MSMT→Market, 26.66%) and noise-degraded-A' at σ=0.062 (matched 26.66%)

Report mAP and Rank-1 before and after rerank. Pre-registered prediction:
rerank gain on noise-degraded condition >> rerank gain on real transfer at
the same starting mAP, because real transfer's reciprocity has collapsed.

Do not tune parameters.
"""
import json, time
import numpy as np
from pathlib import Path

import sys; sys.path.insert(0, '/workspace/reid-tk')
from torchreid.reid.metrics import evaluate_rank
from torchreid.reid.utils.rerank import re_ranking

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K1, K2, LAM = 20, 6, 0.3   # default params per Zhong CVPR 2017


def load_cell(cell):
    files = {
        "A'": ('market_test_noREA.npz', 3368),
        'C':  ('bot_msmt_seed42_market.npz', 3368),
        'D':  ('bot_msmt_seed42_msmt.npz', 11659),
        "B'": ('msmt17_test_noREA.npz', 11659),
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


def map_r1(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    distmat = -(q_emb @ g_emb.T)
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_cids, g_cids, max_rank=10)
    return float(mAP * 100), float(cmc[0] * 100)


def map_r1_rerank(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    print(f"    computing q_g...", flush=True)
    q_g = (1.0 - q_emb @ g_emb.T).astype(np.float32)
    print(f"    computing q_q...", flush=True)
    q_q = (1.0 - q_emb @ q_emb.T).astype(np.float32)
    print(f"    computing g_g ({len(g_emb)}×{len(g_emb)})...", flush=True)
    g_g = (1.0 - g_emb @ g_emb.T).astype(np.float32)
    print(f"    re-ranking (k1={K1}, k2={K2}, λ={LAM})...", flush=True)
    final_dist = re_ranking(q_g, q_q, g_g, k1=K1, k2=K2, lambda_value=LAM)
    print(f"    evaluate...", flush=True)
    cmc, mAP = evaluate_rank(final_dist, q_pids, g_pids, q_cids, g_cids, max_rank=10)
    # Free memory
    del q_g, q_q, g_g, final_dist
    return float(mAP * 100), float(cmc[0] * 100)


def evaluate_condition(label, q, qp, qc, g, gp, gc):
    print(f"\n  --- {label} ---", flush=True)
    t0 = time.time()
    m_b, r_b = map_r1(q, qp, qc, g, gp, gc)
    print(f"    baseline:  mAP={m_b:.2f}%  R-1={r_b:.2f}%  t={(time.time()-t0)/60:.1f}m", flush=True)
    t1 = time.time()
    m_r, r_r = map_r1_rerank(q, qp, qc, g, gp, gc)
    print(f"    reranked:  mAP={m_r:.2f}%  R-1={r_r:.2f}%  t={(time.time()-t1)/60:.1f}m", flush=True)
    return {'label': label, 'mAP_baseline': m_b, 'R1_baseline': r_b,
            'mAP_rerank': m_r, 'R1_rerank': r_r,
            'delta_mAP': m_r - m_b, 'delta_R1': r_r - r_b}


def main():
    results = {}

    # --- Market: C vs noise-A' at σ=0.062 ---
    print("\n=== MARKET: C (MSMT→Market) vs noise-A' at matched 26.66% mAP ===", flush=True)
    q_A, qp_A, qc_A, G_A, gp_A, gc_A = load_cell("A'")
    q_C, qp_C, qc_C, G_C, gp_C, gc_C = load_cell('C')
    results['Market'] = {}
    results['Market']['C_real'] = evaluate_condition(
        'Market C (real transfer MSMT→Market)', q_C, qp_C, qc_C, G_C, gp_C, gc_C)
    print("\n  Noise-degrade A' at σ=0.062...", flush=True)
    G_A_noise, Q_A_noise = noise_degrade(G_A, q_A, 0.062)
    results['Market']['noise_A'] = evaluate_condition(
        "Market noise-A' at σ=0.062", Q_A_noise, qp_A, qc_A, G_A_noise, gp_A, gc_A)

    # --- MSMT: B' vs noise-D at σ=0.0625 ---
    print("\n=== MSMT: B' (Market→MSMT) vs noise-D at matched 5.40% mAP ===", flush=True)
    q_D, qp_D, qc_D, G_D, gp_D, gc_D = load_cell('D')
    q_B, qp_B, qc_B, G_B, gp_B, gc_B = load_cell("B'")
    results['MSMT'] = {}
    results['MSMT']['B_real'] = evaluate_condition(
        "MSMT B' (real transfer Market→MSMT)", q_B, qp_B, qc_B, G_B, gp_B, gc_B)
    print("\n  Noise-degrade D at σ=0.0625...", flush=True)
    G_D_noise, Q_D_noise = noise_degrade(G_D, q_D, 0.0625)
    results['MSMT']['noise_D'] = evaluate_condition(
        'MSMT noise-D at σ=0.0625', Q_D_noise, qp_D, qc_D, G_D_noise, gp_D, gc_D)

    # Summary
    print("\n\n=== SUMMARY ===")
    print(f"{'condition':<40} {'base mAP':>9} {'rr mAP':>9} {'ΔmAP':>9} {'base R-1':>9} {'rr R-1':>9} {'ΔR-1':>9}")
    for eval_set, conditions in results.items():
        for key, row in conditions.items():
            print(f"{row['label']:<40} {row['mAP_baseline']:>9.2f} {row['mAP_rerank']:>9.2f} {row['delta_mAP']:>+9.2f} {row['R1_baseline']:>9.2f} {row['R1_rerank']:>9.2f} {row['delta_R1']:>+9.2f}")

    results['params'] = {'k1': K1, 'k2': K2, 'lambda': LAM}
    out_path = RESULTS / 'e32_rerank.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
