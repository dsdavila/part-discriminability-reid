"""
E33 — k-reciprocal re-ranking on randproj-degraded features at matched mAP.

Tests whether the E32 ratio (rerank gain noise >> rerank gain real transfer) is
driven by reciprocity (reciprocity-account prediction) or by i.i.d. denoising
(denoising-account prediction).

Randproj is systematic, correlated corruption with reciprocity nearly equal to
isotropic noise (cc 0.7746 vs 0.7512 on MSMT at matched mAP).

Reciprocity account:   randproj gains ≈ noise gains ≫ real transfer gains
Denoising account:     randproj gains ≪ noise gains, near real transfer gains
"""
import json, time
import numpy as np
from pathlib import Path
import sys; sys.path.insert(0, '/workspace/reid-tk')
from torchreid.reid.metrics import evaluate_rank
from torchreid.reid.utils.rerank import re_ranking

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
K1, K2, LAM = 20, 6, 0.3
MSMT_K_RAND = 17   # from noise_curve_v2: randproj to mAP ≈ 5.4%
MKT_K_RAND  = 17   # from noise_curve_v2: randproj to mAP ≈ 26.66%


def load_cell(cell):
    files = {
        "A'": ('market_test_noREA.npz', 3368),
        'D':  ('bot_msmt_seed42_msmt.npz', 11659),
    }
    fn, n_q = files[cell]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return (emb[:n_q], d['pids'][:n_q], d['cids'][:n_q],
            emb[n_q:], d['pids'][n_q:], d['cids'][n_q:])


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def rand_proj_degrade(G, Q, k_target, seed=42):
    D = G.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Qmat, _ = np.linalg.qr(R)
    return l2(G @ Qmat), l2(Q @ Qmat)


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

    # Market: randproj-A' at k=17 → mAP ≈ 26.66%
    print("\n=== MARKET: randproj-A' at k=17 (matched 26.66% mAP) ===", flush=True)
    q_A, qp_A, qc_A, G_A, gp_A, gc_A = load_cell("A'")
    print(f"  randproj-degrade A' at k={MKT_K_RAND}...", flush=True)
    G_A_rp, Q_A_rp = rand_proj_degrade(G_A, q_A, MKT_K_RAND)
    results['Market_randproj'] = evaluate_condition(
        "Market randproj-A' k=17", Q_A_rp, qp_A, qc_A, G_A_rp, gp_A, gc_A)

    # MSMT: randproj-D at k=17 → mAP ≈ 5.4%
    print("\n=== MSMT: randproj-D at k=17 (matched 5.40% mAP) ===", flush=True)
    q_D, qp_D, qc_D, G_D, gp_D, gc_D = load_cell('D')
    print(f"  randproj-degrade D at k={MSMT_K_RAND}...", flush=True)
    G_D_rp, Q_D_rp = rand_proj_degrade(G_D, q_D, MSMT_K_RAND)
    results['MSMT_randproj'] = evaluate_condition(
        "MSMT randproj-D k=17", Q_D_rp, qp_D, qc_D, G_D_rp, gp_D, gc_D)

    print("\n\n=== SUMMARY (E32 + E33) ===")
    print(f"{'condition':<40} {'base mAP':>9} {'rr mAP':>9} {'ΔmAP':>9} {'base R-1':>9} {'rr R-1':>9} {'ΔR-1':>9}")
    # Load E32 for comparison
    try:
        e32 = json.load(open(RESULTS / 'e32_rerank.json'))
        for ev_set, conds in e32.items():
            if ev_set == 'params': continue
            for k, row in conds.items():
                print(f"{row['label']:<40} {row['mAP_baseline']:>9.2f} {row['mAP_rerank']:>9.2f} {row['delta_mAP']:>+9.2f} {row['R1_baseline']:>9.2f} {row['R1_rerank']:>9.2f} {row['delta_R1']:>+9.2f}")
    except Exception as e:
        print(f"could not load E32: {e}")
    for label, row in results.items():
        print(f"{row['label']:<40} {row['mAP_baseline']:>9.2f} {row['mAP_rerank']:>9.2f} {row['delta_mAP']:>+9.2f} {row['R1_baseline']:>9.2f} {row['R1_rerank']:>9.2f} {row['delta_R1']:>+9.2f}")

    results['params'] = {'k1': K1, 'k2': K2, 'lambda': LAM,
                         'MSMT_k_rand': MSMT_K_RAND, 'MKT_k_rand': MKT_K_RAND}
    out_path = RESULTS / 'e33_randproj_rerank.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
