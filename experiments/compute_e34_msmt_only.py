"""
E34 — MSMT-only re-run with better σ range for R-1-matched noise rerank.

Market side already computed in e34_rerank.json first pass (σ=0.065, R-1=51.81).
This script completes MSMT: σ ∈ {0.065, 0.068, 0.070} to bracket R-1=15.50
(from previous scan: σ=0.0625 → R-1=23.00, σ=0.075 → R-1=8.79;
linear interp predicts σ≈0.069 for R-1=15.50).
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


def load_cell(cell):
    files = {'D':  ('bot_msmt_seed42_msmt.npz', 11659)}
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
    del q_g, q_q, g_g, final_dist
    return float(mAP * 100), float(cmc[0] * 100)


def main():
    out_json = RESULTS / 'e34_rerank_R1matched.json'
    if out_json.exists():
        results = json.load(open(out_json))
    else:
        results = {'params': {'k1': K1, 'k2': K2, 'lambda': LAM,
                              'R1_target_MSMT': 15.50, 'R1_target_Market': 53.86}}

    # Known results from first pass
    results['Market_scan'] = [
        {'sigma': 0.0650, 'mAP': 21.83, 'R1': 51.81},
        {'sigma': 0.0700, 'mAP': 15.31, 'R1': 41.36},
        {'sigma': 0.0750, 'mAP': 10.54, 'R1': 31.21},
        {'sigma': 0.0800, 'mAP':  7.27, 'R1': 22.98},
    ]
    results['Market_E34'] = {
        'label': "Market noise-A' at σ=0.0650",
        'sigma': 0.0650,
        'mAP_baseline': 21.83, 'R1_baseline': 51.81,
        'mAP_rerank': 51.16, 'R1_rerank': 66.33,
        'delta_mAP': 51.16 - 21.83, 'delta_R1': 66.33 - 51.81,
    }

    print("\n=== MSMT: noise-D at R-1 matched to B' (15.50) ===", flush=True)
    q_D, qp_D, qc_D, G_D, gp_D, gc_D = load_cell('D')

    # Scan σ closer to target. Priors: σ=0.0625 → R-1=23.00, σ=0.075 → R-1=8.79.
    sigmas = [0.065, 0.068, 0.070]
    scan = []
    for s in sigmas:
        Gd, Qd = noise_degrade(G_D, q_D, s)
        t0 = time.time()
        m, r = map_r1(Qd, qp_D, qc_D, Gd, gp_D, gc_D)
        print(f"    σ={s:.4f}  mAP={m:.2f}%  R-1={r:.2f}%  t={(time.time()-t0)/60:.1f}m",
              flush=True)
        scan.append({'sigma': s, 'mAP': m, 'R1': r})
        del Gd, Qd

    best = min(scan, key=lambda r: abs(r['R1'] - 15.50))
    print(f"\n  picked σ={best['sigma']:.4f}  R-1={best['R1']:.2f}% "
          f"(|Δ|={abs(best['R1']-15.50):.2f})", flush=True)
    results['MSMT_scan'] = scan

    # Rerank at picked σ
    s = best['sigma']
    print(f"\n  --- MSMT noise-D at σ={s:.4f}: rerank ---", flush=True)
    Gd, Qd = noise_degrade(G_D, q_D, s)
    t0 = time.time()
    m_b, r_b = map_r1(Qd, qp_D, qc_D, Gd, gp_D, gc_D)
    print(f"    baseline:  mAP={m_b:.2f}%  R-1={r_b:.2f}%  t={(time.time()-t0)/60:.1f}m",
          flush=True)
    t1 = time.time()
    m_r, r_r = map_r1_rerank(Qd, qp_D, qc_D, Gd, gp_D, gc_D)
    print(f"    reranked:  mAP={m_r:.2f}%  R-1={r_r:.2f}%  t={(time.time()-t1)/60:.1f}m",
          flush=True)

    results['MSMT_E34'] = {
        'label': f"MSMT noise-D at σ={s:.4f}",
        'sigma': s,
        'mAP_baseline': m_b, 'R1_baseline': r_b,
        'mAP_rerank': m_r, 'R1_rerank': r_r,
        'delta_mAP': m_r - m_b, 'delta_R1': r_r - r_b,
    }

    # Summary
    print("\n\n=== SUMMARY (E32 + E33 + E34) ===")
    hdr = f"{'condition':<44} {'base mAP':>9} {'rr mAP':>9} {'ΔmAP':>9} {'base R-1':>9} {'rr R-1':>9} {'ΔR-1':>9}"
    print(hdr)
    try:
        e32 = json.load(open(RESULTS / 'e32_rerank.json'))
        for ev_set, conds in e32.items():
            if ev_set == 'params': continue
            for k, row in conds.items():
                print(f"{row['label']:<44} {row['mAP_baseline']:>9.2f} {row['mAP_rerank']:>9.2f} {row['delta_mAP']:>+9.2f} {row['R1_baseline']:>9.2f} {row['R1_rerank']:>9.2f} {row['delta_R1']:>+9.2f}")
    except Exception as e:
        print(f"could not load E32: {e}")
    try:
        e33 = json.load(open(RESULTS / 'e33_randproj_rerank.json'))
        for k, row in e33.items():
            if k == 'params': continue
            print(f"{row['label']:<44} {row['mAP_baseline']:>9.2f} {row['mAP_rerank']:>9.2f} {row['delta_mAP']:>+9.2f} {row['R1_baseline']:>9.2f} {row['R1_rerank']:>9.2f} {row['delta_R1']:>+9.2f}")
    except Exception as e:
        print(f"could not load E33: {e}")
    for k in ('Market_E34', 'MSMT_E34'):
        row = results[k]
        print(f"{row['label']:<44} {row['mAP_baseline']:>9.2f} {row['mAP_rerank']:>9.2f} {row['delta_mAP']:>+9.2f} {row['R1_baseline']:>9.2f} {row['R1_rerank']:>9.2f} {row['delta_R1']:>+9.2f}")

    with open(out_json, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_json}")


if __name__ == '__main__':
    main()
