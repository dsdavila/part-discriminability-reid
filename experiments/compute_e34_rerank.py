"""
E34 — k-reciprocal re-ranking on noise-degraded features at R-1-matched σ.

Replaces E32's mAP-matching with Rank-1-matching. From E32:
  noise-D at σ=0.0625 → mAP=5.45 but R-1=23.00 (1.48× B' R-1 of 15.50)
  noise-A' at σ=0.062 → mAP=26.64 but R-1=57.93 (above C R-1 of 53.86)

To discriminate reciprocity-vs-denoising from rank-distribution-artifact:
re-run noise arms at σ chosen so baseline R-1 matches real transfer's R-1.

Targets:
  MSMT noise-D: R-1 = 15.50 (B')
  Market noise-A': R-1 = 53.86 (C)

Phase 1: scan a few σ values, compute baseline mAP+R-1 only (no rerank).
Phase 2: at picked σ (closest to target R-1), run full rerank.
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


def search_sigma(Q, qp, qc, G, gp, gc, sigmas, r1_target, label):
    print(f"\n  --- {label}: σ-search for R-1={r1_target} ---", flush=True)
    rows = []
    for s in sigmas:
        Gd, Qd = noise_degrade(G, Q, s)
        t0 = time.time()
        m, r = map_r1(Qd, qp, qc, Gd, gp, gc)
        print(f"    σ={s:.4f}  mAP={m:.2f}%  R-1={r:.2f}%  t={(time.time()-t0)/60:.1f}m",
              flush=True)
        rows.append({'sigma': s, 'mAP': m, 'R1': r})
        del Gd, Qd
    best = min(rows, key=lambda r: abs(r['R1'] - r1_target))
    print(f"    picked σ={best['sigma']:.4f}  R-1={best['R1']:.2f}% "
          f"(|Δ|={abs(best['R1']-r1_target):.2f})", flush=True)
    return best['sigma'], rows


def evaluate_rerank_at(label, Q, qp, qc, G, gp, gc, sigma):
    print(f"\n  --- {label} at σ={sigma:.4f} ---", flush=True)
    Gd, Qd = noise_degrade(G, Q, sigma)
    t0 = time.time()
    m_b, r_b = map_r1(Qd, qp, qc, Gd, gp, gc)
    print(f"    baseline:  mAP={m_b:.2f}%  R-1={r_b:.2f}%  t={(time.time()-t0)/60:.1f}m",
          flush=True)
    t1 = time.time()
    m_r, r_r = map_r1_rerank(Qd, qp, qc, Gd, gp, gc)
    print(f"    reranked:  mAP={m_r:.2f}%  R-1={r_r:.2f}%  t={(time.time()-t1)/60:.1f}m",
          flush=True)
    del Gd, Qd
    return {'label': label, 'sigma': sigma,
            'mAP_baseline': m_b, 'R1_baseline': r_b,
            'mAP_rerank': m_r, 'R1_rerank': r_r,
            'delta_mAP': m_r - m_b, 'delta_R1': r_r - r_b}


def main():
    results = {'params': {'k1': K1, 'k2': K2, 'lambda': LAM,
                          'R1_target_MSMT': 15.50, 'R1_target_Market': 53.86}}

    # --- Market first (cheap) ---
    print("\n=== MARKET: noise-A' at R-1 matched to C (53.86) ===", flush=True)
    q_A, qp_A, qc_A, G_A, gp_A, gc_A = load_cell("A'")
    # E32 had σ=0.062 → R-1=57.93 (above target). Need larger σ.
    mkt_sigmas = [0.065, 0.070, 0.075, 0.080]
    mkt_sigma, mkt_scan = search_sigma(q_A, qp_A, qc_A, G_A, gp_A, gc_A,
                                        mkt_sigmas, 53.86, "Market noise-A'")
    results['Market_scan'] = mkt_scan
    results['Market_E34'] = evaluate_rerank_at(
        f"Market noise-A' at σ={mkt_sigma:.4f}",
        q_A, qp_A, qc_A, G_A, gp_A, gc_A, mkt_sigma)
    del q_A, qp_A, qc_A, G_A, gp_A, gc_A

    # --- MSMT (expensive) ---
    print("\n=== MSMT: noise-D at R-1 matched to B' (15.50) ===", flush=True)
    q_D, qp_D, qc_D, G_D, gp_D, gc_D = load_cell('D')
    # E32 had σ=0.0625 → R-1=23.00. Need larger σ.
    msmt_sigmas = [0.075, 0.085, 0.095]
    msmt_sigma, msmt_scan = search_sigma(q_D, qp_D, qc_D, G_D, gp_D, gc_D,
                                          msmt_sigmas, 15.50, "MSMT noise-D")
    results['MSMT_scan'] = msmt_scan
    results['MSMT_E34'] = evaluate_rerank_at(
        f"MSMT noise-D at σ={msmt_sigma:.4f}",
        q_D, qp_D, qc_D, G_D, gp_D, gc_D, msmt_sigma)

    # --- Summary ---
    print("\n\n=== SUMMARY (E32 + E33 + E34) ===")
    hdr = f"{'condition':<44} {'base mAP':>9} {'rr mAP':>9} {'ΔmAP':>9} {'base R-1':>9} {'rr R-1':>9} {'ΔR-1':>9}"
    print(hdr)
    # Load E32 + E33 for comparison
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

    out_path = RESULTS / 'e34_rerank_R1matched.json'
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
