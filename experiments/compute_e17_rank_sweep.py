#!/usr/bin/env python3
"""
E17 — effective-rank sweep (claude-chat rev 92).

For each of the four (model, eval-set) cells from the mAP 2×2, sweeps over
k ∈ {0,1,2,4,8,16,32,64,128,256} removing the top-k (most-variance) and
bottom-k (least-variance) gallery eigendirections from both queries and
gallery.  At each k, recomputes:
  - effective rank from the actual ablated gallery spectrum
  - mAP and Rank-1 (standard protocol, same-cam exclusion, L2-normalised)

Harness gate: k=0 must reproduce the 2×2 exactly:
  A=85.84%/261.2  B=2.56%/133.2  C=26.66%/207.2  D=41.62%/359.9

Decisive checks (claude-chat rev 92):
  MSMT17 eval: ablate BoT-MSMT17 (D) top-k until eff_rank ≤ 133.2.
               Compare its mAP to Cell B (2.56%).
  Market eval: ablate BoT-Market (A) top-k until eff_rank ≤ 207.2.
               Compare its mAP to Cell C (26.66%).
"""

import json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

CACHE_DIR = Path("/tmp/e12_feature_cache")
OUT_PATH  = Path("/data/papers/low_dimensional_reid/results/e17_rank_sweep_results.json")
BATCH_Q   = 2000

KS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

# 2×2 reference values for the harness gate
REF_MAP  = {'A': 85.84, 'B': 2.56, 'C': 26.66, 'D': 41.62}
REF_RANK = {'A': 261.2, 'B': 133.2, 'C': 207.2, 'D': 359.9}


def effective_rank(emb):
    """Effective rank = exp(-Σ p_i log p_i) where p_i = λ_i / Σ λ_j.
    Matches compute_bot_msmt_2x2.py: centered covariance, D×D matrix eigdecomp
    (equivalent to SVD of centered data but much faster for N >> D).
    """
    X = emb.astype(np.float64)
    Xc = X - X.mean(0)
    n = len(Xc)
    cov = Xc.T @ Xc / (n - 1)         # centered sample covariance (D, D)
    eigvals = np.linalg.eigvalsh(cov)  # ascending order, real
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = eigvals[eigvals > 1e-12]
    p = eigvals / eigvals.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-300))))


def project_out(X, V):
    """Remove subspace spanned by columns of V from rows of X.
    V: (D, k) orthonormal columns (eigenvectors).
    X: (N, D) feature matrix.
    Returns: (N, D) float32.
    """
    X64 = X.astype(np.float64)
    return (X64 - X64 @ V @ V.T).astype(np.float32)


def retrieval_map_batched(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    """mAP and Rank-1 with same-cam exclusion. L2-normalises internally."""
    q = (q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)).astype(np.float32)
    g = (g_emb / (np.linalg.norm(g_emb, axis=1, keepdims=True) + 1e-10)).astype(np.float32)
    aps, r1_hits = [], 0
    for s in range(0, len(q_pids), BATCH_Q):
        e = min(s + BATCH_Q, len(q_pids))
        sims = q[s:e] @ g.T
        for bi, qi in enumerate(range(s, e)):
            qid, qcam = q_pids[qi], q_cids[qi]
            order = np.argsort(-sims[bi])
            junk = (g_pids == qid) & (g_cids == qcam)
            order = order[~junk[order]]
            good = g_pids[order] == qid
            if not good.any():
                continue
            if good[0]:
                r1_hits += 1
            pos = np.where(good)[0]
            aps.append(float(np.mean(np.arange(1, good.sum() + 1) / (pos + 1))))
    map_pct = float(np.mean(aps)) * 100 if aps else 0.0
    r1_pct  = r1_hits / len(q_pids) * 100
    return map_pct, r1_pct


def load_split(path, n_query):
    d = np.load(path)
    emb = d['emb'].astype(np.float32)
    return (emb[:n_query], d['pids'][:n_query], d['cids'][:n_query],
            emb[n_query:], d['pids'][n_query:], d['cids'][n_query:])


def run_cell(cell_name, q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    print(f"\n=== Cell {cell_name} ({len(q_pids)} queries × {len(g_pids)} gallery) ===", flush=True)
    n_q, n_g = len(q_pids), len(g_pids)

    # Eigendecompose gallery covariance once via D×D matrix (not SVD of N×D).
    # Center gallery; eigenvectors sorted descending by eigenvalue.
    print("  Eigendecomposing gallery covariance...", flush=True)
    g_mean = g_emb.mean(axis=0).astype(np.float64)
    g_c = g_emb.astype(np.float64) - g_mean
    cov = g_c.T @ g_c / (n_g - 1)        # (D, D)
    eigvals_cov, V = np.linalg.eigh(cov)  # ascending; V columns are eigenvectors
    V = V[:, ::-1].copy()                 # (D, D), columns sorted descending
    D = V.shape[0]
    print(f"  D={D}, n_gallery={n_g}", flush=True)

    # Full test set (query + gallery) used for effective rank — matches reference formula
    full_emb = np.concatenate([q_emb, g_emb], axis=0)

    rows = []
    for end in ['top', 'bottom']:
        for k in KS:
            if k == 0:
                q_abl = q_emb
                g_abl = g_emb
                full_abl = full_emb
            else:
                k_clip = min(k, D - 1)
                vecs = V[:, :k_clip] if end == 'top' else V[:, -k_clip:]
                q_abl = project_out(q_emb, vecs)
                g_abl = project_out(g_emb, vecs)
                full_abl = project_out(full_emb, vecs)

            er = effective_rank(full_abl)      # full test set, centered cov
            map_pct, r1_pct = retrieval_map_batched(
                q_abl, q_pids, q_cids, g_abl, g_pids, g_cids)

            row = {
                'cell': cell_name,
                'end': end,
                'k': k,
                'effective_rank': round(er, 3),
                'mAP_pct': round(map_pct, 4),
                'rank1_pct': round(r1_pct, 4),
            }
            rows.append(row)
            print(f"  {end}-{k:3d}: eff_rank={er:7.2f}  mAP={map_pct:7.3f}%  R1={r1_pct:5.2f}%", flush=True)

    return rows


def main():
    # Verify split boundaries against live dataset
    print("Verifying dataset splits...")
    mkt  = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17',     sample=False)
    n_mkt_q  = sum(1 for s in mkt.query  if not s.is_junk)
    n_msmt_q = sum(1 for s in msmt.query if not s.is_junk)
    assert n_mkt_q  == 3368,  f"Market query count: {n_mkt_q}"
    assert n_msmt_q == 11655, f"MSMT17 query count: {n_msmt_q}"
    print(f"  Market query={n_mkt_q}, MSMT17 query={n_msmt_q} ✓", flush=True)

    cells_spec = [
        ('A', CACHE_DIR / "market_test.npz",            n_mkt_q,  "BoT-Market → Market"),
        ('B', CACHE_DIR / "msmt17_test.npz",            n_msmt_q, "BoT-Market → MSMT17"),
        ('C', CACHE_DIR / "bot_msmt_seed42_market.npz", n_mkt_q,  "BoT-MSMT17 → Market"),
        ('D', CACHE_DIR / "bot_msmt_seed42_msmt.npz",   n_msmt_q, "BoT-MSMT17 → MSMT17"),
    ]

    all_rows = []
    for cell_name, path, n_q, label in cells_spec:
        print(f"\nLoading {label} from {path.name}...", flush=True)
        data = load_split(path, n_q)
        rows = run_cell(cell_name, *data)
        all_rows.extend(rows)

    # ── Harness gate ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("HARNESS GATE  (k=0 must match 2×2 within tolerance)")
    print("="*60)
    gate_ok = True
    for c in 'ABCD':
        row = next(r for r in all_rows
                   if r['cell'] == c and r['end'] == 'top' and r['k'] == 0)
        map_ok  = abs(row['mAP_pct'] - REF_MAP[c]) < 0.05
        rank_ok = abs(row['effective_rank'] - REF_RANK[c]) < 2.0
        status  = "PASS" if (map_ok and rank_ok) else "FAIL"
        print(f"  Cell {c}: mAP={row['mAP_pct']:.2f}% (ref {REF_MAP[c]:.2f})  "
              f"eff_rank={row['effective_rank']:.1f} (ref {REF_RANK[c]:.1f})  [{status}]")
        if status == "FAIL":
            gate_ok = False

    if not gate_ok:
        print("\nHARNESS GATE FAILED — stopping. Do not sweep on unvalidated harness.")
        return

    print("Harness gate PASSED.\n")

    # ── Decisive checks ───────────────────────────────────────────────────────
    # Strategy: find crossings where one cell's top-k eff_rank sweep matches
    # another cell's baseline (k=0) eff_rank.  Bottom-k barely changes eff_rank
    # (< 1% shift for k≤256), so we compare top-k sweeps of the cross-domain
    # cells against the in-domain baseline.
    decisive = {}

    # Check 1 (MSMT17 eval):
    #   Find k where Cell B top-k eff_rank = Cell D k=0 eff_rank (≈359.9).
    #   Then compare Cell B mAP at that k vs Cell D mAP (41.62%).
    #   If mAP gap remains large → domain term needed.
    print("="*60)
    print("DECISIVE CHECK 1: MSMT17 eval (Cell B top-k sweep → Cell D baseline)")
    cellD_k0 = next(r for r in all_rows if r['cell'] == 'D' and r['end'] == 'top' and r['k'] == 0)
    cellB_top = sorted([r for r in all_rows if r['cell'] == 'B' and r['end'] == 'top'],
                       key=lambda r: r['k'])
    print(f"  Target eff_rank = Cell D k=0: {cellD_k0['effective_rank']:.2f}")
    for r in cellB_top:
        m = " ← above D baseline" if r['effective_rank'] >= cellD_k0['effective_rank'] else ""
        print(f"  Cell B top-{r['k']:3d}: eff_rank={r['effective_rank']:7.2f}  mAP={r['mAP_pct']:7.3f}%{m}")
    B_at_D_rank = next((r for r in cellB_top if r['effective_rank'] >= cellD_k0['effective_rank']), None)
    if B_at_D_rank:
        diff = B_at_D_rank['mAP_pct'] - cellD_k0['mAP_pct']
        verdict = "RANK_SUFFICIENT" if abs(diff) < 2.0 else "DOMAIN_TERM_NEEDED"
        print(f"\n  → Cell B reaches Cell D's eff_rank at k={B_at_D_rank['k']}: "
              f"eff_rank={B_at_D_rank['effective_rank']:.1f}, "
              f"Cell B mAP={B_at_D_rank['mAP_pct']:.3f}% vs Cell D mAP={cellD_k0['mAP_pct']:.3f}%")
        print(f"  → gap={diff:+.3f}pp, ratio={B_at_D_rank['mAP_pct']/cellD_k0['mAP_pct']:.2f}×  [{verdict}]")
        decisive['MSMT17_eval'] = {
            'strategy': 'Cell B top-k ablated to match Cell D baseline eff_rank',
            'cell_D_baseline': {'k': 0, 'eff_rank': cellD_k0['effective_rank'],
                                'mAP_pct': cellD_k0['mAP_pct']},
            'cell_B_matched': B_at_D_rank,
            'mAP_ratio': round(B_at_D_rank['mAP_pct'] / cellD_k0['mAP_pct'], 4),
            'verdict': verdict,
        }
    else:
        print(f"\n  → Cell B top-k never reaches Cell D's eff_rank in this sweep.")
        decisive['MSMT17_eval'] = {
            'strategy': 'Cell B top-k ablated to match Cell D baseline eff_rank',
            'cell_D_baseline': {'k': 0, 'eff_rank': cellD_k0['effective_rank'],
                                'mAP_pct': cellD_k0['mAP_pct']},
            'verdict': 'NO_CROSSING_FOUND',
        }

    # Check 2 (Market eval, mirror):
    #   Find k where Cell C top-k eff_rank = Cell A k=0 eff_rank (≈261.2).
    #   Then compare Cell C mAP at that k vs Cell A mAP (85.84%).
    print("\n" + "="*60)
    print("DECISIVE CHECK 2: Market eval mirror (Cell C top-k sweep → Cell A baseline)")
    cellA_k0 = next(r for r in all_rows if r['cell'] == 'A' and r['end'] == 'top' and r['k'] == 0)
    cellC_top = sorted([r for r in all_rows if r['cell'] == 'C' and r['end'] == 'top'],
                       key=lambda r: r['k'])
    print(f"  Target eff_rank = Cell A k=0: {cellA_k0['effective_rank']:.2f}")
    for r in cellC_top:
        m = " ← above A baseline" if r['effective_rank'] >= cellA_k0['effective_rank'] else ""
        print(f"  Cell C top-{r['k']:3d}: eff_rank={r['effective_rank']:7.2f}  mAP={r['mAP_pct']:7.3f}%{m}")
    C_at_A_rank = next((r for r in cellC_top if r['effective_rank'] >= cellA_k0['effective_rank']), None)
    if C_at_A_rank:
        diff = C_at_A_rank['mAP_pct'] - cellA_k0['mAP_pct']
        verdict = "RANK_SUFFICIENT" if abs(diff) < 5.0 else "DOMAIN_TERM_NEEDED"
        print(f"\n  → Cell C reaches Cell A's eff_rank at k={C_at_A_rank['k']}: "
              f"eff_rank={C_at_A_rank['effective_rank']:.1f}, "
              f"Cell C mAP={C_at_A_rank['mAP_pct']:.3f}% vs Cell A mAP={cellA_k0['mAP_pct']:.3f}%")
        print(f"  → gap={diff:+.3f}pp, ratio={C_at_A_rank['mAP_pct']/cellA_k0['mAP_pct']:.2f}×  [{verdict}]")
        decisive['Market_eval'] = {
            'strategy': 'Cell C top-k ablated to match Cell A baseline eff_rank',
            'cell_A_baseline': {'k': 0, 'eff_rank': cellA_k0['effective_rank'],
                                'mAP_pct': cellA_k0['mAP_pct']},
            'cell_C_matched': C_at_A_rank,
            'mAP_ratio': round(C_at_A_rank['mAP_pct'] / cellA_k0['mAP_pct'], 4),
            'verdict': verdict,
        }
    else:
        print(f"\n  → Cell C top-k never reaches Cell A's eff_rank in this sweep.")
        decisive['Market_eval'] = {
            'strategy': 'Cell C top-k ablated to match Cell A baseline eff_rank',
            'cell_A_baseline': {'k': 0, 'eff_rank': cellA_k0['effective_rank'],
                                'mAP_pct': cellA_k0['mAP_pct']},
            'verdict': 'NO_CROSSING_FOUND',
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        'experiment': 'E17',
        'description': (
            'Rank sweep: for each 2x2 cell, project out top-k and bottom-k '
            'gallery eigendirections from both query and gallery, then measure '
            'effective rank and mAP at each k.'
        ),
        'protocol': 'standard query/gallery, same-cam exclusion, L2-normalised',
        'k_values': KS,
        'reference_2x2': {
            c: {'mAP_pct': REF_MAP[c], 'effective_rank': REF_RANK[c]}
            for c in 'ABCD'
        },
        'decisive_checks': decisive,
        'rows': all_rows,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
