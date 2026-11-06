#!/usr/bin/env python3
"""
E17b — Rank-sweep with proper descending rank instruments.

Two instruments that genuinely REDUCE effective rank:
  1. Top-k retention: project ONTO leading k eigendirections (not out of them).
     k = 2048 (identity) → 1024 → 512 → 256 → 128 → 64 → 32 → 16
  2. Spectral sharpening: λᵢ → λᵢ^γ for γ > 1; solved analytically for targets.

Decisive tests:
  MSMT17 eval: reduce Cell D (eff_rank=359.9) to 133.2 (Cell B intact) → compare mAP to 2.56%
  Market eval:  reduce Cell A (eff_rank=261.2) to 208.9 (Cell C s42 intact) → compare to 26.66%
  (Note: 208.9 = seed-42 value; reference 207.2 = 2-seed average, both noted in output)

Harness gate: k=2048 retention must match 2x2 baseline mAP and eff_rank.

Output: /data/papers/low_dimensional_reid/results/e17b_results.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reid-tk"))

CACHE_DIR = Path("/tmp/e12_feature_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Retention sweep: k from large (near-identity) to small (most restricted)
KS_RETENTION = [2048, 1024, 512, 256, 128, 64, 32, 16]
# Sharpening gamma sweep (1.0 = identity)
GAMMAS_SWEEP = [1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]

# Reference 2x2 values (seed 42 only, standard protocol)
REF_MAP = {"A": 85.84, "B": 2.56, "C": 26.66, "D": 41.62}
REF_EFFRANK = {"A": 261.20, "B": 133.30, "C": 208.86, "D": 359.91}

# Decisive test targets
# MSMT17 eval: reduce Cell D to Cell B's eff_rank
DECISIVE_D_TARGET = REF_EFFRANK["B"]  # 133.30
# Market eval: reduce Cell A to Cell C's eff_rank (seed 42 value)
DECISIVE_A_TARGET = REF_EFFRANK["C"]  # 208.86


def effective_rank(X):
    """Centered covariance via eigvalsh (matches reference formula)."""
    X = X.astype(np.float64)
    Xc = X - X.mean(0)
    n = len(Xc)
    cov = Xc.T @ Xc / (n - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = eigvals[eigvals > 1e-12]
    p = eigvals / eigvals.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-300))))


def effective_rank_from_eigvals(eigvals_sorted_desc, gamma=1.0):
    """Compute eff_rank analytically for sharpened spectrum (pre-L2-norm approximation).
    Used for fast binary search of gamma.
    """
    lam = np.maximum(eigvals_sorted_desc, 0.0)
    lam_gamma = np.power(lam[lam > 1e-20], gamma)
    lam_gamma /= lam_gamma.sum()
    return float(np.exp(-np.sum(lam_gamma * np.log(lam_gamma + 1e-300))))


def project_onto(X, V_k):
    """Project X onto subspace spanned by columns of V_k (top-k eigenvectors).
    Returns projected + L2-renormalized features.
    """
    X64 = X.astype(np.float64)
    proj = X64 @ V_k @ V_k.T      # (n, D) — projection onto top-k subspace
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (proj / norms).astype(np.float32)


def sharpen_features(X, U, eigvals, gamma):
    """Apply metric sharpening via coordinate scaling in eigenbasis.
    X: (n, D) — L2-normalized unit-sphere features (NOT pre-centered)
    U: (D, D) — eigenvectors (columns), sorted descending eigenvalue
    eigvals: (D,) — eigenvalues in descending order

    Transform: project onto eigenbasis, scale by λᵢ^((γ-1)/2), L2-renormalize.
    At γ=1: returns X unchanged (exact identity → preserves reference mAP/eff_rank).
    At γ>1: emphasizes high-variance directions (sharpening → lower eff_rank).
    """
    if gamma == 1.0:
        return X
    X64 = X.astype(np.float64)
    # Project onto eigenbasis
    coeffs = X64 @ U                                             # (n, D)
    # Scale by λᵢ^((γ-1)/2)
    scale = np.power(np.maximum(eigvals, 1e-20), (gamma - 1) / 2.0)
    coeffs_scaled = coeffs * scale[np.newaxis, :]               # (n, D)
    # L2-renormalize (stay on unit sphere in scaled space)
    norms = np.linalg.norm(coeffs_scaled, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return (coeffs_scaled / norms).astype(np.float32)


def retrieval_map_batched(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    """mAP via cosine similarity (features assumed pre-normalized). Returns (mAP%, R1%)."""
    n_q = len(q_pids)
    aps = []
    r1_hits = []
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        qb = q_emb[start:end].astype(np.float32)
        sim = qb @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi, idx in enumerate(order):
            qpid = q_pids[start + qi]; qcid = q_cids[start + qi]
            gp = g_pids[idx]; gc = g_cids[idx]
            mask_valid = ~((gp == qpid) & (gc == qcid))
            mask_pos = (gp == qpid) & mask_valid
            if mask_pos.sum() == 0:
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            n_pos = len(pos_in_valid)
            prec = (np.arange(1, n_pos + 1)) / (pos_in_valid + 1)
            aps.append(prec.mean())
            # R1: first valid match is positive?
            r1_hits.append(int(mask_pos[valid_idx[0]]) if len(valid_idx) > 0 else 0)
    map_pct = float(np.mean(aps) * 100) if aps else 0.0
    r1_pct = float(np.mean(r1_hits) * 100) if r1_hits else 0.0
    return map_pct, r1_pct


def solve_gamma_for_target(eigvals, target_er, tol=0.1):
    """Find gamma s.t. effective_rank_from_eigvals(eigvals, gamma) ≈ target_er.
    Uses the pre-L2-norm approximation (fast). If target is above current eff_rank
    (at gamma=1), return gamma=1.0 with a warning.
    """
    er_at_1 = effective_rank_from_eigvals(eigvals, gamma=1.0)
    if target_er >= er_at_1:
        return None, er_at_1  # can't increase eff_rank with sharpening

    def f(g):
        return effective_rank_from_eigvals(eigvals, g) - target_er

    # Search range: gamma in [1.0, 100.0]
    try:
        gamma_opt = brentq(f, 1.0 + 1e-6, 100.0, xtol=1e-4, maxiter=100)
    except ValueError:
        # At gamma=100, still above target? Try larger
        er_100 = effective_rank_from_eigvals(eigvals, gamma=100.0)
        if er_100 > target_er:
            return 100.0, er_100
        gamma_opt = brentq(f, 1.0 + 1e-6, 1000.0, xtol=1e-4, maxiter=100)

    er_achieved = effective_rank_from_eigvals(eigvals, gamma_opt)
    return float(gamma_opt), float(er_achieved)


def run_cell(cell_name, q_emb, q_pids, q_cids, g_emb, g_pids, g_cids,
             decisive_target=None):
    """Run both instruments for a cell. Returns a list of result dicts."""
    print(f"\n=== Cell {cell_name} ({len(q_pids)} queries × {len(g_pids)} gallery) ===", flush=True)
    D = g_emb.shape[1]
    n_g = len(g_pids)

    # Eigendecompose gallery covariance (centered, D×D)
    print("  Eigendecomposing gallery covariance...", flush=True)
    g_mean = g_emb.mean(axis=0).astype(np.float64)
    g_c = g_emb.astype(np.float64) - g_mean
    cov = g_c.T @ g_c / (n_g - 1)
    eigvals_cov, U = np.linalg.eigh(cov)
    # Sort descending
    idx_desc = np.argsort(eigvals_cov)[::-1]
    eigvals_cov = eigvals_cov[idx_desc]
    U = U[:, idx_desc].copy()

    # Full test set for eff_rank
    full_emb = np.concatenate([q_emb, g_emb], axis=0)

    results = []

    # =========================================================================
    # INSTRUMENT 1: Top-k retention
    # =========================================================================
    print("  Instrument 1: Top-k retention", flush=True)
    for k in KS_RETENTION:
        k_clip = min(k, D)
        V_k = U[:, :k_clip]  # top-k eigenvectors

        if k >= D:
            # Identity: no change (k=2048 = all dimensions)
            q_out, g_out = q_emb, g_emb
            full_out = full_emb
            label = "identity"
        else:
            q_out = project_onto(q_emb, V_k)
            g_out = project_onto(g_emb, V_k)
            full_out = project_onto(full_emb, V_k)
            label = f"top{k}"

        er = effective_rank(full_out)
        map_pct, r1_pct = retrieval_map_batched(q_out, q_pids, q_cids, g_out, g_pids, g_cids)
        print(f"    k={k:5d} ({label}): eff_rank={er:.2f}  mAP={map_pct:.3f}%  R1={r1_pct:.2f}%", flush=True)

        results.append({
            "cell": cell_name,
            "instrument": "top_k_retention",
            "k": k,
            "effective_rank": er,
            "map_pct": map_pct,
            "r1_pct": r1_pct,
        })

    # =========================================================================
    # INSTRUMENT 2: Spectral sharpening
    # =========================================================================
    print("  Instrument 2: Spectral sharpening", flush=True)
    # sharpen_features takes raw L2-normalized features; at gamma=1.0 returns identity.

    for gamma in GAMMAS_SWEEP:
        q_out = sharpen_features(q_emb, U, eigvals_cov, gamma)
        g_out = sharpen_features(g_emb, U, eigvals_cov, gamma)
        full_out = sharpen_features(full_emb, U, eigvals_cov, gamma)

        er = effective_rank(full_out)
        map_pct, r1_pct = retrieval_map_batched(q_out, q_pids, q_cids, g_out, g_pids, g_cids)
        er_analytic = effective_rank_from_eigvals(eigvals_cov, gamma)
        print(f"    γ={gamma:.1f}: eff_rank={er:.2f} (analytic={er_analytic:.2f})  mAP={map_pct:.3f}%  R1={r1_pct:.2f}%", flush=True)

        results.append({
            "cell": cell_name,
            "instrument": "spectral_sharpening",
            "gamma": gamma,
            "effective_rank": er,
            "effective_rank_analytic": er_analytic,
            "map_pct": map_pct,
            "r1_pct": r1_pct,
        })

    # =========================================================================
    # DECISIVE TEST: sharpen to specific target eff_rank
    # =========================================================================
    if decisive_target is not None:
        print(f"  Decisive test: sharpen Cell {cell_name} to eff_rank={decisive_target:.2f}", flush=True)
        gamma_opt, er_analytic_opt = solve_gamma_for_target(eigvals_cov, decisive_target)

        if gamma_opt is None:
            print(f"    Cannot reduce eff_rank to {decisive_target:.2f} (current={er_analytic_opt:.2f})", flush=True)
            results.append({
                "cell": cell_name,
                "instrument": "decisive_test",
                "target_eff_rank": decisive_target,
                "gamma": None,
                "note": f"target {decisive_target:.2f} >= current {er_analytic_opt:.2f}, not achievable by sharpening",
                "effective_rank": effective_rank(full_emb),
                "map_pct": results[0]["map_pct"],  # use k=2048 (identity)
            })
        else:
            q_out = sharpen_features(q_emb, U, eigvals_cov, gamma_opt)
            g_out = sharpen_features(g_emb, U, eigvals_cov, gamma_opt)
            full_out = sharpen_features(full_emb, U, eigvals_cov, gamma_opt)
            er = effective_rank(full_out)
            map_pct, r1_pct = retrieval_map_batched(q_out, q_pids, q_cids, g_out, g_pids, g_cids)
            print(f"    γ={gamma_opt:.4f}: eff_rank={er:.2f} (target={decisive_target:.2f})  mAP={map_pct:.3f}%  R1={r1_pct:.2f}%", flush=True)

            results.append({
                "cell": cell_name,
                "instrument": "decisive_test",
                "target_eff_rank": decisive_target,
                "gamma": gamma_opt,
                "effective_rank": er,
                "effective_rank_analytic": er_analytic_opt,
                "map_pct": map_pct,
                "r1_pct": r1_pct,
            })

    return results


def main():
    print("E17b: Rank sweep with proper descending rank instruments", flush=True)

    # Load cached features
    c_mkt = np.load(CACHE_DIR / "market_test.npz")
    mkt_emb, mkt_pids, mkt_cids = c_mkt["emb"], c_mkt["pids"], c_mkt["cids"]
    c_msmt = np.load(CACHE_DIR / "msmt17_test.npz")
    msmt_emb, msmt_pids, msmt_cids = c_msmt["emb"], c_msmt["pids"], c_msmt["cids"]
    c_bmkt = np.load(CACHE_DIR / "bot_msmt_seed42_market.npz")
    bmkt_emb = c_bmkt["emb"]
    c_bmsmt = np.load(CACHE_DIR / "bot_msmt_seed42_msmt.npz")
    bmsmt_emb = c_bmsmt["emb"]

    n_q_mkt = 3368; n_q_msmt = 11655

    all_results = []

    # Cell A: BoT-Market → Market
    all_results += run_cell(
        "A",
        mkt_emb[:n_q_mkt], mkt_pids[:n_q_mkt], mkt_cids[:n_q_mkt],
        mkt_emb[n_q_mkt:], mkt_pids[n_q_mkt:], mkt_cids[n_q_mkt:],
        decisive_target=DECISIVE_A_TARGET,
    )

    # Cell B: BoT-Market → MSMT17
    all_results += run_cell(
        "B",
        msmt_emb[:n_q_msmt], msmt_pids[:n_q_msmt], msmt_cids[:n_q_msmt],
        msmt_emb[n_q_msmt:], msmt_pids[n_q_msmt:], msmt_cids[n_q_msmt:],
        decisive_target=None,  # Cell B is the reference target, not the test
    )

    # Cell C: BoT-MSMT17 → Market
    all_results += run_cell(
        "C",
        bmkt_emb[:n_q_mkt], mkt_pids[:n_q_mkt], mkt_cids[:n_q_mkt],
        bmkt_emb[n_q_mkt:], mkt_pids[n_q_mkt:], mkt_cids[n_q_mkt:],
        decisive_target=None,
    )

    # Cell D: BoT-MSMT17 → MSMT17
    all_results += run_cell(
        "D",
        bmsmt_emb[:n_q_msmt], msmt_pids[:n_q_msmt], msmt_cids[:n_q_msmt],
        bmsmt_emb[n_q_msmt:], msmt_pids[n_q_msmt:], msmt_cids[n_q_msmt:],
        decisive_target=DECISIVE_D_TARGET,
    )

    # =========================================================================
    # Harness gate (k=2048 retention = identity = baseline)
    # =========================================================================
    print("\n" + "="*60, flush=True)
    print("HARNESS GATE (k=2048 retention must match 2×2 baseline)", flush=True)
    print("="*60, flush=True)

    tol_map = 0.05; tol_er = 2.0
    gate_pass = True
    for cell in ["A", "B", "C", "D"]:
        k2048 = next(r for r in all_results
                     if r["cell"] == cell and r.get("instrument") == "top_k_retention" and r["k"] == 2048)
        map_ok = abs(k2048["map_pct"] - REF_MAP[cell]) <= tol_map
        er_ok = abs(k2048["effective_rank"] - REF_EFFRANK[cell]) <= tol_er
        status = "PASS" if (map_ok and er_ok) else "FAIL"
        if not (map_ok and er_ok): gate_pass = False
        print(f"  Cell {cell}: mAP={k2048['map_pct']:.3f}% (ref {REF_MAP[cell]:.2f}) "
              f"er={k2048['effective_rank']:.2f} (ref {REF_EFFRANK[cell]:.2f}) [{status}]", flush=True)

    print(f"\nHarness gate: {'PASSED' if gate_pass else 'FAILED'}", flush=True)

    # =========================================================================
    # Decisive test readout
    # =========================================================================
    print("\n" + "="*60, flush=True)
    print("DECISIVE TESTS", flush=True)
    print("="*60, flush=True)

    # MSMT17 eval: Cell D (in-domain) sharpened to Cell B's eff_rank vs Cell B intact
    d_dec = next((r for r in all_results if r["cell"] == "D" and r.get("instrument") == "decisive_test"), None)
    b_intact = next(r for r in all_results if r["cell"] == "B" and r.get("instrument") == "top_k_retention" and r["k"] == 2048)
    if d_dec:
        print(f"\nMSMT17 eval decisive test:", flush=True)
        print(f"  Cell D sharpened to eff_rank={d_dec['effective_rank']:.2f} (target={DECISIVE_D_TARGET:.2f}):  mAP={d_dec['map_pct']:.3f}%", flush=True)
        print(f"  Cell B intact (eff_rank={b_intact['effective_rank']:.2f}):  mAP={b_intact['map_pct']:.3f}%", flush=True)
        if d_dec.get("gamma") is not None:
            gap = d_dec["map_pct"] - b_intact["map_pct"]
            ratio = d_dec["map_pct"] / max(b_intact["map_pct"], 1e-6)
            print(f"  Gap: {gap:+.3f}pp  Ratio: {ratio:.2f}×", flush=True)
            print(f"  γ used: {d_dec['gamma']:.4f}", flush=True)
            verdict = "IN_DOMAIN_DOMINATES" if ratio > 3.0 else ("COMPARABLE" if ratio > 1.5 else "CLOSE")
            print(f"  Verdict: {verdict}", flush=True)
            d_dec["decisive_gap_pp"] = gap
            d_dec["decisive_ratio"] = ratio
            d_dec["decisive_verdict"] = verdict

    # Market eval: Cell A sharpened to Cell C's eff_rank vs Cell C intact
    a_dec = next((r for r in all_results if r["cell"] == "A" and r.get("instrument") == "decisive_test"), None)
    c_intact = next(r for r in all_results if r["cell"] == "C" and r.get("instrument") == "top_k_retention" and r["k"] == 2048)
    if a_dec:
        print(f"\nMarket eval decisive test:", flush=True)
        print(f"  Cell A sharpened to eff_rank={a_dec['effective_rank']:.2f} (target={DECISIVE_A_TARGET:.2f}):  mAP={a_dec['map_pct']:.3f}%", flush=True)
        print(f"  Cell C intact (eff_rank={c_intact['effective_rank']:.2f}):  mAP={c_intact['map_pct']:.3f}%", flush=True)
        if a_dec.get("gamma") is not None:
            gap = a_dec["map_pct"] - c_intact["map_pct"]
            ratio = a_dec["map_pct"] / max(c_intact["map_pct"], 1e-6)
            print(f"  Gap: {gap:+.3f}pp  Ratio: {ratio:.2f}×", flush=True)
            print(f"  γ used: {a_dec['gamma']:.4f}", flush=True)
            verdict = "IN_DOMAIN_DOMINATES" if ratio > 3.0 else ("COMPARABLE" if ratio > 1.5 else "CLOSE")
            print(f"  Verdict: {verdict}", flush=True)
            a_dec["decisive_gap_pp"] = gap
            a_dec["decisive_ratio"] = ratio
            a_dec["decisive_verdict"] = verdict

    # Save
    out_path = RESULTS_DIR / "e17b_results.json"
    output = {
        "description": "E17b rank sweep with top-k retention and spectral sharpening",
        "reference": {
            "map": REF_MAP,
            "eff_rank": REF_EFFRANK,
            "cell_c_note": "208.86 = seed42 value; reference 207.2 = 2-seed average from compute_bot_msmt_2x2.py",
        },
        "decisive_targets": {
            "D_to_B_eff_rank": DECISIVE_D_TARGET,
            "A_to_C_eff_rank": DECISIVE_A_TARGET,
        },
        "harness_gate_passed": gate_pass,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
