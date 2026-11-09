"""
B1 correction (rev 303): project out the evaluated arm's OWN camera-offset
span B_cam, not the cross-model learned span from Item 4.

For each of the four cells (A', B', C, D):
  - Load gallery and query features (L2-normalized).
  - Build B_cam = top SVD basis of gallery camera centroid offsets,
    rank = n_cams - 1 (full span of the mean-centered cam offsets).
  - Project B_cam out of both gallery and query, renormalize to unit L2.
  - Recompute mAP under standard protocol (same (pid, cam) junk excluded).
  - Identity-clustered bootstrap for delta mAP CI.
  - Report B_cam's own concentration: eff rank of cam-offset singular
    values, energy of B_cam in top-K PCs of gallery.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
RNG_SEED = 42

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',   'n_q': 3368,  'label': "Market model → Market (id)"},
    'B_prime': {'file': 'msmt17_test_noREA.npz',   'n_q': 11659, 'label': "Market model → MSMT (ood)"},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)"},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz','n_q': 11659, 'label': "MSMT model → MSMT (id)"},
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def camera_offsets(X, cids):
    mu_g = X.mean(0)
    unique_c = np.unique(cids)
    return np.stack([X[cids == c].mean(0) - mu_g for c in unique_c])


def top_r_basis_with_s(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:r].T, s


def participation_in_pcs(B, PCs):
    """||proj_PCs(B)||^2_F / ||B||^2_F for orthonormal B."""
    coef = B.T @ PCs
    return float((coef ** 2).sum() / B.shape[1])


def top_K_cov_basis(X, K):
    mu = X.mean(0); Xc = X - mu
    n, d = Xc.shape
    if d <= n:
        C = Xc.T @ Xc / max(n - 1, 1)
        w, v = np.linalg.eigh(C.astype(np.float64))
        order = np.argsort(-w)[:K]
        return v[:, order].astype(np.float32), w[order].astype(np.float32)
    G = Xc @ Xc.T / max(n - 1, 1)
    w, u = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(-w)[:K]
    w_sel = w[order]; u_sel = u[:, order]
    v = (Xc.T @ u_sel) / np.sqrt(np.maximum(w_sel, 1e-12) * (n - 1))
    v = v / np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-12)
    return v.astype(np.float32), w_sel.astype(np.float32)


def project_out(X, B):
    """Remove B's span from rows of X: X - X @ B @ B.T. Then renormalize."""
    X_proj = X - (X @ B) @ B.T
    return l2(X_proj)


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, chunk=200):
    """Per-query AP, standard protocol (same (pid,cam) junk excluded).
    Returns an array of length n_q with NaN for queries with no good match."""
    n_q = len(q_pids)
    aps = np.full(n_q, np.nan, dtype=np.float64)
    for i0 in range(0, n_q, chunk):
        i1 = min(i0 + chunk, n_q)
        sims = q_emb[i0:i1] @ g_emb.T  # (chunk, n_g)
        for j, i in enumerate(range(i0, i1)):
            qid, qcam = q_pids[i], q_cids[i]
            junk = (g_pids == qid) & (g_cids == qcam)
            order = np.argsort(-sims[j])
            order = order[~junk[order]]
            good = g_pids[order] == qid
            if not good.any():
                continue
            pos = np.where(good)[0]
            aps[i] = float(np.mean((np.arange(1, good.sum() + 1) / (pos + 1))))
    return aps


def identity_clustered_bootstrap_map(aps_base, aps_proj, q_pids, n_boot=500, rng_seed=42):
    """Resample identities (not queries) and aggregate mAP per draw.
    Return mean, median, CI on delta = proj - base."""
    rng = np.random.default_rng(rng_seed)
    valid = ~(np.isnan(aps_base) | np.isnan(aps_proj))
    base_valid = aps_base[valid]
    proj_valid = aps_proj[valid]
    pid_valid = q_pids[valid]
    unique_pids = np.unique(pid_valid)
    # Precompute AP lists per pid
    pid_idx = {p: np.where(pid_valid == p)[0] for p in unique_pids}
    diffs = []
    bases = []
    projs = []
    for _ in range(n_boot):
        sampled_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        idxs = np.concatenate([pid_idx[p] for p in sampled_pids])
        b = base_valid[idxs].mean()
        p = proj_valid[idxs].mean()
        bases.append(b); projs.append(p); diffs.append(p - b)
    diffs = np.array(diffs); bases = np.array(bases); projs = np.array(projs)
    return {
        'base_mAP_obs':  float(np.nanmean(base_valid)),
        'proj_mAP_obs':  float(np.nanmean(proj_valid)),
        'delta_pp_point': float(np.nanmean(proj_valid) - np.nanmean(base_valid)) * 100,
        'delta_pp_median_boot': float(np.median(diffs)) * 100,
        'delta_pp_ci': [float(np.quantile(diffs, 0.025)) * 100, float(np.quantile(diffs, 0.975)) * 100],
        'base_mAP_pct_ci': [float(np.quantile(bases, 0.025)) * 100, float(np.quantile(bases, 0.975)) * 100],
        'n_valid_queries': int(valid.sum()),
        'n_unique_pids': int(len(unique_pids)),
        'n_boot': n_boot,
    }


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    n_cams = len(np.unique(g_cids))
    r = n_cams - 1  # full span of mean-centered cam offsets
    print(f"  n_q={n_q}  n_g={len(g_pids)}  n_cams={n_cams}  r (project out) = {r}", flush=True)

    # Build B_cam from gallery
    offsets = camera_offsets(g_emb, g_cids)
    B_cam, s = top_r_basis_with_s(offsets, r)
    eff_rank = float((s ** 2).sum() ** 2 / ((s ** 2) ** 2).sum())
    print(f"  cam-offset sing. values: {np.round(s, 4)}", flush=True)
    print(f"  cam-offset effective rank = {eff_rank:.3f}", flush=True)
    # Energy of B_cam in top-K PCs of gallery feature cov
    arm_pc_energy = {}
    for K in [20, 50, 100]:
        PCs, _ = top_K_cov_basis(g_emb, K)
        arm_pc_energy[f'B_cam_energy_top_{K}_PCs'] = participation_in_pcs(B_cam, PCs)
        print(f"  B_cam energy in top-{K} PCs of gallery = {arm_pc_energy[f'B_cam_energy_top_{K}_PCs']:.4f}", flush=True)

    # Baseline mAP
    print(f"  computing baseline mAP...", flush=True)
    aps_base = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    base_map_pct = float(np.nanmean(aps_base)) * 100
    print(f"  baseline mAP = {base_map_pct:.4f}%", flush=True)

    # Project out own B_cam, remeasure
    print(f"  projecting out own B_cam and remeasuring...", flush=True)
    q_emb_p = project_out(q_emb, B_cam)
    g_emb_p = project_out(g_emb, B_cam)
    aps_proj = per_query_ap(q_emb_p, q_pids, q_cids, g_emb_p, g_pids, g_cids)
    proj_map_pct = float(np.nanmean(aps_proj)) * 100
    print(f"  projected mAP = {proj_map_pct:.4f}%  delta = {proj_map_pct - base_map_pct:+.4f} pp", flush=True)

    # Identity-clustered bootstrap
    print(f"  running identity-clustered bootstrap (n={N_BOOT})...", flush=True)
    boot = identity_clustered_bootstrap_map(aps_base, aps_proj, q_pids, n_boot=N_BOOT, rng_seed=RNG_SEED)
    print(f"  delta mAP = {boot['delta_pp_point']:+.4f} pp  CI [{boot['delta_pp_ci'][0]:+.4f}, {boot['delta_pp_ci'][1]:+.4f}]", flush=True)

    return {
        'cell': tag,
        'label': info['label'],
        'file': info['file'],
        'n_cams': n_cams,
        'r_projected': r,
        'cam_offset_singular_values': s.tolist(),
        'cam_offset_effective_rank': eff_rank,
        **arm_pc_energy,
        **boot,
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)
    out_path = RESULTS / 'e43_B1_own_Bcam_projection.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    print("\n\n=== B1 summary (own-B_cam projection, full span) ===")
    for tag in CELLS:
        r = out[tag]
        print(f"  {tag} ({r['label']}): base={r['base_mAP_obs']*100:.3f}%  proj={r['proj_mAP_obs']*100:.3f}%  delta={r['delta_pp_point']:+.3f} pp  CI [{r['delta_pp_ci'][0]:+.3f}, {r['delta_pp_ci'][1]:+.3f}]")


if __name__ == '__main__':
    main()
