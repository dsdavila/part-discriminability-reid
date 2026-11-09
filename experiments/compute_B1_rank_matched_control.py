"""
B1 rank-matched control (rev 305). For each of the four cells, project
out two rank-matched surrogate spans and remeasure mAP:

  (i)  identity-offset top-r span: top-r SVD of identity centroids
       centered at global mean (r = n_cams - 1 to match B_cam).
  (ii) structure-matched partition span: random k-partition of items
       (k = n_cams) size-matched to the real camera distribution;
       top-r SVD of fake-camera offsets. 20 draws per cell, average
       delta reported; worst/best draw also recorded.

If (i) or (ii) also gains ~2 pp on B' and C, the camera-specific
claim dies.

Identity-clustered bootstrap n=500 for each delta CI.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
N_PART_DRAWS = 20
RNG_SEED = 42

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)"},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)"},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)"},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)"},
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def top_r_basis(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:r].T


def project_out(X, B):
    X_proj = X - (X @ B) @ B.T
    return l2(X_proj)


def per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, chunk=200):
    n_q = len(q_pids)
    aps = np.full(n_q, np.nan, dtype=np.float64)
    for i0 in range(0, n_q, chunk):
        i1 = min(i0 + chunk, n_q)
        sims = q_emb[i0:i1] @ g_emb.T
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


def boot_delta(aps_base, aps_proj, q_pids, n_boot=N_BOOT, rng_seed=RNG_SEED):
    rng = np.random.default_rng(rng_seed)
    valid = ~(np.isnan(aps_base) | np.isnan(aps_proj))
    b = aps_base[valid]; p = aps_proj[valid]; pid = q_pids[valid]
    uids = np.unique(pid); idx_by_pid = {u: np.where(pid == u)[0] for u in uids}
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(uids, size=len(uids), replace=True)
        idxs = np.concatenate([idx_by_pid[u] for u in sampled])
        diffs.append((p[idxs].mean() - b[idxs].mean()) * 100)
    diffs = np.array(diffs)
    return {
        'base_pct': float(np.nanmean(b)) * 100,
        'proj_pct': float(np.nanmean(p)) * 100,
        'delta_pp_point': float(np.nanmean(p) - np.nanmean(b)) * 100,
        'delta_pp_ci': [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
    }


def identity_offset_span(X, pids, r):
    mu_g = X.mean(0)
    uids = np.unique(pids)
    cents = np.stack([X[pids == p].mean(0) - mu_g for p in uids])
    return top_r_basis(cents, r)


def partition_offset_span(X, cam_sizes, rng, r):
    mu_g = X.mean(0)
    perm = rng.permutation(len(X))
    idx = 0
    offsets = []
    for n_imgs in cam_sizes:
        group = perm[idx:idx+n_imgs]
        idx += n_imgs
        offsets.append(X[group].mean(0) - mu_g)
    return top_r_basis(np.stack(offsets), r)


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    n_cams = len(np.unique(g_cids))
    r = n_cams - 1
    print(f"  n_cams={n_cams}  r={r}", flush=True)

    # Baseline mAP
    print(f"  baseline mAP...", flush=True)
    aps_base = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    base = float(np.nanmean(aps_base)) * 100
    print(f"  baseline = {base:.4f}%", flush=True)

    # (i) Identity-offset surrogate — uses gallery pids (identities)
    print(f"  identity-offset surrogate (r={r})...", flush=True)
    B_id = identity_offset_span(g_emb, g_pids, r)
    q_p = project_out(q_emb, B_id); g_p = project_out(g_emb, B_id)
    aps_id = per_query_ap(q_p, q_pids, q_cids, g_p, g_pids, g_cids)
    id_res = boot_delta(aps_base, aps_id, q_pids)
    print(f"  identity-offset delta = {id_res['delta_pp_point']:+.4f} pp  CI [{id_res['delta_pp_ci'][0]:+.4f}, {id_res['delta_pp_ci'][1]:+.4f}]", flush=True)

    # (ii) Structure-matched partition surrogate — n draws
    print(f"  structure-matched partition surrogate ({N_PART_DRAWS} draws, r={r})...", flush=True)
    cam_sizes = [int((g_cids == c).sum()) for c in np.unique(g_cids)]
    rng = np.random.default_rng(RNG_SEED)
    part_deltas = []
    part_draws = []
    for di in range(N_PART_DRAWS):
        B_part = partition_offset_span(g_emb, cam_sizes, rng, r)
        q_p = project_out(q_emb, B_part); g_p = project_out(g_emb, B_part)
        aps_part = per_query_ap(q_p, q_pids, q_cids, g_p, g_pids, g_cids)
        d_pp = (float(np.nanmean(aps_part)) - float(np.nanmean(aps_base))) * 100
        part_deltas.append(d_pp)
        part_draws.append({'draw': di, 'delta_pp': d_pp})
        if di < 3 or di == N_PART_DRAWS - 1:
            print(f"    draw {di}: delta = {d_pp:+.4f} pp", flush=True)
    part_deltas = np.array(part_deltas)
    # Bootstrap only on the mean-draw for CI on the summary stat
    part_mean = float(part_deltas.mean())
    part_ci = [float(np.quantile(part_deltas, 0.025)), float(np.quantile(part_deltas, 0.975))]
    print(f"  partition surrogate: mean delta = {part_mean:+.4f} pp  (CI across draws [{part_ci[0]:+.4f}, {part_ci[1]:+.4f}], min={part_deltas.min():+.4f}, max={part_deltas.max():+.4f})", flush=True)

    return {
        'cell': tag,
        'label': info['label'],
        'n_cams': n_cams,
        'r_projected': r,
        'baseline_mAP_pct': base,
        'identity_offset_surrogate': id_res,
        'partition_surrogate': {
            'n_draws': N_PART_DRAWS,
            'mean_delta_pp': part_mean,
            'ci_across_draws_pp': part_ci,
            'min_delta_pp': float(part_deltas.min()),
            'max_delta_pp': float(part_deltas.max()),
            'per_draw_delta_pp': part_deltas.tolist(),
        },
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)
    out_path = RESULTS / 'e43_B1_rank_matched_control.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary table
    print("\n\n=== B1 rank-matched control summary ===")
    print(f"{'cell':9s}  {'baseline':>10s}  {'id-offset delta':>18s}  {'partition mean delta':>22s}  {'part range':>18s}")
    for tag in CELLS:
        r = out[tag]
        print(f"{tag:9s}  {r['baseline_mAP_pct']:9.3f}%  {r['identity_offset_surrogate']['delta_pp_point']:+7.3f} pp [{r['identity_offset_surrogate']['delta_pp_ci'][0]:+.2f}, {r['identity_offset_surrogate']['delta_pp_ci'][1]:+.2f}]  {r['partition_surrogate']['mean_delta_pp']:+7.3f} pp                [{r['partition_surrogate']['min_delta_pp']:+.3f}, {r['partition_surrogate']['max_delta_pp']:+.3f}]")


if __name__ == '__main__':
    main()
