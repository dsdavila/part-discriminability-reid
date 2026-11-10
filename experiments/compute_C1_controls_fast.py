"""
C1 controls FAST (rev 331 request, rev 346 reminder).

Rewrites the C1 controls with per-camera chunked similarity to avoid
the O(n * d * n) per-item loop. Adds inner-loop logging.

C1-null: identical to prior (per-arm random-partition null for F_wcam,
500 draws). Already landed on B' at rev 342; redoing cleanly here.

C1-AUC: same-cam vs diff-cam per-item AUC, matched negative counts per
item.
  For each item x with identity p, camera c:
    same-cam pos scores = sim(x, (p, c) items \\ x)
    same-cam neg scores = sim(x, (!=p, c) items)
    diff-cam pos scores = sim(x, (p, !=c) items)
    diff-cam neg scores = sim(x, (!=p, !=c) items)
    if all four non-empty:
      n_match = min(|sc_neg|, |dc_neg|)
      sc_neg_sel = subsample sc_neg to n_match
      dc_neg_sel = subsample dc_neg to n_match
      auc_sc = AUC(sc_pos, sc_neg_sel); auc_dc = AUC(dc_pos, dc_neg_sel)
  Aggregate per-identity (mean of its items' AUCs) and then across
  identities.
"""
import json
import numpy as np
from pathlib import Path
import time

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_NULL_DRAWS = 500
RNG_SEED = 42
CHUNK = 500

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)"},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)"},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)"},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)"},
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def f_wcam_on_partition(X, pids, group_id, n_groups):
    s_b = 0.0; s_w = 0.0
    for k in range(n_groups):
        mask_k = (group_id == k)
        if mask_k.sum() == 0:
            continue
        X_k = X[mask_k]; p_k = pids[mask_k]
        mu_k = X_k.mean(0)
        pids_in_k = np.unique(p_k)
        for i in pids_in_k:
            mask_ik = (p_k == i); n_ik = mask_ik.sum()
            X_ik = X_k[mask_ik]; mu_ik = X_ik.mean(0)
            s_b += n_ik * np.sum((mu_ik - mu_k) ** 2)
            s_w += np.sum((X_ik - mu_ik) ** 2)
    return float(s_b), float(s_w)


def auc_pos_vs_neg(s_pos, s_neg):
    if len(s_pos) == 0 or len(s_neg) == 0:
        return np.nan
    n_sorted = np.sort(s_neg)
    ranks_below = np.searchsorted(n_sorted, s_pos, side='left')
    ranks_tied  = np.searchsorted(n_sorted, s_pos, side='right') - ranks_below
    return (ranks_below.sum() + 0.5 * ranks_tied.sum()) / (len(s_pos) * len(s_neg))


def c1_null_fast(X, pids, cids, n_null, rng_seed, log_every=100):
    unique_c = np.unique(cids)
    n_cams = len(unique_c)
    cam_sizes = [int((cids == c).sum()) for c in unique_c]
    cid_to_group = {c: i for i, c in enumerate(unique_c)}
    group_id_real = np.array([cid_to_group[c] for c in cids])
    s_b_obs, s_w_obs = f_wcam_on_partition(X, pids, group_id_real, n_cams)
    f_obs = s_b_obs / max(s_w_obs, 1e-12)
    rng = np.random.default_rng(rng_seed)
    null_f = []
    for i in range(n_null):
        perm = rng.permutation(len(X))
        gid = np.empty(len(X), dtype=np.int64); idx = 0
        for k, nk in enumerate(cam_sizes):
            gid[perm[idx:idx+nk]] = k; idx += nk
        s_b, s_w = f_wcam_on_partition(X, pids, gid, n_cams)
        null_f.append(s_b / max(s_w, 1e-12))
        if i < 3 or (i+1) % log_every == 0 or i == n_null - 1:
            print(f"      null draw {i+1}/{n_null}: F={null_f[-1]:.4f}", flush=True)
    return f_obs, np.array(null_f)


def c1_auc_fast(X, pids, cids, rng_seed, log_every=1000):
    """Per-item AUC: same-cam vs diff-cam, matched negative counts."""
    n = len(X)
    rng = np.random.default_rng(rng_seed)
    # Per-item results
    auc_sc = np.full(n, np.nan)
    auc_dc = np.full(n, np.nan)
    item_matched = np.zeros(n, dtype=bool)
    # Precompute per-cam masks
    unique_c = np.unique(cids)
    print(f"    processing per camera, chunked...", flush=True)
    t0 = time.time()
    items_processed = 0
    for c in unique_c:
        mask_c = (cids == c)
        idx_c = np.where(mask_c)[0]
        idx_not_c = np.where(~mask_c)[0]
        items_c_X = X[idx_c]
        items_notc_X = X[idx_not_c]
        p_c = pids[idx_c]
        p_notc = pids[idx_not_c]
        n_c = len(idx_c); n_notc = len(idx_not_c)
        print(f"    cam {c}: n_c={n_c}  n_notc={n_notc}  elapsed={time.time() - t0:.1f}s", flush=True)
        for i0 in range(0, n_c, CHUNK):
            i1 = min(i0 + CHUNK, n_c)
            batch = items_c_X[i0:i1]
            # same-cam sims to all items in c (incl self)
            sims_cc = batch @ items_c_X.T  # (b, n_c)
            # diff-cam sims to items not in c
            sims_cnotc = batch @ items_notc_X.T  # (b, n_notc)
            for j in range(i1 - i0):
                i_local = i0 + j
                gi = idx_c[i_local]
                qp = p_c[i_local]
                sc_pos_mask = (p_c == qp); sc_pos_mask[i_local] = False
                sc_neg_mask = (p_c != qp)
                dc_pos_mask = (p_notc == qp)
                dc_neg_mask = (p_notc != qp)
                n_scp = int(sc_pos_mask.sum()); n_scn = int(sc_neg_mask.sum())
                n_dcp = int(dc_pos_mask.sum()); n_dcn = int(dc_neg_mask.sum())
                if n_scp < 1 or n_scn < 1 or n_dcp < 1 or n_dcn < 1:
                    continue
                item_matched[gi] = True
                n_match = min(n_scn, n_dcn)
                sc_neg_sel_idx = rng.choice(n_scn, size=n_match, replace=False)
                dc_neg_sel_idx = rng.choice(n_dcn, size=n_match, replace=False)
                sc_pos_s = sims_cc[j, sc_pos_mask]
                sc_neg_s = sims_cc[j, sc_neg_mask][sc_neg_sel_idx]
                dc_pos_s = sims_cnotc[j, dc_pos_mask]
                dc_neg_s = sims_cnotc[j, dc_neg_mask][dc_neg_sel_idx]
                auc_sc[gi] = auc_pos_vs_neg(sc_pos_s, sc_neg_s)
                auc_dc[gi] = auc_pos_vs_neg(dc_pos_s, dc_neg_s)
            items_processed += (i1 - i0)
            if items_processed % log_every < CHUNK:
                print(f"      items processed: {items_processed}  elapsed={time.time() - t0:.1f}s", flush=True)
    return auc_sc, auc_dc, item_matched


def aggregate_identity_auc(auc_sc, auc_dc, pids, item_matched):
    """Compute per-identity mean AUC (only for matched items), then mean across identities."""
    uids = np.unique(pids)
    id_sc = []; id_dc = []
    for p in uids:
        mask = (pids == p) & item_matched & ~np.isnan(auc_sc) & ~np.isnan(auc_dc)
        if mask.sum() == 0:
            continue
        id_sc.append(float(auc_sc[mask].mean()))
        id_dc.append(float(auc_dc[mask].mean()))
    return np.array(id_sc), np.array(id_dc)


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    n_cams = len(np.unique(g_cids))
    print(f"  n_gallery={len(g_pids)}  n_identities={len(np.unique(g_pids))}  n_cams={n_cams}", flush=True)

    # C1-null
    print(f"  C1-null ({N_NULL_DRAWS} draws)...", flush=True)
    f_obs, null_f = c1_null_fast(g_emb, g_pids, g_cids, N_NULL_DRAWS, RNG_SEED)
    null_mean = float(null_f.mean()); null_sd = float(null_f.std(ddof=1))
    null_ci = [float(np.quantile(null_f, 0.025)), float(np.quantile(null_f, 0.975))]
    null_min = float(null_f.min()); null_max = float(null_f.max())
    lift_total = f_obs - 0.0  # vs F_pooled baseline; we'll compute shares below
    print(f"  F_obs={f_obs:.4f}  null mean={null_mean:.4f}  sd={null_sd:.5f}  CI=[{null_ci[0]:.4f}, {null_ci[1]:.4f}]  range=[{null_min:.4f}, {null_max:.4f}]", flush=True)

    # Also recompute F_pooled from scratch for lift decomposition
    uids = np.unique(g_pids)
    mu_g = g_emb.mean(0)
    s_b_p = 0.0; s_w_p = 0.0
    for i in uids:
        mask = (g_pids == i); n_i = mask.sum()
        X_i = g_emb[mask]; mu_i = X_i.mean(0)
        s_b_p += n_i * np.sum((mu_i - mu_g) ** 2)
        s_w_p += np.sum((X_i - mu_i) ** 2)
    f_pooled = s_b_p / max(s_w_p, 1e-12)
    print(f"  F_pooled={f_pooled:.4f}  F_wcam - F_pooled (total lift)={f_obs - f_pooled:.4f}", flush=True)
    generic_lift = null_mean - f_pooled
    camera_specific_lift = f_obs - null_mean
    print(f"  lift decomposition: generic={generic_lift:.4f} ({generic_lift/(f_obs - f_pooled)*100:.1f}%)  camera-specific={camera_specific_lift:.4f} ({camera_specific_lift/(f_obs - f_pooled)*100:.1f}%)", flush=True)

    # C1-AUC
    print(f"  C1-AUC (fast per-camera chunked, logs every 1000)...", flush=True)
    auc_sc, auc_dc, matched = c1_auc_fast(g_emb, g_pids, g_cids, RNG_SEED)
    id_sc, id_dc = aggregate_identity_auc(auc_sc, auc_dc, g_pids, matched)
    matched_items = int(matched.sum())
    matched_ids = len(id_sc)
    sc_mean = float(id_sc.mean()) if len(id_sc) else np.nan
    dc_mean = float(id_dc.mean()) if len(id_dc) else np.nan
    sc_median = float(np.median(id_sc)) if len(id_sc) else np.nan
    dc_median = float(np.median(id_dc)) if len(id_dc) else np.nan
    diff_mean = sc_mean - dc_mean if not np.isnan(sc_mean) else np.nan
    print(f"  matched items={matched_items}/{len(g_pids)}  matched identities={matched_ids}", flush=True)
    print(f"  same-cam AUC: mean={sc_mean:.4f} median={sc_median:.4f}", flush=True)
    print(f"  diff-cam AUC: mean={dc_mean:.4f} median={dc_median:.4f}", flush=True)
    print(f"  diff mean (same-cam - diff-cam) = {diff_mean:+.4f}", flush=True)

    return {
        'cell': tag, 'label': info['label'], 'n_cams': n_cams,
        'F_wcam_obs': f_obs, 'F_pooled': f_pooled,
        'C1_null': {
            'mean': null_mean, 'sd': null_sd, 'ci': null_ci,
            'range': [null_min, null_max],
            'n_draws': N_NULL_DRAWS,
        },
        'lift_decomp': {
            'total_lift_F_wcam_minus_F_pooled': f_obs - f_pooled,
            'generic_lift_null_minus_pooled': generic_lift,
            'camera_specific_lift_obs_minus_null': camera_specific_lift,
            'generic_share_pct': float(generic_lift / (f_obs - f_pooled) * 100) if (f_obs - f_pooled) != 0 else 0.0,
            'camera_specific_share_pct': float(camera_specific_lift / (f_obs - f_pooled) * 100) if (f_obs - f_pooled) != 0 else 0.0,
        },
        'C1_AUC': {
            'matched_items': matched_items, 'matched_identities': matched_ids,
            'same_cam_auc_mean': sc_mean, 'same_cam_auc_median': sc_median,
            'diff_cam_auc_mean': dc_mean, 'diff_cam_auc_median': dc_median,
            'diff_mean': diff_mean,
        },
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)
    out_path = RESULTS / 'e43_C1_controls_fast.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= C1 CONTROLS FAST SUMMARY =========")
    print("\nC1-null + lift decomp:")
    print(f"{'cell':9s}  {'F_pooled':>9s}  {'F_wcam':>9s}  {'null':>9s}  {'gen share':>10s}  {'cam share':>10s}")
    for tag in CELLS:
        r = out[tag]; n = r['C1_null']; ld = r['lift_decomp']
        print(f"{tag:9s}  {r['F_pooled']:8.4f}  {r['F_wcam_obs']:8.4f}  {n['mean']:8.4f}  {ld['generic_share_pct']:9.1f}%  {ld['camera_specific_share_pct']:9.1f}%")

    print("\nC1-AUC (same-cam vs diff-cam, matched counts):")
    print(f"{'cell':9s}  {'n_items':>8s}  {'n_ids':>6s}  {'same-cam':>10s}  {'diff-cam':>10s}  {'diff':>8s}")
    for tag in CELLS:
        r = out[tag]['C1_AUC']
        print(f"{tag:9s}  {r['matched_items']:8d}  {r['matched_identities']:6d}  {r['same_cam_auc_mean']:10.4f}  {r['diff_cam_auc_mean']:10.4f}  {r['diff_mean']:+8.4f}")


if __name__ == '__main__':
    main()
