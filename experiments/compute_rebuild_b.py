"""
(b) rebuild: identity-controlled camera offsets (rev 319).

For each (identity i, camera c) present in gallery:
  mu_ic = mean of items with that (i, c)
  mu_i  = mean over items of identity i (item-weighted)
  delta_ic = mu_ic - mu_i   # identity-controlled offset
Then for each camera c:
  nu_c = mean over identities present in c of delta_ic
Centre {nu_c} at their mean and take top-r SVD -> B_cam (b).

Rerun on all four cells:
  - B1 own-B_cam projection + rank-matched partition surrogate (20 draws).
  - B2 oracle snap using nu_c as the lane position.
  - A2 angle between B_cam and id_top_r (MSMT only, both arms).
  - A3 per-lane margins using nu_c-projected positions as the lanes.

Identity-clustered bootstrap n=500 for every delta mAP CI.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
N_PART_DRAWS = 20
N_NULL_A2 = 1000
N_PAIR_A2 = 50000
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


def camera_offsets_b(X, pids, cids):
    """Construction (b): identity-controlled camera offsets.
    Returns nu_c (n_cams, d) and uc (list of unique camera ids in order)."""
    unique_pids = np.unique(pids)
    uc = np.unique(cids)
    # Accumulate (c, delta) pairs
    per_cam_deltas = {c: [] for c in uc}
    for p in unique_pids:
        mask_p = (pids == p)
        items_p = X[mask_p]
        if len(items_p) < 2:
            continue  # need at least 2 items to have any within-identity structure
        mu_i = items_p.mean(0)
        cams_p = cids[mask_p]
        for c in np.unique(cams_p):
            mu_ic = items_p[cams_p == c].mean(0)
            per_cam_deltas[c].append(mu_ic - mu_i)
    nu_c = []
    for c in uc:
        if len(per_cam_deltas[c]) == 0:
            nu_c.append(np.zeros(X.shape[1], dtype=X.dtype))
        else:
            nu_c.append(np.mean(np.stack(per_cam_deltas[c]), axis=0))
    nu_c = np.stack(nu_c)
    # Centre
    nu_c = nu_c - nu_c.mean(0)
    return nu_c, uc


def top_r_basis(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:r].T, s


def project_out(X, B):
    X_proj = X - (X @ B) @ B.T
    return l2(X_proj)


def identity_centroids_centered(X, pids):
    mu_g = X.mean(0)
    unique_p = np.unique(pids)
    return np.stack([X[pids == p].mean(0) - mu_g for p in unique_p])


def principal_angles(B1, B2):
    M = B1.T @ B2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


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


def boot_delta(aps_base, aps_mod, q_pids, n_boot=N_BOOT, rng_seed=RNG_SEED):
    rng = np.random.default_rng(rng_seed)
    valid = ~(np.isnan(aps_base) | np.isnan(aps_mod))
    b = aps_base[valid]; m = aps_mod[valid]; pid = q_pids[valid]
    uids = np.unique(pid); idx_by_pid = {u: np.where(pid == u)[0] for u in uids}
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(uids, size=len(uids), replace=True)
        idxs = np.concatenate([idx_by_pid[u] for u in sampled])
        diffs.append((m[idxs].mean() - b[idxs].mean()) * 100)
    diffs = np.array(diffs)
    return {
        'base_pct': float(np.nanmean(b)) * 100,
        'mod_pct': float(np.nanmean(m)) * 100,
        'delta_pp_point': float(np.nanmean(m) - np.nanmean(b)) * 100,
        'delta_pp_ci': [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
    }


def oracle_snap_b(X, cids, uc, nu_c, B_cam):
    """Snap each row's component along B_cam to its true cam's nu_c
    component along B_cam. Then L2-normalize.
    """
    cid_to_row = {c: i for i, c in enumerate(uc)}
    cam_rows = np.array([cid_to_row[c] for c in cids])
    cam_nu_per_item = nu_c[cam_rows]
    X_coords  = X @ B_cam
    nu_coords = cam_nu_per_item @ B_cam
    X_snap = X - X_coords @ B_cam.T + nu_coords @ B_cam.T
    return l2(X_snap)


def load_cell(info):
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    return q_emb, q_pids, q_cids, g_emb, g_pids, g_cids


# =========================================================================
# B1 (b): own-B_cam projection + rank-matched partition surrogate
# =========================================================================
def run_B1_b():
    print("\n========= B1 (b) =========", flush=True)
    out = {}
    for tag, info in CELLS.items():
        print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(info)
        n_cams = len(np.unique(g_cids)); r = n_cams - 1
        print(f"  n_cams={n_cams}  r={r}", flush=True)

        # (b) B_cam
        nu_c, uc = camera_offsets_b(g_emb, g_pids, g_cids)
        B_b, s_b = top_r_basis(nu_c, r)
        s_list = s_b.tolist()
        eff_rank = float((s_b ** 2).sum() ** 2 / ((s_b ** 2) ** 2).sum())
        print(f"  (b) nu_c sing. values: {np.round(s_b, 5)}", flush=True)
        print(f"  (b) eff rank = {eff_rank:.3f}", flush=True)

        # Baseline mAP
        aps_base = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        base = float(np.nanmean(aps_base)) * 100
        print(f"  baseline = {base:.4f}%", flush=True)

        # (b) projection
        q_p = project_out(q_emb, B_b); g_p = project_out(g_emb, B_b)
        aps_proj = per_query_ap(q_p, q_pids, q_cids, g_p, g_pids, g_cids)
        proj_res = boot_delta(aps_base, aps_proj, q_pids)
        print(f"  (b) B_cam projection: proj={proj_res['mod_pct']:.4f}%  delta={proj_res['delta_pp_point']:+.4f} pp  CI [{proj_res['delta_pp_ci'][0]:+.4f}, {proj_res['delta_pp_ci'][1]:+.4f}]", flush=True)

        # Rank-matched partition surrogate (random partition of ITEMS, not identity-aware;
        # same control as before -- it tests "remove any r directions of comparable size")
        cam_sizes = [int((g_cids == c).sum()) for c in np.unique(g_cids)]
        rng = np.random.default_rng(RNG_SEED)
        part_deltas = []
        mu_g = g_emb.mean(0)
        for di in range(N_PART_DRAWS):
            perm = rng.permutation(len(g_emb))
            idx = 0
            fake_offsets = []
            for n_imgs in cam_sizes:
                group = perm[idx:idx+n_imgs]
                idx += n_imgs
                fake_offsets.append(g_emb[group].mean(0) - mu_g)
            fake_offsets = np.stack(fake_offsets)
            B_fake, _ = top_r_basis(fake_offsets, r)
            q_p_f = project_out(q_emb, B_fake); g_p_f = project_out(g_emb, B_fake)
            aps_f = per_query_ap(q_p_f, q_pids, q_cids, g_p_f, g_pids, g_cids)
            d_pp = (float(np.nanmean(aps_f)) - float(np.nanmean(aps_base))) * 100
            part_deltas.append(d_pp)
        part_deltas = np.array(part_deltas)
        part_mean = float(part_deltas.mean())
        print(f"  partition surrogate mean delta = {part_mean:+.4f} pp  range [{part_deltas.min():+.4f}, {part_deltas.max():+.4f}]", flush=True)

        excess = proj_res['delta_pp_point'] - part_mean
        print(f"  (b) camera-specific excess = {excess:+.4f} pp", flush=True)

        out[tag] = {
            'label': info['label'], 'n_cams': n_cams, 'r': r,
            'nu_sv': s_list, 'eff_rank': eff_rank,
            **proj_res,
            'partition_mean_delta_pp': part_mean,
            'partition_range_pp': [float(part_deltas.min()), float(part_deltas.max())],
            'partition_per_draw_pp': part_deltas.tolist(),
            'camera_specific_excess_pp': excess,
        }
    return out


# =========================================================================
# B2 (b): oracle snap using nu_c as lane position
# =========================================================================
def run_B2_b():
    print("\n========= B2 (b) oracle snap =========", flush=True)
    out = {}
    for tag, info in CELLS.items():
        print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(info)
        n_cams = len(np.unique(g_cids)); r = n_cams - 1
        nu_c, uc = camera_offsets_b(g_emb, g_pids, g_cids)
        B_b, _ = top_r_basis(nu_c, r)

        aps_base = per_query_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        base = float(np.nanmean(aps_base)) * 100
        print(f"  baseline = {base:.4f}%", flush=True)

        q_snap = oracle_snap_b(q_emb, q_cids, uc, nu_c, B_b)
        g_snap = oracle_snap_b(g_emb, g_cids, uc, nu_c, B_b)
        aps_snap = per_query_ap(q_snap, q_pids, q_cids, g_snap, g_pids, g_cids)
        snap_res = boot_delta(aps_base, aps_snap, q_pids)
        print(f"  (b) oracle snap: snap={snap_res['mod_pct']:.4f}%  delta={snap_res['delta_pp_point']:+.4f} pp  CI [{snap_res['delta_pp_ci'][0]:+.4f}, {snap_res['delta_pp_ci'][1]:+.4f}]", flush=True)
        out[tag] = {'label': info['label'], **snap_res}
    return out


# =========================================================================
# A2 (b): MSMT-only, angle between B_cam (b) and id_top_r
# =========================================================================
def run_A2_b():
    print("\n========= A2 (b) =========", flush=True)
    R = 5
    cells_a2 = {
        'id':  ('bot_msmt_seed42_msmt.npz',  11659),
        'ood': ('msmt17_test_noREA.npz',     11659),
    }
    arm = {}
    for tag, (fname, n_q) in cells_a2.items():
        d = np.load(CACHE_DIR / fname); emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
        X = l2(emb[n_q:]); pid = pids[n_q:]; cid = cids[n_q:]
        arm[tag] = {'X': X, 'pids': pid, 'cids': cid}
        id_top_r, _ = top_r_basis(identity_centroids_centered(X, pid), R)
        nu_c, uc = camera_offsets_b(X, pid, cid)
        B_b, _ = top_r_basis(nu_c, R)
        angles = principal_angles(B_b, id_top_r)
        arm[tag]['id_top_r'] = id_top_r
        arm[tag]['B_b'] = B_b
        arm[tag]['obs_angle'] = float(np.median(angles))
        print(f"  arm={tag}: median angle between (b) B_cam and id_top_r = {arm[tag]['obs_angle']:.3f} deg", flush=True)
    shift = arm['ood']['obs_angle'] - arm['id']['obs_angle']
    print(f"  observed shift (ood - id) = {shift:+.3f} deg", flush=True)

    # Structure-matched null (same as rev 302): random 15-partition
    rng_id = np.random.default_rng(42); rng_ood = np.random.default_rng(1337)
    null_arms = {}
    for tag, rng in [('id', rng_id), ('ood', rng_ood)]:
        X = arm[tag]['X']; cid = arm[tag]['cids']
        cam_sizes = [int((cid == c).sum()) for c in np.unique(cid)]
        mu_g = X.mean(0)
        angs = []
        for _ in range(N_NULL_A2):
            perm = rng.permutation(len(X))
            idx = 0; fake = []
            for n_imgs in cam_sizes:
                group = perm[idx:idx+n_imgs]; idx += n_imgs
                fake.append(X[group].mean(0) - mu_g)
            fake = np.stack(fake)
            B_fake, _ = top_r_basis(fake, R)
            angs.append(float(np.median(principal_angles(B_fake, arm[tag]['id_top_r']))))
        null_arms[tag] = np.array(angs)
    rng_pair = np.random.default_rng(2026)
    idx_i = rng_pair.integers(0, N_NULL_A2, N_PAIR_A2)
    idx_j = rng_pair.integers(0, N_NULL_A2, N_PAIR_A2)
    shift_null = null_arms['ood'][idx_j] - null_arms['id'][idx_i]
    pct_below = float((shift_null < shift).mean())
    z = (shift - shift_null.mean()) / shift_null.std(ddof=1)
    print(f"  structure-matched shift null: mean={shift_null.mean():+.3f}  sd={shift_null.std(ddof=1):.3f}", flush=True)
    print(f"  excess = {shift - shift_null.mean():+.3f}  z = {z:+.3f}  empirical p = {pct_below:.4f}", flush=True)
    return {
        'id_angle': arm['id']['obs_angle'],
        'ood_angle': arm['ood']['obs_angle'],
        'shift': shift,
        'null_shift_mean': float(shift_null.mean()),
        'null_shift_sd': float(shift_null.std(ddof=1)),
        'excess': float(shift - shift_null.mean()),
        'z': float(z),
        'p_one_tail': pct_below,
    }


# =========================================================================
# A3 (b): per-lane margins using nu_c-projected items as lanes
# =========================================================================
def run_A3_b():
    """Lanes are the ν_c positions (not normalized the same way as A3 (a)).
    Keep full-D ν_c as the lane vector; compute cosine to each lane, then
    report nearest/second-nearest ratios in Euclidean and angular form.
    Note: ν_c is small-magnitude (identity-controlled residual), so cosine
    similarity may be dominated by angle between ν_c vectors, not by length.
    """
    print("\n========= A3 (b) per-lane margins =========", flush=True)
    out = {}
    for tag, info in CELLS.items():
        print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
        d = np.load(CACHE_DIR / info['file'])
        emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
        n_q = info['n_q']
        X_all = l2(emb); cids_all = cids
        g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
        nu_c, uc = camera_offsets_b(g_emb, g_pids, g_cids)
        # Lanes as unit vectors in direction of ν_c (for angular comparison on sphere)
        lane_norms = np.linalg.norm(nu_c, axis=1, keepdims=True)
        lanes = nu_c / np.maximum(lane_norms, 1e-12)

        sims = X_all @ lanes.T
        # Chordal distance = sqrt(2 - 2 sim)
        eucl = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))
        angles = np.degrees(np.arccos(np.clip(sims, -1.0, 1.0)))

        sorted_e = np.sort(eucl, axis=1); sorted_a = np.sort(angles, axis=1)
        near_e = sorted_e[:, 0]; sec_e = sorted_e[:, 1]
        near_a = sorted_a[:, 0]; sec_a = sorted_a[:, 1]
        ratio_e = near_e / np.maximum(sec_e, 1e-12)
        ratio_a = near_a / np.maximum(sec_a, 1e-12)

        med_e = float(np.median(ratio_e))
        med_a = float(np.median(ratio_a))
        rng = np.random.default_rng(42)
        N_B = 500
        e_b = [np.median(ratio_e[rng.integers(0, len(ratio_e), len(ratio_e))]) for _ in range(N_B)]
        a_b = [np.median(ratio_a[rng.integers(0, len(ratio_a), len(ratio_a))]) for _ in range(N_B)]
        e_ci = [float(np.quantile(e_b, 0.025)), float(np.quantile(e_b, 0.975))]
        a_ci = [float(np.quantile(a_b, 0.025)), float(np.quantile(a_b, 0.975))]

        cid_to_row = {c: i for i, c in enumerate(uc)}
        true_row = np.array([cid_to_row[c] for c in cids_all])
        acc_e = float((np.argmin(eucl, axis=1) == true_row).mean())
        acc_a = float((np.argmin(angles, axis=1) == true_row).mean())

        def verdict(r):
            if r >= 0.9: return "near-arbitrary (>=0.9)"
            if r <= 0.7: return "well-defined (<=0.7)"
            return "indeterminate (0.7-0.9)"

        print(f"  median nearest/second ratio (Euclidean) = {med_e:.4f}  CI [{e_ci[0]:.4f}, {e_ci[1]:.4f}]  -> {verdict(med_e)}", flush=True)
        print(f"  median nearest/second ratio (angular)   = {med_a:.4f}  CI [{a_ci[0]:.4f}, {a_ci[1]:.4f}]  -> {verdict(med_a)}", flush=True)
        print(f"  nearest-lane accuracy vs true cam: Euclidean={acc_e:.4f}  angular={acc_a:.4f}", flush=True)
        out[tag] = {
            'label': info['label'], 'n_cams': len(uc),
            'ratio_e': med_e, 'ratio_a': med_a, 'ci_e': e_ci, 'ci_a': a_ci,
            'acc_e': acc_e, 'acc_a': acc_a,
            'e_verdict': verdict(med_e), 'a_verdict': verdict(med_a),
        }
    return out


def main():
    out = {}
    out['B1_b'] = run_B1_b()
    out['B2_b'] = run_B2_b()
    out['A2_b'] = run_A2_b()
    out['A3_b'] = run_A3_b()

    out_path = RESULTS / 'e43_rebuild_b.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    print("\n\n========= (b) REBUILD SUMMARY =========")
    print("\nB1 (b): own B_cam projection vs partition surrogate")
    for tag in CELLS:
        r = out['B1_b'][tag]
        print(f"  {tag:9s}: base={r['base_pct']:.3f}%  (b)delta={r['delta_pp_point']:+.3f} pp CI [{r['delta_pp_ci'][0]:+.3f}, {r['delta_pp_ci'][1]:+.3f}]  partition={r['partition_mean_delta_pp']:+.3f}  excess={r['camera_specific_excess_pp']:+.3f} pp")

    print("\nB2 (b) oracle snap:")
    for tag in CELLS:
        r = out['B2_b'][tag]
        d_pp = r['delta_pp_point']
        band = "dead (<1)" if abs(d_pp) < 1 else ("beats-all (1-5)" if abs(d_pp) < 5 else "real (>5)")
        print(f"  {tag:9s}: base={r['base_pct']:.3f}%  delta={d_pp:+.3f} pp  CI [{r['delta_pp_ci'][0]:+.3f}, {r['delta_pp_ci'][1]:+.3f}]  ({band})")

    print("\nA2 (b) MSMT:")
    r = out['A2_b']
    print(f"  id_angle={r['id_angle']:.3f}  ood_angle={r['ood_angle']:.3f}  shift={r['shift']:+.3f}  excess={r['excess']:+.3f}  z={r['z']:+.3f}  p={r['p_one_tail']:.4f}")

    print("\nA3 (b):")
    for tag in CELLS:
        r = out['A3_b'][tag]
        print(f"  {tag:9s}: Eucl ratio={r['ratio_e']:.4f} ({r['e_verdict']})  Ang ratio={r['ratio_a']:.4f}  nearest-cam-acc Eucl={r['acc_e']:.4f}")


if __name__ == '__main__':
    main()
