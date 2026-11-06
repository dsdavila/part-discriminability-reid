"""
E25 v2: vectorised AP-loss attribution budget.
Per spec from claude-chat rev 159.
"""
import sys; sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import json
from pathlib import Path

OUT   = Path('/data/papers/low_dimensional_reid/results')
CACHE = Path('/tmp/e12_feature_cache')

FILES = {
    'A': ('market_test.npz',            3368),
    'B': ('msmt17_test.npz',            11659),
    'C': ('bot_msmt_seed42_market.npz', 3368),
    'D': ('bot_msmt_seed42_msmt.npz',   11659),
}
RNG        = np.random.default_rng(42)
BOOT_N     = 500
REG_SAMPLE = 100_000

def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(n, 1e-12)).astype(np.float32)

def load_cell(cell):
    fname, nq = FILES[cell]
    d = np.load(CACHE / fname)
    emb  = l2(d['emb'].astype(np.float32))
    pids, cids = d['pids'], d['cids']
    return emb[:nq], pids[:nq], cids[:nq], emb[nq:], pids[nq:], cids[nq:]

def compute_centroids(emb, pids):
    unique = np.unique(pids)
    C = np.zeros((len(unique), emb.shape[1]), dtype=np.float32)
    for i, pid in enumerate(unique):
        c = emb[pids == pid].mean(0)
        C[i] = c / (np.linalg.norm(c) + 1e-12)
    return unique, C

def compute_hub_k10(q_emb, g_emb, g_pids, g_cids, q_pids, q_cids, batch=256):
    N_Q, N_G = len(q_emb), len(g_emb)
    counts = np.zeros(N_G, dtype=np.int32)
    for s in range(0, N_Q, batch):
        e = min(s + batch, N_Q)
        sim = q_emb[s:e] @ g_emb.T
        for bi in range(e - s):
            qi = s + bi
            sv = sim[bi].copy()
            sv[(g_pids == int(q_pids[qi])) & (g_cids == int(q_cids[qi]))] = -np.inf
            top10 = np.argpartition(sv, -10)[-10:]
            counts[top10] += 1
    return counts

def run_cell(cell, q_emb, q_pids, q_cids, g_emb, g_pids, g_cids,
             pid_list, C_mat, mu_hat, hub_scores,
             pair_angle_mat, sorted_pair_angles, n_pairs,
             batch=256):
    N_Q, N_G = len(q_emb), len(g_emb)
    TAGS = ['overlap', 'hub', 'cone', 'far_tail', 'outlier_q']

    cone_scores = (g_emb @ mu_hat).astype(np.float32)
    hub_sorted  = np.sort(hub_scores.astype(np.float32))
    cone_sorted = np.sort(cone_scores)

    pid_to_idx  = {int(p): i for i, p in enumerate(pid_list)}
    gal_pid_idx = np.array([pid_to_idx.get(int(p), -1) for p in g_pids])

    safe_idx    = np.maximum(gal_pid_idx, 0)
    gal_cent_cos = np.clip(
        np.einsum('nd,nd->n', g_emb, C_mat[safe_idx]), -1.0, 1.0)
    gal_cent_cos[gal_pid_idx < 0] = 0.0
    gal_cent_angle  = np.degrees(np.arccos(gal_cent_cos)).astype(np.float32)
    gal_cent_sorted = np.sort(gal_cent_angle)

    quint_sum = {t: np.zeros(5) for t in TAGS}
    ap_list   = []
    sigma_list = []
    G_list    = []
    reg_buf   = []

    for s in range(0, N_Q, batch):
        e   = min(s + batch, N_Q)
        sim = (q_emb[s:e] @ g_emb.T).astype(np.float32)

        for bi in range(e - s):
            qi   = s + bi
            qpid = int(q_pids[qi])
            qcid = int(q_cids[qi])

            junk_mask    = (g_pids == qpid) & (g_cids == qcid)
            correct_mask = (g_pids == qpid) & (~junk_mask)
            G = int(correct_mask.sum())
            if G == 0: continue

            sv = sim[bi].copy()
            sv[junk_mask] = -np.inf
            order = np.argsort(-sv)

            ranked_pids = g_pids[order]
            valid_flags = sv[order] > -1e30

            correct_pos = np.where((ranked_pids == qpid) & valid_flags)[0]
            if len(correct_pos) == 0: continue
            r   = correct_pos + 1
            r_G = int(r[-1])

            offender_pos = np.where(
                (ranked_pids != qpid) & valid_flags & (np.arange(N_G) < r_G)
            )[0]

            AP = float(np.sum(np.arange(1, G+1) / r)) / G
            ap_list.append(AP)
            G_list.append(G)

            if len(offender_pos) == 0:
                sigma_list.append(0.0)
                continue

            # Vectorised ΔAP
            j_arr   = np.arange(1, G+1, dtype=np.float32)
            contrib = np.where(r > 1,
                               j_arr / (r.astype(np.float32) * (r.astype(np.float32)-1)),
                               0.0)
            cum_contrib = np.cumsum(contrib[::-1])[::-1]

            idxs = np.searchsorted(r, offender_pos + 1, 'right')
            valid_off = idxs < G
            dap_arr = np.zeros(len(offender_pos), dtype=np.float32)
            dap_arr[valid_off] = cum_contrib[idxs[valid_off]] / G

            sigma = float(dap_arr.sum())
            sigma_list.append(sigma)
            if sigma == 0: continue

            offender_g_idx = order[offender_pos]

            # Overlap tag
            q_pid_i   = pid_to_idx.get(qpid, -1)
            off_pid_i = gal_pid_idx[offender_g_idx]
            if q_pid_i >= 0 and n_pairs > 0:
                valid_off_pid = off_pid_i >= 0
                pair_a = np.where(
                    valid_off_pid,
                    pair_angle_mat[q_pid_i, np.maximum(off_pid_i, 0)],
                    90.0)
                ov_pct = np.searchsorted(sorted_pair_angles, pair_a).astype(np.float32) / n_pairs
            else:
                ov_pct = np.full(len(offender_pos), 0.5, dtype=np.float32)
            ov_pct = np.clip(ov_pct, 0.0, 1.0)

            # Hub tag
            hub_v   = hub_scores[offender_g_idx].astype(np.float32)
            hub_pct = np.searchsorted(hub_sorted, hub_v).astype(np.float32) / N_G

            # Cone tag
            cone_v   = cone_scores[offender_g_idx]
            cone_pct = np.searchsorted(cone_sorted, cone_v).astype(np.float32) / N_G

            # Far-tail tag (vectorised suffix mean)
            correct_g_idx = order[correct_pos]
            c_angles = gal_cent_angle[correct_g_idx].astype(np.float32)
            suffix_sum   = np.cumsum(c_angles[::-1])[::-1]
            suffix_count = np.arange(G, 0, -1, dtype=np.float32)
            suffix_mean  = suffix_sum / suffix_count
            ft_angles = np.where(
                valid_off,
                suffix_mean[np.minimum(idxs, G-1)],
                float(gal_cent_sorted[N_G // 2])
            ).astype(np.float32)
            ft_pct = np.searchsorted(gal_cent_sorted, ft_angles).astype(np.float32) / N_G

            # Outlier-query tag
            if q_pid_i >= 0:
                qu = q_emb[qi] / (np.linalg.norm(q_emb[qi]) + 1e-12)
                angle_qc = float(np.degrees(np.arccos(np.clip(qu @ C_mat[q_pid_i], -1.0, 1.0))))
            else:
                angle_qc = 45.0
            oq_pct = float(np.searchsorted(gal_cent_sorted, angle_qc)) / N_G
            oq_arr = np.full(len(offender_pos), oq_pct, dtype=np.float32)

            # Quintile accumulation
            for ti, t in enumerate(TAGS):
                pcts = [ov_pct, hub_pct, cone_pct, ft_pct, oq_arr][ti]
                bins = np.clip(np.floor(pcts * 5).astype(int), 0, 4)
                for b in range(5):
                    quint_sum[t][b] += float(dap_arr[bins == b].sum())

            # Regression buffer
            items = np.column_stack([ov_pct, hub_pct, cone_pct, ft_pct, oq_arr, dap_arr])
            reg_buf.append(items.astype(np.float32))

        if (s // batch) % 20 == 0:
            print(f"  {min(e,N_Q)}/{N_Q} queries done", flush=True)

    # Normalise
    for t in TAGS:
        tot = quint_sum[t].sum()
        if tot > 0: quint_sum[t] /= tot

    if reg_buf:
        reg_arr = np.vstack(reg_buf)
        if len(reg_arr) > REG_SAMPLE:
            idx = RNG.choice(len(reg_arr), REG_SAMPLE, replace=False)
            reg_arr = reg_arr[idx]
    else:
        reg_arr = np.zeros((0, 6), dtype=np.float32)

    return (np.array(ap_list), np.array(sigma_list), np.array(G_list),
            {t: quint_sum[t].tolist() for t in TAGS}, reg_arr)


def variance_partition(reg_arr):
    if len(reg_arr) < 200: return None
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    TAGS = ['overlap', 'hub', 'cone', 'far_tail', 'outlier_q']
    X = reg_arr[:, :5].astype(np.float64)
    y = reg_arr[:,  5].astype(np.float64)
    pairs = [(i,j) for i in range(5) for j in range(i+1,5)]
    Xint = np.column_stack([X[:,i]*X[:,j] for i,j in pairs])
    Xf = np.column_stack([X, Xint])
    sc = StandardScaler(); Xf = sc.fit_transform(Xf)
    model = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5).fit(Xf, y)
    yp = model.predict(Xf)
    ss = np.var(y) * len(y) + 1e-12
    r2_full = float(1 - np.sum((y-yp)**2) / ss)
    uniq = {}
    for ti, tag in enumerate(TAGS):
        keep_main = [j for j in range(5) if j != ti]
        keep_int  = [5+k for k, (a,b) in enumerate(pairs) if ti not in (a,b)]
        keep = keep_main + keep_int
        m2 = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5).fit(Xf[:,keep], y)
        r2_sub = float(1 - np.sum((y - m2.predict(Xf[:,keep]))**2) / ss)
        uniq[tag] = float(r2_full - r2_sub)
    return {'r2_full': r2_full, 'residual': float(1-r2_full),
            'unique_r2': uniq, 'n': int(len(y))}


# ─── MAIN ──────────────────────────────────────────────────────────────────
print("=== E25 v2: AP-loss attribution budget ===\n", flush=True)

out_path = OUT / 'e25_attribution_results.json'
existing = {}
if out_path.exists():
    with open(out_path) as f:
        existing = json.load(f)
    print(f"Loaded existing results: cells {list(existing.keys())}\n", flush=True)

results = dict(existing)

for cell in ['A', 'C', 'D', 'B']:
    if cell in results:
        print(f"Cell {cell}: already done, skipping.\n", flush=True)
        continue
    print(f"\n{'='*55}\nCell {cell}\n{'='*55}", flush=True)
    q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(cell)
    N_Q, N_G = len(q_emb), len(g_emb)
    print(f"  Q={N_Q}, G={N_G}", flush=True)

    print("  Building centroids...", flush=True)
    pid_list, C_mat = compute_centroids(g_emb, g_pids)
    n_ids = len(pid_list)
    mu     = g_emb.mean(0)
    mu_hat = (mu / (np.linalg.norm(mu) + 1e-12)).astype(np.float32)

    print("  Building centroid Gram matrix...", flush=True)
    Gg = np.clip(C_mat @ C_mat.T, -1.0, 1.0)
    pair_angle_mat = np.degrees(np.arccos(Gg)).astype(np.float32)
    np.fill_diagonal(pair_angle_mat, 0.0)
    rows, cols = np.tril_indices(n_ids, k=-1)
    sorted_pair_angles = np.sort(pair_angle_mat[rows, cols])
    n_pairs = len(sorted_pair_angles)
    print(f"  n_id_pairs={n_pairs:,}", flush=True)

    print("  Computing hub scores (k=10)...", flush=True)
    hub_scores = compute_hub_k10(q_emb, g_emb, g_pids, g_cids, q_pids, q_cids)
    print(f"  Hub mean={hub_scores.mean():.1f} max={hub_scores.max()}", flush=True)

    print("  Attribution loop...", flush=True)
    ap_arr, sigma_arr, G_arr, quint, reg_arr = run_cell(
        cell, q_emb, q_pids, q_cids, g_emb, g_pids, g_cids,
        pid_list, C_mat, mu_hat, hub_scores,
        pair_angle_mat, sorted_pair_angles, n_pairs
    )

    mean_ap     = float(ap_arr.mean() * 100) if len(ap_arr) else 0.0
    total_def   = float((1 - ap_arr).sum()) if len(ap_arr) else 0.0
    total_sigma = float(sigma_arr.sum())
    superadd    = float(total_sigma / total_def) if total_def > 0 else None

    print(f"\n  mAP={mean_ap:.2f}%  Σ(1-AP)={total_def:.3f}  ΣΔAP={total_sigma:.3f}  ratio={superadd:.3f}")
    print("  Quintile shares [Q1..Q5 = low→high pctile]:")
    for t, v in quint.items():
        print(f"    {t:12s}: {[f'{x:.3f}' for x in v]}")

    print("  Variance partitioning...", flush=True)
    vp = variance_partition(reg_arr)
    if vp:
        print(f"  R²={vp['r2_full']:.4f}  residual={vp['residual']:.4f}  n={vp['n']}")
        print(f"  Unique R²: {vp['unique_r2']}")

    if len(ap_arr) >= 10:
        ba = [float(RNG.choice(ap_arr, len(ap_arr), replace=True).mean()) for _ in range(BOOT_N)]
        bs = [float(RNG.choice(sigma_arr, len(sigma_arr), replace=True).mean()) for _ in range(BOOT_N)]
        ci_ap = [float(np.percentile(ba, 2.5)), float(np.percentile(ba, 97.5))]
        ci_s  = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    else:
        ci_ap = ci_s = [None, None]

    results[cell] = {
        'N_Q': N_Q, 'N_G': N_G, 'n_gal_ids': int(n_ids),
        'mean_ap_pct': mean_ap,
        'total_ap_deficit': total_def, 'total_sigma_dap': total_sigma,
        'super_additivity': superadd,
        'quint_bins': quint,
        'variance_partition': vp,
        'G_stats': {k: float(f(G_arr)) for k, f in [
            ('mean', np.mean), ('median', np.median),
            ('p10', lambda x: np.percentile(x, 10)),
            ('p90', lambda x: np.percentile(x, 90)),
        ]} if len(G_arr) else {},
        'boot_ci_mean_ap': ci_ap,
        'boot_ci_sigma': ci_s,
    }

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Checkpoint saved → {out_path}", flush=True)

print("\n=== Summary ===")
for cell in ['A','C','D','B']:
    if cell not in results: continue
    r = results[cell]
    vp = r.get('variance_partition') or {}
    print(f"  Cell {cell}: mAP={r['mean_ap_pct']:.2f}%  ratio={r['super_additivity']:.3f}  R²={vp.get('r2_full','n/a')}")
