"""
C1 - Within-camera identity separation (rev 323 + claude-chat's pre-reg).

For each of the four cells (A', B', C, D):
  (1) Standard pooled scatter decomposition:
      S_between_pooled = sum_i n_i ||mu_i - mu_global||^2
      S_within_pooled  = sum_{x in id i} ||x - mu_i||^2
      F_pooled = S_between_pooled / S_within_pooled

  (2) Within-camera scatter, pooled over cameras (camera held fixed):
      Per camera c:
        I_c = identities present in c
        mu_{i,c} = mean over items with (i, c)
        mu_c     = mean over all items in c
        S_between_c = sum_{i in I_c} n_{i,c} ||mu_{i,c} - mu_c||^2
        S_within_c  = sum_{x in (i,c)} ||x - mu_{i,c}||^2
      Pool: S_between_wcam = sum_c S_between_c
            S_within_wcam  = sum_c S_within_c
      F_wcam = S_between_wcam / S_within_wcam

  (3) Per-query same-camera AUC (geometry only; excludes same-pid-same-cam
      from retrieval but uses it deliberately here):
      For each query item x in gallery with cam c:
        pos_set = items (i=pid(x), c) excluding x
        neg_set = items (i!=pid(x), c)
        if |pos_set| >= 1 and |neg_set| >= 1:
          AUC_x = P(sim(x, pos) > sim(x, neg))
      Report mean and distribution.

Identity-clustered bootstrap for the F ratios, n=500.

Report label: geometry only, never as an mAP-adjacent number.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
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


def pooled_scatter(X, pids):
    mu_g = X.mean(0)
    uids = np.unique(pids)
    s_between = 0.0
    s_within = 0.0
    for i in uids:
        mask = (pids == i)
        n_i = mask.sum()
        X_i = X[mask]
        mu_i = X_i.mean(0)
        s_between += n_i * np.sum((mu_i - mu_g) ** 2)
        s_within += np.sum((X_i - mu_i) ** 2)
    return float(s_between), float(s_within)


def within_camera_scatter(X, pids, cids):
    """Per-camera scatter decomposition, then pool."""
    unique_c = np.unique(cids)
    s_between_c = {}
    s_within_c = {}
    for c in unique_c:
        mask_c = (cids == c)
        X_c = X[mask_c]; p_c = pids[mask_c]
        mu_c = X_c.mean(0)
        pids_in_c = np.unique(p_c)
        sb = 0.0; sw = 0.0
        for i in pids_in_c:
            mask_ic = (p_c == i)
            n_ic = mask_ic.sum()
            X_ic = X_c[mask_ic]
            mu_ic = X_ic.mean(0)
            sb += n_ic * np.sum((mu_ic - mu_c) ** 2)
            sw += np.sum((X_ic - mu_ic) ** 2)
        s_between_c[int(c)] = float(sb)
        s_within_c[int(c)] = float(sw)
    s_between_total = sum(s_between_c.values())
    s_within_total = sum(s_within_c.values())
    return s_between_total, s_within_total, s_between_c, s_within_c


def same_cam_auc(X, pids, cids):
    """Per-item same-camera AUC on cosine similarity.
    For each item x with cam c:
      pos_set = items in c with same pid, excluding self
      neg_set = items in c with different pid
    Compute AUC = P(sim(x, pos) > sim(x, neg)).
    Return mean, median, per-item array (only items with both sets non-empty).
    """
    n = len(X)
    aucs = np.full(n, np.nan)
    unique_c = np.unique(cids)
    sims_X = X @ X.T  # full similarity matrix - OK for Market (16k) but NOT for MSMT (94k)
    # Note: this is memory heavy for MSMT. Use per-camera slices to save memory.
    for c in unique_c:
        mask_c = (cids == c)
        idx_c = np.where(mask_c)[0]
        X_c = X[idx_c]
        p_c = pids[idx_c]
        sims_c = X_c @ X_c.T  # (n_c, n_c)
        for local_j, j in enumerate(idx_c):
            qp = p_c[local_j]
            pos_mask = (p_c == qp)
            pos_mask[local_j] = False  # exclude self
            neg_mask = (p_c != qp)
            if pos_mask.sum() < 1 or neg_mask.sum() < 1:
                continue
            s_pos = sims_c[local_j, pos_mask]
            s_neg = sims_c[local_j, neg_mask]
            # AUC = P(s_pos > s_neg); equivalent to Mann-Whitney U / (|pos||neg|)
            # Efficient: sort s_neg, for each s_pos count s_neg < s_pos
            s_neg_sorted = np.sort(s_neg)
            ranks_below = np.searchsorted(s_neg_sorted, s_pos, side='left')
            ranks_tied  = np.searchsorted(s_neg_sorted, s_pos, side='right') - ranks_below
            auc = (ranks_below.sum() + 0.5 * ranks_tied.sum()) / (len(s_pos) * len(s_neg))
            aucs[j] = auc
    return aucs


def same_cam_auc_memsafe(X, pids, cids):
    """Memory-safe version: never materialise full XX^T matrix."""
    n = len(X)
    aucs = np.full(n, np.nan)
    unique_c = np.unique(cids)
    for c in unique_c:
        mask_c = (cids == c)
        idx_c = np.where(mask_c)[0]
        X_c = X[idx_c]
        p_c = pids[idx_c]
        n_c = len(idx_c)
        sims_c = X_c @ X_c.T
        # Precompute pid groups
        unique_p_in_c = np.unique(p_c)
        for local_j in range(n_c):
            j = idx_c[local_j]
            qp = p_c[local_j]
            pos_mask = (p_c == qp)
            pos_mask[local_j] = False
            neg_mask = (p_c != qp)
            if pos_mask.sum() < 1 or neg_mask.sum() < 1:
                continue
            s_pos = sims_c[local_j, pos_mask]
            s_neg = sims_c[local_j, neg_mask]
            s_neg_sorted = np.sort(s_neg)
            ranks_below = np.searchsorted(s_neg_sorted, s_pos, side='left')
            ranks_tied  = np.searchsorted(s_neg_sorted, s_pos, side='right') - ranks_below
            auc = (ranks_below.sum() + 0.5 * ranks_tied.sum()) / (len(s_pos) * len(s_neg))
            aucs[j] = auc
    return aucs


def boot_f_ratios(X, pids, cids, n_boot, rng_seed):
    """Identity-clustered bootstrap of (F_pooled, F_wcam)."""
    rng = np.random.default_rng(rng_seed)
    uids = np.unique(pids)
    pid_idx = {int(i): np.where(pids == i)[0] for i in uids}
    pooled_boot = []
    wcam_boot = []
    for _ in range(n_boot):
        sampled = rng.choice(uids, size=len(uids), replace=True)
        idxs = np.concatenate([pid_idx[int(p)] for p in sampled])
        # Need unique pid labels for bootstrap (resampling identities can
        # create duplicates with same pid label, breaking the between-id
        # scatter). Give each sampled identity a fresh label.
        new_pids = np.empty(len(idxs), dtype=np.int64)
        offset = 0
        for new_label, p in enumerate(sampled):
            n_items_p = len(pid_idx[int(p)])
            new_pids[offset:offset+n_items_p] = new_label
            offset += n_items_p
        X_b = X[idxs]
        c_b = cids[idxs]
        sb_p, sw_p = pooled_scatter(X_b, new_pids)
        f_p = sb_p / max(sw_p, 1e-12)
        sb_w, sw_w, _, _ = within_camera_scatter(X_b, new_pids, c_b)
        f_w = sb_w / max(sw_w, 1e-12)
        pooled_boot.append(f_p); wcam_boot.append(f_w)
    return np.array(pooled_boot), np.array(wcam_boot)


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    n_cams = len(np.unique(g_cids))
    print(f"  n_gallery={len(g_pids)}  n_identities={len(np.unique(g_pids))}  n_cams={n_cams}", flush=True)

    # Pooled scatter
    sb_p, sw_p = pooled_scatter(g_emb, g_pids)
    f_pooled = sb_p / max(sw_p, 1e-12)
    print(f"  S_between_pooled={sb_p:.3f}  S_within_pooled={sw_p:.3f}  F_pooled={f_pooled:.4f}", flush=True)

    # Within-camera scatter
    sb_w, sw_w, sb_c, sw_c = within_camera_scatter(g_emb, g_pids, g_cids)
    f_wcam = sb_w / max(sw_w, 1e-12)
    print(f"  S_between_wcam ={sb_w:.3f}  S_within_wcam ={sw_w:.3f}  F_wcam  ={f_wcam:.4f}", flush=True)

    # Per-camera F
    per_cam_f = []
    for c in sb_c:
        f_c = sb_c[c] / max(sw_c[c], 1e-12)
        per_cam_f.append(f_c)
    per_cam_f = np.array(per_cam_f)
    print(f"  per-cam F: median={np.median(per_cam_f):.4f}  min={per_cam_f.min():.4f}  max={per_cam_f.max():.4f}  range={per_cam_f.tolist()}", flush=True)

    # Bootstrap CIs on F_pooled, F_wcam
    print(f"  bootstrapping F ratios (n={N_BOOT})...", flush=True)
    boot_p, boot_w = boot_f_ratios(g_emb, g_pids, g_cids, N_BOOT, RNG_SEED)
    ci_p = [float(np.quantile(boot_p, 0.025)), float(np.quantile(boot_p, 0.975))]
    ci_w = [float(np.quantile(boot_w, 0.025)), float(np.quantile(boot_w, 0.975))]
    print(f"  F_pooled CI: [{ci_p[0]:.4f}, {ci_p[1]:.4f}]", flush=True)
    print(f"  F_wcam   CI: [{ci_w[0]:.4f}, {ci_w[1]:.4f}]", flush=True)

    # Same-camera AUC
    print(f"  same-camera per-item AUC...", flush=True)
    aucs = same_cam_auc_memsafe(g_emb, g_pids, g_cids)
    valid = ~np.isnan(aucs)
    auc_mean = float(np.nanmean(aucs))
    auc_median = float(np.nanmedian(aucs))
    print(f"  same-cam AUC mean={auc_mean:.4f}  median={auc_median:.4f}  n_valid={valid.sum()}/{len(aucs)}", flush=True)

    return {
        'cell': tag,
        'label': info['label'],
        'n_cams': n_cams,
        'S_between_pooled': sb_p,
        'S_within_pooled': sw_p,
        'F_pooled': f_pooled,
        'F_pooled_ci': ci_p,
        'S_between_wcam': sb_w,
        'S_within_wcam': sw_w,
        'F_wcam': f_wcam,
        'F_wcam_ci': ci_w,
        'per_cam_F_between_over_within': per_cam_f.tolist(),
        'same_cam_auc_mean': auc_mean,
        'same_cam_auc_median': auc_median,
        'same_cam_auc_n_valid': int(valid.sum()),
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)
    out_path = RESULTS / 'e43_C1_within_cam_identity.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n========= C1 SUMMARY (within-camera identity separation) =========")
    print(f"{'cell':9s}  {'F_pooled':>9s}  {'F_wcam':>9s}  {'F_wcam/F_pooled':>17s}  {'same-cam AUC':>13s}")
    for tag in CELLS:
        r = out[tag]
        ratio = r['F_wcam'] / r['F_pooled']
        print(f"{tag:9s}  {r['F_pooled']:8.4f}  {r['F_wcam']:8.4f}  {ratio:16.4f}  {r['same_cam_auc_mean']:12.4f}")

    # Pair-wise id vs ood comparison per model
    print("\n\n  id vs ood per model (F_pooled and F_wcam):")
    for label, id_tag, ood_tag in [('Market model', 'A_prime', 'B_prime'), ('MSMT model', 'D', 'C')]:
        r_id = out[id_tag]; r_ood = out[ood_tag]
        p_ratio = r_ood['F_pooled'] / r_id['F_pooled']
        w_ratio = r_ood['F_wcam'] / r_id['F_wcam']
        auc_diff = r_ood['same_cam_auc_mean'] - r_id['same_cam_auc_mean']
        print(f"    {label}: F_pooled ood/id = {p_ratio:.4f}  F_wcam ood/id = {w_ratio:.4f}  same-cam AUC diff = {auc_diff:+.4f}")


if __name__ == '__main__':
    main()
