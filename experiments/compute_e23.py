"""
E23 — Per-identity design matrix.

Build one table per cell: one row per test identity, ten geometric predictors,
plus per-ID AP as the target. All four cells. Five analyses in order:
  1. Spearman correlation matrix (10×10)
  2. PCA on standardised predictors (scree + 90%-variance components)
  3. Univariate Spearman of each predictor vs AP (bootstrap CI, n_boot=500)
  4. Multiple regression (RidgeCV + GBM) with variance partitioning
  5. In-domain vs OOD structural comparison (A vs C within Market; D vs B within MSMT17)

Predictors (per test identity):
  1. nearest_train_angle   : arccos(max cos sim from test ID centroid to any train ID centroid) [deg]
  2. mean_train_angle       : mean arccos over all train ID centroids [deg]
  3. nearest_test_angle     : arccos(max cos sim to any OTHER test ID centroid) [deg]
  4. overlap_ratio          : angular_radius / nearest_other_test_centroid_dist
                              angular_radius = 1 - min(cos(gallery_image, id_centroid))
                              nearest_dist   = 1 - max(cos between different test centroids)
  5. cos_id_global          : cos(unit id centroid, unit global gallery mean)
  6. within_id_spread_p75   : 75th pctile of arccos(gallery_image, id_centroid) [deg]
  7. within_id_spread_p90   : 90th pctile of arccos(gallery_image, id_centroid) [deg]
  8. within_id_spread_p95   : 95th pctile of arccos(gallery_image, id_centroid) [deg]
  9. mean_hub_score         : mean hub score of ID's gallery images (k-occ approx, k=10, N_HUB=5000)
 10. centroid_norm          : L2 norm of unnormalised gallery mean (before unit-normalising)
 11. n_gallery_imgs         : count of gallery images for that ID
 12. n_cameras              : count of unique cameras for that ID

Target: per_id_ap = mean AP over queries of that identity
        (canonical predicate: junk = same-pid & same-cam only)

Pairing rules:
  - Within eval set only: Market (A, C); MSMT17 (D, B)
  - 80/20 ID split for regression (sorted IDs, 80% train)
  - Bootstrap CIs over identities (not images), n_boot=500, RNG=42

Cache: /tmp/e12_feature_cache/, /tmp/e18_train_cache/
Output: /data/papers/low_dimensional_reid/results/e23_design_matrix.json
"""
import sys
sys.path.insert(0, "/workspace/reid-tk")

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

CACHE_DIR = Path("/tmp/e12_feature_cache")
TRAIN_DIR = Path("/tmp/e18_train_cache")
RESULTS   = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT   = 3368
N_Q_MSMT  = 11655
RNG       = np.random.default_rng(42)
N_BOOT    = 500
N_HUB     = 5000
K_HUB     = 10

PREDICTOR_NAMES = [
    "nearest_train_angle",
    "mean_train_angle",
    "nearest_test_angle",
    "overlap_ratio",
    "cos_id_global",
    "within_id_spread_p75",
    "within_id_spread_p90",
    "within_id_spread_p95",
    "mean_hub_score",
    "centroid_norm",
    "n_gallery_imgs",
    "n_cameras",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cell(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb  = d["emb"].astype(np.float32)
    pids = d["pids"]
    cids = d["cids"]
    return (emb[:n_q], pids[:n_q], cids[:n_q],
            emb[n_q:],  pids[n_q:], cids[n_q:])

def load_train(fname):
    d = np.load(TRAIN_DIR / fname)
    return d["emb"].astype(np.float32), d["pids"]


# ── Centroid utilities ────────────────────────────────────────────────────────

def compute_raw_centroids(emb, pids):
    """Returns (raw_centroids, unit_centroids, unique_ids, id2i)."""
    unique_ids = np.unique(pids)
    D = emb.shape[1]
    id2i = {uid: i for i, uid in enumerate(unique_ids)}
    raw = np.zeros((len(unique_ids), D), dtype=np.float64)
    cnt = np.zeros(len(unique_ids))
    for x, pid in zip(emb, pids):
        i = id2i[pid]; raw[i] += x.astype(np.float64); cnt[i] += 1
    raw /= cnt[:, None]
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = (raw / np.maximum(norms, 1e-9)).astype(np.float32)
    return raw.astype(np.float32), unit, unique_ids, id2i, norms.squeeze()


# ── Predictor 1: nearest_train_angle ─────────────────────────────────────────

def compute_nearest_train_angle(unit_gal, gal_ids, unit_train):
    """arccos(max cos sim to any train centroid). Returns array, one per gal ID."""
    cos_block = unit_gal.astype(np.float64) @ unit_train.astype(np.float64).T
    max_cos = np.clip(cos_block.max(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(max_cos))


# ── Predictor 2: mean_train_angle ────────────────────────────────────────────

def compute_mean_train_angle(unit_gal, unit_train, chunk=256):
    """mean arccos over all train centroids. Chunked to avoid large allocations."""
    n_gal  = unit_gal.shape[0]
    n_trn  = unit_train.shape[0]
    mean_angles = np.zeros(n_gal, dtype=np.float64)
    for s in range(0, n_gal, chunk):
        e = min(s + chunk, n_gal)
        cos_blk = unit_gal[s:e].astype(np.float64) @ unit_train.astype(np.float64).T
        angles_blk = np.degrees(np.arccos(np.clip(cos_blk, -1.0, 1.0)))
        mean_angles[s:e] = angles_blk.mean(axis=1)
    return mean_angles


# ── Predictor 3: nearest_test_angle ──────────────────────────────────────────

def compute_nearest_test_angle(unit_gal):
    """arccos(max cos sim to any OTHER test ID centroid)."""
    sim = unit_gal.astype(np.float64) @ unit_gal.astype(np.float64).T
    np.fill_diagonal(sim, -2.0)
    max_cos = np.clip(sim.max(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(max_cos))


# ── Predictor 4: overlap_ratio ───────────────────────────────────────────────

def compute_overlap_ratio(g_emb, g_pids, unit_gal_c, gal_ids, id2i):
    """angular_radius / nearest_other_test_centroid_dist (E20b definition)."""
    n_ids = len(gal_ids)
    # nearest other centroid distance
    sim = unit_gal_c.astype(np.float64) @ unit_gal_c.astype(np.float64).T
    np.fill_diagonal(sim, -2.0)
    nearest_sim  = sim.max(axis=1)
    nearest_dist = np.clip(1.0 - nearest_sim, 0, 2)
    # per-ID angular radius
    ang_radii = np.zeros(n_ids)
    for idx, uid in enumerate(gal_ids):
        mask = g_pids == uid
        X = g_emb[mask].astype(np.float64)
        u = unit_gal_c[idx].astype(np.float64)
        cos_sims = X @ u / np.maximum(np.linalg.norm(X, axis=1) * np.linalg.norm(u), 1e-9)
        ang_radii[idx] = float(1.0 - np.clip(cos_sims, -1, 1).min())
    return ang_radii / np.maximum(nearest_dist, 1e-9)


# ── Predictor 5: cos_id_global ────────────────────────────────────────────────

def compute_cos_id_global(unit_gal_c, g_emb):
    """cos(unit id centroid, unit global gallery mean)."""
    global_mean = g_emb.astype(np.float64).mean(axis=0)
    global_unit = global_mean / np.maximum(np.linalg.norm(global_mean), 1e-9)
    return (unit_gal_c.astype(np.float64) @ global_unit).astype(np.float32)


# ── Predictors 6-8: within_id_spread ─────────────────────────────────────────

def compute_within_id_spread(g_emb, g_pids, unit_gal_c, gal_ids):
    """75/90/95th percentile of arccos(gallery_image, id_centroid) [deg] per ID."""
    p75 = np.zeros(len(gal_ids))
    p90 = np.zeros(len(gal_ids))
    p95 = np.zeros(len(gal_ids))
    for idx, uid in enumerate(gal_ids):
        mask = g_pids == uid
        X = g_emb[mask].astype(np.float64)
        u = unit_gal_c[idx].astype(np.float64)
        cos_vals = X @ u / np.maximum(np.linalg.norm(X, axis=1) * np.linalg.norm(u), 1e-9)
        angles = np.degrees(np.arccos(np.clip(cos_vals, -1, 1)))
        p75[idx] = np.percentile(angles, 75)
        p90[idx] = np.percentile(angles, 90)
        p95[idx] = np.percentile(angles, 95)
    return p75, p90, p95


# ── Predictor 9: mean_hub_score ───────────────────────────────────────────────

def compute_hub_scores(g_emb, rng, n_hub=N_HUB, k=K_HUB, batch=500):
    """
    Approximate hub score per gallery image.
    Sample n_hub gallery items as "queries". For each, find k NNs from the
    full gallery. hub_count[i] = how many query items have i in their k-NN.
    Normalize by n_hub. Non-sampled items that never appear as NN get score 0.
    Returns dict {gallery_index -> hub_score}.
    """
    n_gal = g_emb.shape[0]
    sample_idx = rng.choice(n_gal, size=min(n_hub, n_gal), replace=False)
    hub_count = np.zeros(n_gal, dtype=np.float32)
    emb_sample = g_emb[sample_idx].astype(np.float32)
    n_sample = len(sample_idx)
    # Map sample positions for self-exclusion
    sample_set = set(sample_idx.tolist())
    for s in range(0, n_sample, batch):
        e = min(s + batch, n_sample)
        sim = emb_sample[s:e].astype(np.float32) @ g_emb.astype(np.float32).T
        # Exclude exact self
        for qi in range(e - s):
            orig_idx = int(sample_idx[s + qi])
            sim[qi, orig_idx] = -1e9
        top_k = np.argpartition(-sim, k, axis=1)[:, :k]
        for row in top_k:
            hub_count[row] += 1
    hub_count /= float(n_sample)
    return hub_count  # array of length n_gal

def mean_hub_score_per_id(hub_count, g_pids, gal_ids):
    """Mean hub score over gallery images of each ID."""
    return np.array([hub_count[g_pids == uid].mean() for uid in gal_ids])


# ── AP computation ────────────────────────────────────────────────────────────

def compute_per_id_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    """Returns dict uid -> per-ID mean AP."""
    n_q = len(q_pids)
    aps = []
    for s in range(0, n_q, batch):
        e = min(s + batch, n_q)
        sim = q_emb[s:e].astype(np.float32) @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi in range(e - s):
            qp, qc = q_pids[s+qi], q_cids[s+qi]
            gp = g_pids[order[qi]]; gc = g_cids[order[qi]]
            valid = ~((gp == qp) & (gc == qc))  # junk = same-pid & same-cam
            pos   = (gp == qp) & valid
            if pos.sum() == 0:
                aps.append(0.0); continue
            vi  = np.where(valid)[0]
            pi  = np.where(pos[vi])[0]
            prec = np.arange(1, len(pi)+1) / (pi + 1)
            aps.append(float(prec.mean()))
        if s % 5000 == 0:
            print(f"    AP rows {s}/{n_q}", flush=True)
    aps = np.array(aps)
    unique_ids = np.unique(q_pids)
    return {uid: float(aps[q_pids == uid].mean()) for uid in unique_ids}, float(aps.mean()*100)


# ── Statistics ────────────────────────────────────────────────────────────────

def boot_spearman(x, y, rng, n_boot=N_BOOT):
    n = len(x); rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rhos.append(float(spearmanr(x[idx], y[idx]).statistic))
    return np.array(rhos)

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x).astype(float), rankdata(y).astype(float), rankdata(z).astype(float)
    def resid(a, b):
        b_ = b - b.mean(); a_ = a - a.mean()
        slope = (a_ * b_).sum() / ((b_**2).sum() + 1e-12)
        return a - slope * b - (a.mean() - slope * b.mean())
    return float(pearsonr(resid(rx, rz), resid(ry, rz))[0])


# ── Analysis 1: Spearman correlation matrix ───────────────────────────────────

def analysis_corr_matrix(X, names):
    n_p = X.shape[1]
    mat = np.zeros((n_p, n_p))
    for i in range(n_p):
        for j in range(n_p):
            mat[i, j] = float(spearmanr(X[:, i], X[:, j]).statistic)
    print("  Spearman correlation matrix:", flush=True)
    hdr = "  " + "".join(f"{n[:8]:>10s}" for n in names)
    print(hdr, flush=True)
    for i, n in enumerate(names):
        row = "  " + f"{n[:14]:<14s}" + "".join(f"{mat[i,j]:>10.3f}" for j in range(n_p))
        print(row, flush=True)
    return mat.tolist()


# ── Analysis 2: PCA ───────────────────────────────────────────────────────────

def analysis_pca(X):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(random_state=42)
    pca.fit(X_std)
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    print(f"  PCA: components for 90% variance = {n90}", flush=True)
    print(f"  Scree: {' '.join(f'{v:.3f}' for v in evr)}", flush=True)
    return {
        "explained_variance_ratio": evr.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "n_components_90pct": n90,
    }


# ── Analysis 3: Univariate Spearman ──────────────────────────────────────────

def analysis_univariate(X, y, names, rng):
    results = {}
    print("  Univariate Spearman vs AP:", flush=True)
    for j, name in enumerate(names):
        x = X[:, j]
        rho = float(spearmanr(x, y).statistic)
        boots = boot_spearman(x, y, rng)
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        print(f"    {name:<28s}: rho={rho:+.4f}  95%CI [{ci[0]:+.4f},{ci[1]:+.4f}]", flush=True)
        results[name] = {"spearman": rho, "boot_ci_95": ci, "n": int(len(x))}
    return results


# ── Analysis 4: Multiple regression + variance partitioning ──────────────────

def analysis_regression(X, y, names, rng):
    n = len(y)
    unique_ids_idx = np.arange(n)
    # 80/20 split by sorted ID index (identities already sorted when building the matrix)
    n_train = int(0.8 * n)
    train_idx = unique_ids_idx[:n_train]
    test_idx  = unique_ids_idx[n_train:]

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Ridge
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000], cv=5)
    ridge.fit(X_tr_s, y_tr)
    r2_ridge = float(r2_score(y_te, ridge.predict(X_te_s)))
    print(f"  Ridge CV: alpha={ridge.alpha_:.3g}  held-out R²={r2_ridge:.4f}", flush=True)

    # GBM
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                    learning_rate=0.1, random_state=42)
    gbm.fit(X_tr, y_tr)
    r2_gbm = float(r2_score(y_te, gbm.predict(X_te)))
    importances = dict(zip(names, gbm.feature_importances_.tolist()))
    print(f"  GBM: held-out R²={r2_gbm:.4f}", flush=True)
    print("  GBM feature importances:", flush=True)
    for nm, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"    {nm:<28s}: {imp:.4f}", flush=True)

    # Variance partitioning: unique contribution = full_R² - R²_without_predictor
    vp = {}
    for j, name in enumerate(names):
        cols_keep = [k for k in range(X.shape[1]) if k != j]
        X_tr_drop = scaler.transform(X_tr)[:, cols_keep]
        X_te_drop = scaler.transform(X_te)[:, cols_keep]
        r = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000], cv=5)
        r.fit(X_tr_drop, y_tr)
        r2_drop = float(r2_score(y_te, r.predict(X_te_drop)))
        vp[name] = float(r2_ridge - r2_drop)
    print("  Variance partitioning (unique R² contribution):", flush=True)
    for nm, v in sorted(vp.items(), key=lambda x: -abs(x[1])):
        print(f"    {nm:<28s}: {v:+.4f}", flush=True)

    return {
        "ridge": {
            "alpha": float(ridge.alpha_),
            "held_out_r2": r2_ridge,
            "n_train": int(n_train),
            "n_test":  int(n - n_train),
        },
        "gbm": {
            "held_out_r2": r2_gbm,
            "feature_importances": importances,
        },
        "variance_partitioning_unique_r2": vp,
    }


# ── Analysis 5: In-domain vs OOD structural comparison ───────────────────────

def analysis_ood_comparison(cell_matrices):
    """
    cell_matrices: dict cell -> {"X": np.array, "y": np.array}
    Within Market: A (in-domain) vs C (OOD)
    Within MSMT17: D (in-domain) vs B (OOD)
    """
    comparisons = {}
    for (in_cell, ood_cell, label) in [("A", "C", "Market"), ("D", "B", "MSMT17")]:
        if in_cell not in cell_matrices or ood_cell not in cell_matrices:
            continue
        Xin, yin = cell_matrices[in_cell]["X"], cell_matrices[in_cell]["y"]
        Xoo, yoo = cell_matrices[ood_cell]["X"], cell_matrices[ood_cell]["y"]
        print(f"\n  === {label}: {in_cell} (in-domain) vs {ood_cell} (OOD) ===", flush=True)
        rows = []
        for j, name in enumerate(PREDICTOR_NAMES):
            rho_in = float(spearmanr(Xin[:, j], yin).statistic)
            rho_oo = float(spearmanr(Xoo[:, j], yoo).statistic)
            rows.append((name, rho_in, rho_oo, max(abs(rho_in), abs(rho_oo))))
        rows.sort(key=lambda x: -x[3])
        print(f"  {'Predictor':<28s}  {in_cell+' (in)':>10s}  {ood_cell+' (OOD)':>10s}", flush=True)
        comp = {}
        for name, rho_in, rho_oo, _ in rows:
            print(f"  {name:<28s}  {rho_in:>+10.4f}  {rho_oo:>+10.4f}", flush=True)
            comp[name] = {f"rho_{in_cell}_in": rho_in, f"rho_{ood_cell}_ood": rho_oo}
        comparisons[label] = comp
    return comparisons


# ── Per-cell feature matrix builder ──────────────────────────────────────────

def build_cell_matrix(cell, fname, n_q, tr_emb, tr_pids, rng):
    print(f"\n=== Cell {cell} ===", flush=True)
    q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(fname, n_q)

    print("  Computing centroids...", flush=True)
    raw_c, unit_c, gal_ids, id2i, raw_norms = compute_raw_centroids(g_emb, g_pids)

    # Training centroids
    print("  Computing train centroids...", flush=True)
    _, unit_tr, _, _, _ = compute_raw_centroids(tr_emb, tr_pids)

    print("  nearest_train_angle...", flush=True)
    nt_angle = compute_nearest_train_angle(unit_c, gal_ids, unit_tr)

    print("  mean_train_angle...", flush=True)
    mn_train_angle = compute_mean_train_angle(unit_c, unit_tr)

    print("  nearest_test_angle...", flush=True)
    nt_test_angle = compute_nearest_test_angle(unit_c)

    print("  overlap_ratio...", flush=True)
    ov_ratio = compute_overlap_ratio(g_emb, g_pids, unit_c, gal_ids, id2i)

    print("  cos_id_global...", flush=True)
    cos_global = compute_cos_id_global(unit_c, g_emb)

    print("  within_id_spread (p75/p90/p95)...", flush=True)
    p75, p90, p95 = compute_within_id_spread(g_emb, g_pids, unit_c, gal_ids)

    print("  hub scores...", flush=True)
    hub_count = compute_hub_scores(g_emb, rng)
    mhub = mean_hub_score_per_id(hub_count, g_pids, gal_ids)

    # centroid_norm
    c_norms = raw_norms.astype(np.float32)

    # n_gallery_imgs, n_cameras
    n_imgs = np.array([(g_pids == uid).sum() for uid in gal_ids], dtype=float)
    n_cams = np.array([len(np.unique(g_cids[g_pids == uid])) for uid in gal_ids], dtype=float)

    print("  computing per-ID AP...", flush=True)
    id_ap, mAP = compute_per_id_ap(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    print(f"  mAP={mAP:.4f}%", flush=True)

    # Intersect IDs: test IDs that have both gallery coverage and query coverage
    gal_id_set = set(gal_ids.tolist())
    qry_id_set = set(id_ap.keys())
    common_ids = sorted(gal_id_set & qry_id_set)
    # Index into gal_ids array
    gal_id_list = gal_ids.tolist()
    gal_id_to_idx = {uid: i for i, uid in enumerate(gal_id_list)}
    common_gal_idx = np.array([gal_id_to_idx[uid] for uid in common_ids])

    n = len(common_ids)
    print(f"  n_common_ids={n}", flush=True)

    X = np.column_stack([
        nt_angle[common_gal_idx],            # 0 nearest_train_angle
        mn_train_angle[common_gal_idx],      # 1 mean_train_angle
        nt_test_angle[common_gal_idx],       # 2 nearest_test_angle
        ov_ratio[common_gal_idx],            # 3 overlap_ratio
        cos_global[common_gal_idx],          # 4 cos_id_global
        p75[common_gal_idx],                 # 5 within_id_spread_p75
        p90[common_gal_idx],                 # 6 within_id_spread_p90
        p95[common_gal_idx],                 # 7 within_id_spread_p95
        mhub[common_gal_idx],               # 8 mean_hub_score
        c_norms[common_gal_idx],             # 9 centroid_norm
        n_imgs[common_gal_idx],              # 10 n_gallery_imgs
        n_cams[common_gal_idx],              # 11 n_cameras
    ])
    y = np.array([id_ap[uid] for uid in common_ids])

    return {
        "X": X, "y": y,
        "common_ids": common_ids,
        "mAP": mAP,
        "n": n,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading training embeddings...", flush=True)
mkt_tr_emb, mkt_tr_pids = load_train("bot_market_market_train.npz")
mst_tr_emb, mst_tr_pids = load_train("bot_msmt_msmt_train.npz")
print(f"  Market-train: {len(mkt_tr_pids)} imgs  "
      f"MSMT17-train: {len(mst_tr_pids)} imgs", flush=True)

CELL_DEFS = {
    "A": ("market_test.npz",            N_Q_MKT,  mkt_tr_emb, mkt_tr_pids),
    "C": ("bot_msmt_seed42_market.npz", N_Q_MKT,  mst_tr_emb, mst_tr_pids),
    "D": ("bot_msmt_seed42_msmt.npz",   N_Q_MSMT, mst_tr_emb, mst_tr_pids),
    "B": ("msmt17_test.npz",            N_Q_MSMT, mkt_tr_emb, mkt_tr_pids),
}

# Build a fresh RNG per cell to keep hub-score sampling deterministic per cell
cell_data = {}
for cell, (fname, n_q, tr_emb, tr_pids) in CELL_DEFS.items():
    cell_rng = np.random.default_rng(42)
    cell_data[cell] = build_cell_matrix(cell, fname, n_q, tr_emb, tr_pids, cell_rng)

results = {}
for cell, dat in cell_data.items():
    X, y = dat["X"], dat["y"]
    print(f"\n\n{'='*60}", flush=True)
    print(f"=== Analyses for Cell {cell}  (n={dat['n']}, mAP={dat['mAP']:.2f}%) ===", flush=True)
    print(f"{'='*60}", flush=True)

    print("\n--- Analysis 1: Spearman Correlation Matrix ---", flush=True)
    corr_mat = analysis_corr_matrix(X, PREDICTOR_NAMES)

    print("\n--- Analysis 2: PCA ---", flush=True)
    pca_res = analysis_pca(X)

    print("\n--- Analysis 3: Univariate Spearman vs AP ---", flush=True)
    univ_res = analysis_univariate(X, y, PREDICTOR_NAMES, np.random.default_rng(42))

    print("\n--- Analysis 4: Multiple Regression + Variance Partitioning ---", flush=True)
    reg_res = analysis_regression(X, y, PREDICTOR_NAMES, np.random.default_rng(42))

    results[cell] = {
        "mAP": dat["mAP"],
        "n_ids": dat["n"],
        "predictor_names": PREDICTOR_NAMES,
        "analysis_1_corr_matrix": corr_mat,
        "analysis_2_pca": pca_res,
        "analysis_3_univariate": univ_res,
        "analysis_4_regression": reg_res,
    }

print("\n--- Analysis 5: In-domain vs OOD Structural Comparison ---", flush=True)
ood_res = analysis_ood_comparison({c: {"X": cell_data[c]["X"], "y": cell_data[c]["y"]}
                                   for c in cell_data})

# Save
out = {
    "protocol": (
        "E23 per-identity design matrix. One row per test identity per cell. "
        "12 predictors (geometry + confounders), target = per-ID AP. "
        "Canonical predicate: junk = same-pid & same-cam only. "
        "Pairing within eval set only (Market: A,C; MSMT17: D,B). "
        "Bootstrap CI over identities (not images), n_boot=500, RNG=42. "
        "80/20 split by sorted ID for regression. "
        "Hub score: k-occurrence approx, k=10, N_HUB=5000 random gallery samples, RNG=42. "
        "mean_train_angle: mean arccos over all train centroids (chunked, chunk=256). "
        "variance_partitioning: unique contribution = full_Ridge_R² - R²_drop_one_predictor."
    ),
    "cell_defs": {
        "A": "BoT-Market model, Market eval, Market-train prototypes",
        "C": "BoT-MSMT17 model, Market eval, MSMT17-train prototypes",
        "D": "BoT-MSMT17 model, MSMT17 eval, MSMT17-train prototypes",
        "B": "BoT-Market model, MSMT17 eval, Market-train prototypes",
    },
    "predictor_names": PREDICTOR_NAMES,
    "seed": "RNG=42",
    "n_boot": N_BOOT,
    "n_hub": N_HUB,
    "k_hub": K_HUB,
    "cells": results,
    "analysis_5_ood_comparison": ood_res,
}

out_path = RESULTS / "e23_design_matrix.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
