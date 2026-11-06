"""
E20c — Angular cap intersection. Answers Daniel's three questions in order.
Protocol: angular cap = centroid direction u_i + angular radius r_i at pctile p of arccos(x.u_i).
Intersection: arccos(u_i.u_j) < r_i + r_j. Containment: arccos(u_i.u_j) + r_i < r_j.
Report for p in [75, 90, 95] (sensitivity).

Prototype source (within-pair, same model AND same prototype set, per corrected E18 Phase 2b):
  A/B pair (BoT-Market): training prototypes = Market-train under BoT-Market
    -> bot_market_market_train.npz (751 training IDs)
  C/D pair (BoT-MSMT17): training prototypes = MSMT17-train under BoT-MSMT17
    -> bot_msmt_msmt_train.npz (1041 training IDs)

Q1: Do test-ID caps intersect within the test set? All four cells.
Q2: Do test-ID caps intersect with training-ID caps (within-pair only)?
Q3: Three-way histogram per test ID: inside-exactly-one / intersects-several / intersects-none training caps.
    + deepest intersection depth.
Also: target-vs-target intersection (Q1 is this), plus target-vs-training (Q2/Q3).

Seed: RNG=42 (not used — deterministic). Cache: e12_feature_cache + e18_train_cache.
"""
import numpy as np, json
from pathlib import Path

CACHE_DIR = Path("/tmp/e12_feature_cache")
TRAIN_CACHE_DIR = Path("/tmp/e18_train_cache")
RESULTS_DIR = Path("/data/papers/low_dimensional_reid/results")
N_Q_MKT = 3368
N_Q_MSMT = 11655
PCTILES = [75, 90, 95]  # angular radius percentiles for sensitivity


def load_gallery(fname, n_query):
    d = np.load(CACHE_DIR / fname)
    g_emb = d["emb"][n_query:].astype(np.float32)
    g_pids = d["pids"][n_query:]
    return g_emb, g_pids


def load_train(fname):
    d = np.load(TRAIN_CACHE_DIR / fname)
    return d["emb"].astype(np.float32), d["pids"]


def build_id_caps(emb, pids, pctiles=PCTILES):
    """
    Per identity: unit centroid u_i, angular radii at requested percentiles.
    Returns: unit_cents (n_ids, D), radii_dict {pctile: array(n_ids)}, unique_ids.
    """
    unique_ids = np.unique(pids)
    n_ids = len(unique_ids)
    D = emb.shape[1]
    id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}

    # Accumulate centroids
    centroids = np.zeros((n_ids, D), dtype=np.float64)
    counts = np.zeros(n_ids)
    for x, pid in zip(emb, pids):
        idx = id_to_idx[pid]
        centroids[idx] += x.astype(np.float64)
        counts[idx] += 1
    centroids /= counts[:, None]
    cent_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    unit_cents = (centroids / cent_norms).astype(np.float32)

    # Angular radii
    radii = {p: np.zeros(n_ids) for p in pctiles}
    for id_idx, uid in enumerate(unique_ids):
        mask = pids == uid
        X = emb[mask].astype(np.float64)
        u = unit_cents[id_idx].astype(np.float64)
        norms_X = np.linalg.norm(X, axis=1)
        cos_sims = np.clip((X @ u) / np.maximum(norms_X, 1e-8), -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_sims))
        for p in pctiles:
            radii[p][id_idx] = float(np.percentile(angles, p))

    return unit_cents, radii, unique_ids


def pairwise_intersection_stats(unit_cents_A, radii_A, unit_cents_B, radii_B, pctile):
    """
    Compute intersection matrix between set A and set B.
    Intersection: arccos(u_i . u_j) < r_i + r_j  (in degrees).
    Containment of A in B: arccos(u_i . u_j) + r_i < r_j.
    Returns: intersection rate, containment-A-in-B rate, per-A intersection counts.
    """
    # Cosine similarity matrix
    cos_AB = unit_cents_A @ unit_cents_B.T  # (n_A, n_B)
    cos_AB = np.clip(cos_AB, -1.0, 1.0)
    angle_AB = np.degrees(np.arccos(cos_AB))  # (n_A, n_B) in degrees

    r_A = radii_A[pctile][:, None]   # (n_A, 1)
    r_B = radii_B[pctile][None, :]   # (1, n_B)

    if unit_cents_A is unit_cents_B:
        # Self-comparison: exclude diagonal
        np.fill_diagonal(angle_AB, 1e9)

    intersects = angle_AB < (r_A + r_B)
    contains_A_in_B = (angle_AB + r_A) < r_B  # A fully inside B

    n_A = len(r_A)
    inter_counts = intersects.sum(axis=1)  # per-A intersection count

    return {
        "intersection_rate": float(intersects.any(axis=1).mean()),
        "containment_rate_A_in_B": float(contains_A_in_B.any(axis=1).mean()),
        "mean_intersection_count": float(inter_counts.mean()),
        "median_intersection_count": float(np.median(inter_counts)),
        "intersects_none_frac": float((inter_counts == 0).mean()),
        "intersects_exactly_one_frac": float((inter_counts == 1).mean()),
        "intersects_several_frac": float((inter_counts > 1).mean()),
        "max_intersection_count": int(inter_counts.max()),
        # Deepest intersection: how much does the tightest cap fit inside another?
        # depth = max over j of (r_B[j] - angle_AB[i,j] - r_A[i]) for each i
        "mean_max_depth_deg": float(np.mean(np.maximum(0, r_B - angle_AB - r_A).max(axis=1))),
    }


print("Loading gallery embeddings...", flush=True)
A_emb, A_pids = load_gallery("market_test.npz",           N_Q_MKT)
B_emb, B_pids = load_gallery("msmt17_test.npz",           N_Q_MSMT)
C_emb, C_pids = load_gallery("bot_msmt_seed42_market.npz", N_Q_MKT)
D_emb, D_pids = load_gallery("bot_msmt_seed42_msmt.npz",  N_Q_MSMT)

print("Loading training embeddings...", flush=True)
# A/B pair uses BoT-Market on Market-train; C/D pair uses BoT-MSMT17 on MSMT17-train
mkt_train_emb, mkt_train_pids = load_train("bot_market_market_train.npz")
msmt_train_emb, msmt_train_pids = load_train("bot_msmt_msmt_train.npz")

print(f"Market-train: {len(mkt_train_emb)} images, {len(np.unique(mkt_train_pids))} IDs", flush=True)
print(f"MSMT17-train: {len(msmt_train_emb)} images, {len(np.unique(msmt_train_pids))} IDs", flush=True)

print("\nBuilding ID caps...", flush=True)
caps = {}
for label, emb, pids in [("A", A_emb, A_pids), ("B", B_emb, B_pids),
                           ("C", C_emb, C_pids), ("D", D_emb, D_pids)]:
    print(f"  Cell {label}: {len(emb)} images, {len(np.unique(pids))} IDs", flush=True)
    caps[label] = build_id_caps(emb, pids)

print("  Market-train...", flush=True)
caps["mkt_train"] = build_id_caps(mkt_train_emb, mkt_train_pids)
print("  MSMT17-train...", flush=True)
caps["msmt_train"] = build_id_caps(msmt_train_emb, msmt_train_pids)

results = {}

# ====== Q1: Test-ID vs Test-ID (within-cell) ======
print("\n=== Q1: Test-ID vs Test-ID cap intersection ===", flush=True)
results["Q1_test_vs_test"] = {}
for label in ["A", "B", "C", "D"]:
    unit_cents, radii, _ = caps[label]
    print(f"  Cell {label}: {len(unit_cents)} IDs", flush=True)
    results["Q1_test_vs_test"][label] = {}
    for p in PCTILES:
        stats = pairwise_intersection_stats(unit_cents, radii, unit_cents, radii, p)
        results["Q1_test_vs_test"][label][f"p{p}"] = stats
        print(f"    p{p}: inter_rate={stats['intersection_rate']:.3f}  none={stats['intersects_none_frac']:.3f}  exactly1={stats['intersects_exactly_one_frac']:.3f}  several={stats['intersects_several_frac']:.3f}", flush=True)

# ====== Q2/Q3: Test-ID vs Training-ID (within-pair) ======
print("\n=== Q2/Q3: Test-ID vs Training-ID (within-pair) ===", flush=True)
print("  Prototype source: A/B -> bot_market_market_train (Market-train, BoT-Market)", flush=True)
print("  Prototype source: C/D -> bot_msmt_msmt_train (MSMT17-train, BoT-MSMT17)", flush=True)

# A/B pair: test vs Market-train prototypes
results["Q2Q3_test_vs_train"] = {}
train_u_mkt, train_r_mkt, _ = caps["mkt_train"]
train_u_msmt, train_r_msmt, _ = caps["msmt_train"]

for label, train_u, train_r in [("A", train_u_mkt, train_r_mkt),
                                  ("B", train_u_mkt, train_r_mkt),
                                  ("C", train_u_msmt, train_r_msmt),
                                  ("D", train_u_msmt, train_r_msmt)]:
    test_u, test_r, _ = caps[label]
    print(f"\n  Cell {label} ({len(test_u)} test IDs vs {len(train_u)} train IDs):", flush=True)
    results["Q2Q3_test_vs_train"][label] = {}
    for p in PCTILES:
        stats = pairwise_intersection_stats(test_u, test_r, train_u, train_r, p)
        results["Q2Q3_test_vs_train"][label][f"p{p}"] = stats
        print(f"    p{p}: inter_rate={stats['intersection_rate']:.3f}  none={stats['intersects_none_frac']:.3f}  exactly1={stats['intersects_exactly_one_frac']:.3f}  several={stats['intersects_several_frac']:.3f}  containment={stats['containment_rate_A_in_B']:.3f}", flush=True)

# ====== Summary table at p90 ======
print("\n=== SUMMARY (p90) ===", flush=True)
print(f"{'Cell':>5} {'test-test inter':>16} {'test-test none':>14} {'test-train inter':>17} {'test-train none':>16} {'containment':>12}", flush=True)
for label in ["A","B","C","D"]:
    q1 = results["Q1_test_vs_test"][label]["p90"]
    q3 = results["Q2Q3_test_vs_train"][label]["p90"]
    print(f"  {label:>3}  {q1['intersection_rate']:>16.3f} {q1['intersects_none_frac']:>14.3f} {q3['intersection_rate']:>17.3f} {q3['intersects_none_frac']:>16.3f} {q3['containment_rate_A_in_B']:>12.3f}", flush=True)

out = {
    "protocol": "angular cap: centroid direction u_i + angular radius r_i at percentile p of arccos(x.u_i) in degrees; intersection: arccos(u_i.u_j) < r_i+r_j; containment of test in train: arccos(u_i.u_j)+r_i < r_j",
    "prototype_source": {
        "A_and_B": "Market-train under BoT-Market: bot_market_market_train.npz (751 training IDs)",
        "C_and_D": "MSMT17-train under BoT-MSMT17: bot_msmt_msmt_train.npz (1041 training IDs)"
    },
    "percentiles_reported": PCTILES,
    "cache": "e12_feature_cache + e18_train_cache",
    "Q1_test_vs_test": results["Q1_test_vs_test"],
    "Q2Q3_test_vs_train": results["Q2Q3_test_vs_train"],
}
out_path = RESULTS_DIR / "e20c_cap_intersection.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → {out_path}", flush=True)
