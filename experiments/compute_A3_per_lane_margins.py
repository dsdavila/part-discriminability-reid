"""
A3 per-lane margins (rev 303 pre-registration).

For each cell, compute the nearest-vs-second-nearest lane distance for
every item. Lane c = that arm's gallery camera-c centroid (unit L2).
Two metrics:
  - Euclidean: ||x - lane_c||
  - Angular (deg): arccos(x . lane_c / (||x|| ||lane_c||))

Report:
  - median(nearest / second-nearest) ratio per cell
  - ratio floors: >= 0.9 = near-arbitrary; <= 0.7 = well-defined
  - difference between arms under 0.02 is not a finding (effect floor, hh)
  - B_cam own concentration per arm (already on record from B1, re-logged)

Claude-chat's claim under test: per-lane margins mechanically squeezed
because ||v_perp||^2 is common to all lanes and carries ~96% of energy,
so the lane-discriminating part is a small residual on a large shared term.
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)"},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)"},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)"},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)"},
}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def top_r_basis_with_s(mat, r):
    U, s, Vt = np.linalg.svd(mat, full_matrices=False)
    return Vt[:r].T, s


def process_cell(tag, info):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    # Use GALLERY to define lanes (they are the reference population)
    g_emb = l2(emb[n_q:]); g_cids = cids[n_q:]
    # Items = all L2-normalized items (gallery + query) for margins
    X_all = l2(emb)
    cids_all = cids
    uc = np.unique(g_cids)
    n_cams = len(uc)
    # Lanes = gallery camera centroids, L2-normalized
    lanes_unnorm = np.stack([g_emb[g_cids == c].mean(0) for c in uc])
    lanes = l2(lanes_unnorm)

    # Euclidean distance each item to each lane
    #   || x - l ||^2 = 2 - 2 x.l   (both unit-norm)
    sims = X_all @ lanes.T  # (n, n_cams)
    # Euclidean = sqrt(2 - 2 sims)
    eucl = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))
    # Angular (degrees)
    angles = np.degrees(np.arccos(np.clip(sims, -1.0, 1.0)))

    # For each item, sort distances; nearest and second-nearest
    sorted_e = np.sort(eucl, axis=1)
    sorted_a = np.sort(angles, axis=1)
    near_e = sorted_e[:, 0]; sec_e = sorted_e[:, 1]
    near_a = sorted_a[:, 0]; sec_a = sorted_a[:, 1]
    ratio_e = near_e / np.maximum(sec_e, 1e-12)
    ratio_a = near_a / np.maximum(sec_a, 1e-12)

    med_e = float(np.median(ratio_e))
    med_a = float(np.median(ratio_a))
    # CIs via simple bootstrap over items (not identity-clustered; margins
    # are per-item properties, resampling items captures the sample-size
    # uncertainty needed here. Item resampling is defensible because the
    # per-item ratio is independent of pid.)
    rng = np.random.default_rng(42)
    N_B = 500
    e_boot = []; a_boot = []
    n = len(ratio_e)
    for _ in range(N_B):
        idx = rng.integers(0, n, n)
        e_boot.append(np.median(ratio_e[idx]))
        a_boot.append(np.median(ratio_a[idx]))
    e_ci = [float(np.quantile(e_boot, 0.025)), float(np.quantile(e_boot, 0.975))]
    a_ci = [float(np.quantile(a_boot, 0.025)), float(np.quantile(a_boot, 0.975))]

    # Lane-assignment agreement with true cam
    nearest_cam_e = np.argmin(eucl, axis=1)
    nearest_cam_a = np.argmin(angles, axis=1)
    # Map cid -> lane row
    cid_to_row = {c: i for i, c in enumerate(uc)}
    true_row = np.array([cid_to_row[c] for c in cids_all])
    acc_e = float((nearest_cam_e == true_row).mean())
    acc_a = float((nearest_cam_a == true_row).mean())

    # Also: mean distance to the TRUE lane (separation of the arm's cameras)
    true_e_dist = eucl[np.arange(len(X_all)), true_row]
    true_a_dist = angles[np.arange(len(X_all)), true_row]

    print(f"  n_cams={n_cams}  n_items={len(X_all)}", flush=True)
    print(f"  median nearest/second-nearest ratio (Euclidean) = {med_e:.4f}  CI [{e_ci[0]:.4f}, {e_ci[1]:.4f}]", flush=True)
    print(f"  median nearest/second-nearest ratio (angular)   = {med_a:.4f}  CI [{a_ci[0]:.4f}, {a_ci[1]:.4f}]", flush=True)
    print(f"  nearest-lane accuracy vs true cam: Euclidean={acc_e:.4f}  angular={acc_a:.4f}", flush=True)
    print(f"  mean dist to true lane: Euclidean={true_e_dist.mean():.4f}  angular(deg)={true_a_dist.mean():.3f}", flush=True)

    # Verdict per floor
    def verdict(r):
        if r >= 0.9: return "near-arbitrary (>=0.9)"
        if r <= 0.7: return "well-defined (<=0.7)"
        return "indeterminate (0.7-0.9)"
    print(f"  Euclidean verdict: {verdict(med_e)}", flush=True)
    print(f"  Angular verdict:   {verdict(med_a)}", flush=True)

    return {
        'cell': tag,
        'label': info['label'],
        'n_cams': n_cams,
        'median_ratio_euclidean': med_e,
        'median_ratio_angular': med_a,
        'ci_euclidean': e_ci,
        'ci_angular': a_ci,
        'nearest_lane_accuracy_euclidean': acc_e,
        'nearest_lane_accuracy_angular': acc_a,
        'mean_true_lane_euclidean': float(true_e_dist.mean()),
        'mean_true_lane_angular_deg': float(true_a_dist.mean()),
        'euclidean_verdict': verdict(med_e),
        'angular_verdict': verdict(med_a),
    }


def main():
    out = {}
    for tag, info in CELLS.items():
        out[tag] = process_cell(tag, info)
    out_path = RESULTS / 'e43_A3_per_lane_margins.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n\n=== A3 summary (median nearest/second-nearest lane ratio) ===")
    print(f"{'cell':9s}  {'Euclidean':>12s}  {'Angular':>12s}  {'E verdict':>22s}  {'A verdict':>22s}")
    for tag in CELLS:
        r = out[tag]
        print(f"{tag:9s}  {r['median_ratio_euclidean']:9.4f}     {r['median_ratio_angular']:9.4f}     {r['euclidean_verdict']:>22s}  {r['angular_verdict']:>22s}")

    # Arm-pair differences
    print("\n\n=== Arm-pair differences (ood - id) ===")
    pairs = [('Market model', 'A_prime', 'B_prime'), ('MSMT model', 'D', 'C')]
    for label, id_tag, ood_tag in pairs:
        de = out[ood_tag]['median_ratio_euclidean'] - out[id_tag]['median_ratio_euclidean']
        da = out[ood_tag]['median_ratio_angular'] - out[id_tag]['median_ratio_angular']
        flag_e = " (<0.02 effect floor, not a finding)" if abs(de) < 0.02 else ""
        flag_a = " (<0.02 effect floor, not a finding)" if abs(da) < 0.02 else ""
        print(f"  {label}: Euclidean {de:+.4f}{flag_e}   angular {da:+.4f}{flag_a}")


if __name__ == '__main__':
    main()
