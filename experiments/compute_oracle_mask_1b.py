"""
Run 1b: oracle hub-masking (Ch2 rev 156).

Define hub set H by ood-arm item k-occurrence (top X% of gallery).
For each query q, forbid any hub item h with id(h) != id(q) from
appearing in q's ranked list. Hub's own identity's queries can still
match it. Gallery size unchanged; no true positives destroyed.

Measures the "pollution contribution" of hubs to other identities'
retrievals in isolation from the gallery-size effect.

Readouts per (dataset, arm, mask level):
  - mAP
  - id-ood gap as fraction of in-domain mAP
  - dropped-query count (should be 0 for 1b by construction, but report)

Pre-registered sign (rev 156): expect low-single-digit mAP gain,
same regime as e39 centering (+0.357 pp on MSMT ood).

Datasets: MSMT, Market. Arms: id, ood. Levels: 0, 0.5, 1, 2, 5, 10%.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
HUBNESS   = RESULTS / 'hubness_koccurrence.json'
MASK_LEVELS_PCT = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

DATASETS = {
    'MSMT': {
        'id':  ('bot_msmt_seed42_msmt.npz',  11659),
        'ood': ('msmt17_test_noREA.npz',     11659),
    },
    'Market': {
        'id':  ('market_test_noREA.npz',      3368),
        'ood': ('bot_msmt_seed42_market.npz', 3368),
    },
}


def load_qg(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32)
    pids = d['pids']; cids = d['cids']
    q = {'emb': emb[:n_q], 'pids': pids[:n_q], 'cids': cids[:n_q]}
    g = {'emb': emb[n_q:], 'pids': pids[n_q:], 'cids': cids[n_q:]}
    return q, g


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def compute_ap_from_sorted(sorted_g_pids, sorted_g_cids, q_pid, q_cid,
                             mask_set):
    """
    Given sorted (by sim) gallery pids/cids, query pid/cid, and a boolean mask set
    (items to skip entirely, masked_global_idx mapping), compute AP under the
    Re-ID convention: same-cam-same-pid gallery items excluded, then AP over
    positives (same-pid, different-cam or at least different index).
    mask_set is a boolean array over gallery indices, True = excluded.
    sorted_g_pids/sorted_g_cids are passed as index arrays into gallery.
    """
    pass  # Not used directly; see compute_map_for_arm below.


def compute_map_masked(sims, q_pids, q_cids, g_pids, g_cids, mask_item_bool,
                        g_hub_pid_of):
    """
    sims:  (Nq, Ng) similarity, unmasked
    mask_item_bool: (Ng,) bool, True if item is a 'hub' candidate
    g_hub_pid_of: gallery pid (== g_pids); used to decide hub masking per query

    For each query q, mask gallery j if mask_item_bool[j] AND g_pids[j] != q_pids[q].
    Also apply same-cam-same-pid exclusion (standard ReID).
    Compute AP per query over positives (g_pids == q_pid AND g_cids != q_cid).
    Return mAP and n_queries_used (queries with ≥1 valid positive after
    exclusion, no additional drops from masking since hub-pid's own queries keep hub).
    """
    Nq, Ng = sims.shape
    ap_sum = 0.0; n_valid = 0
    n_q_zero_pos = 0
    for q in range(Nq):
        qp, qc = q_pids[q], q_cids[q]
        # same-cam-same-pid exclusion
        exc_same_cam = (g_pids == qp) & (g_cids == qc)
        # oracle hub mask: hub AND different pid
        if mask_item_bool is not None:
            hub_mask = mask_item_bool & (g_pids != qp)
        else:
            hub_mask = np.zeros(Ng, dtype=bool)
        keep = ~(exc_same_cam | hub_mask)
        if keep.sum() == 0:
            n_q_zero_pos += 1
            continue
        sv = sims[q][keep]
        gp = g_pids[keep]
        # positives
        pos_mask = (gp == qp)
        n_pos = int(pos_mask.sum())
        if n_pos == 0:
            n_q_zero_pos += 1
            continue
        order = np.argsort(-sv, kind='stable')
        pos_at = pos_mask[order]
        # precision at each positive position
        cum_pos = np.cumsum(pos_at)
        ranks = np.arange(1, len(order) + 1)
        prec_at_pos = cum_pos / ranks
        ap = prec_at_pos[pos_at].sum() / n_pos
        ap_sum += ap
        n_valid += 1
    return (ap_sum / n_valid if n_valid > 0 else 0.0), n_valid, n_q_zero_pos


def main():
    print("Loading hubness JSON...", flush=True)
    hub = json.load(open(HUBNESS))

    out = {'mask_levels_pct': MASK_LEVELS_PCT, 'per_dataset': {}}

    for ds_name, arms in DATASETS.items():
        print(f"\n########## {ds_name} ##########", flush=True)
        # Hub ranking by ood-arm item k-occurrence
        ood_kocc = np.array(hub[ds_name]['per_item_occ']['ood'])  # (Ng,)
        Ng = len(ood_kocc)
        # Sort descending; most hubby first
        hub_order = np.argsort(-ood_kocc, kind='stable')
        print(f"  Ng={Ng}, ood kocc range {ood_kocc.min()}-{ood_kocc.max()}", flush=True)

        ds_out = {'n_gallery': int(Ng), 'per_arm': {}}

        for arm_name, (fname, n_q) in arms.items():
            print(f"\n  [{arm_name}] loading + computing sims...", flush=True)
            q, g = load_qg(fname, n_q)
            qE = l2(q['emb']); gE = l2(g['emb'])
            t0 = time.time()
            sims = qE @ gE.T  # (Nq, Ng), float32
            print(f"    sims: {sims.shape}  t={(time.time()-t0):.1f}s", flush=True)

            arm_out = {'levels': []}
            for level_pct in MASK_LEVELS_PCT:
                n_mask = int(Ng * level_pct / 100.0)
                mask_bool = np.zeros(Ng, dtype=bool)
                if n_mask > 0:
                    mask_bool[hub_order[:n_mask]] = True
                t0 = time.time()
                mAP, n_valid, n_zero = compute_map_masked(
                    sims, q['pids'], q['cids'], g['pids'], g['cids'],
                    mask_bool if n_mask > 0 else None, g['pids']
                )
                mAP_pct = mAP * 100.0
                print(f"    level {level_pct:>5.1f}% (n_mask={n_mask:>5}): "
                      f"mAP={mAP_pct:.3f}  n_valid={n_valid}  "
                      f"n_zero_pos={n_zero}  t={(time.time()-t0):.1f}s", flush=True)
                arm_out['levels'].append({
                    'level_pct': level_pct,
                    'n_mask': n_mask,
                    'mAP_pct': mAP_pct,
                    'n_queries_used': int(n_valid),
                    'n_queries_zero_pos': int(n_zero),
                })
            ds_out['per_arm'][arm_name] = arm_out

        # Compute id-ood gap as fraction of in-domain mAP per level
        id_lvls = {l['level_pct']: l['mAP_pct'] for l in ds_out['per_arm']['id']['levels']}
        ood_lvls = {l['level_pct']: l['mAP_pct'] for l in ds_out['per_arm']['ood']['levels']}
        ds_out['per_level_gap'] = []
        print(f"\n  [{ds_name}] gap (id - ood) / id:", flush=True)
        for lvl in MASK_LEVELS_PCT:
            id_m = id_lvls[lvl]; ood_m = ood_lvls[lvl]
            gap = id_m - ood_m
            gap_frac = gap / id_m if id_m > 0 else 0.0
            ds_out['per_level_gap'].append({
                'level_pct': lvl, 'id_mAP_pct': id_m, 'ood_mAP_pct': ood_m,
                'gap_pp': gap, 'gap_frac_of_id': gap_frac,
            })
            print(f"    level {lvl:>5.1f}%: id={id_m:.3f}  ood={ood_m:.3f}  "
                  f"gap={gap:.3f}pp  gap/id={gap_frac:.4f}", flush=True)

        out['per_dataset'][ds_name] = ds_out

    print("\n========== Summary (ood mAP recovery from oracle masking) ==========")
    for ds_name in DATASETS:
        ood_lvls = out['per_dataset'][ds_name]['per_arm']['ood']['levels']
        baseline = ood_lvls[0]['mAP_pct']
        print(f"\n[{ds_name}] ood baseline mAP = {baseline:.3f}")
        for lvl in ood_lvls:
            delta = lvl['mAP_pct'] - baseline
            print(f"  mask {lvl['level_pct']:>5.1f}%: {lvl['mAP_pct']:.3f} "
                  f"(delta {delta:+.3f} pp)")

    out_path = RESULTS / 'e43_oracle_mask_1b.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
