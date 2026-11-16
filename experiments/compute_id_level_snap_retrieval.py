"""
ID-level snap retrieval: the foundation test (Daniel, Channel 4 rev 656).

"Take that mechanism for mapping OOD samples to training samples, do it,
and use those IDs to score the retrieval. ... I want to know if this
snapping mechanism is even good enough as a starting point."

Per layer L, per arm: map each query and each gallery image to its nearest
training image (whole-image GAP, cosine -- variant (a)'s mechanism), take
that training image's IDENTITY LABEL (not just its raw index -- a
different training image of the same person still counts as a match).

Three measurements:
  (1) Consistency: P(query, true gallery mate assigned to SAME training ID)
      vs P(query, random different-identity gallery item assigned to same
      training ID). Measured query-to-gallery-mate directly (B'-sub has
      ~343 queries over ~330 identities -- not enough repeats per identity
      to measure within-query-set consistency).
  (2) Distinctness: for each assigned training ID, how many distinct OOD
      identities get assigned to it. Full distribution, not just the mean.
  (3) Retrieval under exact ID match: rank gallery by whether its assigned
      training ID equals the query's. Explicit tie policy: all gallery
      items sharing the query's assigned ID are tied at the top (arbitrary
      stable order among ties), everything else follows (also arbitrary
      order) -- mirrors what an actual ID-snap deployment would do, since
      it cannot distinguish within a tied group. mAP computed with
      interpolated precision over the tied block (standard treatment).

(1)+(2) are jointly necessary and sufficient for the snap to preserve
identity structure; (3) is their consequence and should be roughly
predictable from them -- used here as an internal consistency check.

NOTE ON PROVENANCE: the raw per-sample nearest-training-index arrays from
compute_e_snap_a.py and compute_e_snap_a_cell_agreement.py were NOT
persisted to disk (checked: only aggregate P3 stats are saved). This
script re-derives the whole-image nearest-match from scratch rather than
loading indices that do not exist on disk -- flagging per Channel 4 rev
657's citation-discipline rule (name the file+key, and if it is not
there, that is the signal, not a license to assume it is).

Writes /data/papers/low_dimensional_reid/results/e44_id_level_snap_<arm>.json
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import (
    RNG_SEED, N_QUERY, N_GALLERY, load_backbone_and_bn,
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
    extract_gap_vectors_and_terminal, snap_indices, l2,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
N_RAND_DRAWS = 20  # random-gallery-item draws per query for the chance comparison


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32, train_pool_size=3000):
    print(f"\n=== ID-level snap retrieval, arm {arm_name} (model={model_name}) ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    train_pids = np.array([int(s.person_id) for s in train_samples])
    print(f"  train pool: {len(train_samples)} ({len(set(train_pids.tolist()))} distinct train IDs)", flush=True)

    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
    if q_idx is None:
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
    query_samples = [q_all[int(i)] for i in q_idx]
    gallery_samples = [g_all[int(i)] for i in g_idx]
    print(f"  query: {len(query_samples)}  gallery: {len(gallery_samples)}", flush=True)

    tr_gap, _ = extract_gap_vectors_and_terminal(backbone, bn, train_samples, batch=batch,
                                                   mode='train', tag='train')
    q_gap, _, q_pids, q_cams = extract_gap_vectors_and_terminal(backbone, bn, query_samples,
                                                                  batch=batch, mode='test', tag='query')
    g_gap, _, g_pids, g_cams = extract_gap_vectors_and_terminal(backbone, bn, gallery_samples,
                                                                  batch=batch, mode='test', tag='gallery')

    out = {'rev': 656, 'arm': arm_name, 'model': model_name,
           'n_query': len(query_samples), 'n_gallery': len(gallery_samples),
           'train_pool_size': pool_size, 'n_distinct_train_ids': int(len(set(train_pids.tolist()))),
           'per_layer': {}}

    rng_rand = np.random.default_rng(RNG_SEED + 500)

    for Lname in LAYER_NAMES:
        print(f"\n  --- {Lname} ---", flush=True)
        q_nn = snap_indices(q_gap[Lname], tr_gap[Lname])  # (N_q,) train-pool index
        g_nn = snap_indices(g_gap[Lname], tr_gap[Lname])  # (N_g,)
        q_assigned_id = train_pids[q_nn]
        g_assigned_id = train_pids[g_nn]

        n_q = len(q_pids)
        valid = np.zeros(n_q, dtype=bool)
        mate_match = np.zeros(n_q, dtype=bool)
        rand_match_rate = np.zeros(n_q, dtype=float)

        for i in range(n_q):
            same_pid = g_pids == q_pids[i]
            junk = same_pid & (g_cams == q_cams[i])
            mates = np.where(same_pid & ~junk)[0]
            if len(mates) == 0:
                continue
            valid[i] = True
            mate_match[i] = (g_assigned_id[mates] == q_assigned_id[i]).any()  # any mate matches
            diff_pid = np.where(g_pids != q_pids[i])[0]
            if len(diff_pid) > 0:
                draws = rng_rand.choice(diff_pid, size=min(N_RAND_DRAWS, len(diff_pid)), replace=False)
                rand_match_rate[i] = (g_assigned_id[draws] == q_assigned_id[i]).mean()

        n_valid = int(valid.sum())
        p_mate = float(mate_match[valid].mean())
        p_rand = float(rand_match_rate[valid].mean())
        print(f"    (1) consistency: P(mate matches)={p_mate:.4f}  P(random matches)={p_rand:.4f}  "
              f"diff={p_mate-p_rand:+.4f}  n_valid={n_valid}", flush=True)

        # (2) distinctness: for each assigned train ID that gets used at all (by query OR
        # gallery), how many distinct OOD identities (from the QUERY set, which has one
        # entry per OOD identity roughly) land on it.
        assigned_ids_used, counts = np.unique(q_assigned_id, return_counts=True)
        # map assigned_id -> set of distinct query pids assigned to it
        collisions = {}
        for aid, qpid in zip(q_assigned_id, q_pids):
            collisions.setdefault(int(aid), set()).add(int(qpid))
        distinct_counts = np.array([len(v) for v in collisions.values()])
        n_assigned_ids_used = len(collisions)
        collision_rate = float((distinct_counts > 1).mean())
        print(f"    (2) distinctness: {n_assigned_ids_used} distinct train IDs used by "
              f"{n_valid if n_valid else n_q} OOD queries  "
              f"(mean distinct-OOD-per-train-ID={distinct_counts.mean():.2f}, "
              f"median={np.median(distinct_counts):.1f}, max={distinct_counts.max()}, "
              f"collision_rate={collision_rate:.3f})", flush=True)

        # (3) retrieval under exact ID match, tie policy: matching-ID gallery items tied
        # at top, non-matches follow. AP under ties computed by Monte Carlo averaging
        # over random tie-breaks within the matched block (verified against direct
        # simulation; a closed-form shortcut was tried and found WRONG -- see commit
        # history / Channel 4 rev 658 -- so this uses the safe empirical average).
        N_TIE_MC = 50
        aps = np.zeros(n_q, dtype=np.float64)
        r1 = np.zeros(n_q, dtype=bool)
        tie_rng = np.random.default_rng(RNG_SEED + 900)
        for i in range(n_q):
            if not valid[i]:
                continue
            same_pid_all = g_pids == q_pids[i]
            junk = same_pid_all & (g_cams == q_cams[i])
            keep = ~junk
            g_ids_kept = g_assigned_id[keep]
            g_pids_kept = g_pids[keep]
            is_id_match = g_ids_kept == q_assigned_id[i]
            good = g_pids_kept == q_pids[i]
            n_good_total = int(good.sum())
            if n_good_total == 0:
                continue
            n_match = int(is_id_match.sum())
            # R1 under this tie policy: the mate's gallery item is IN the tied top block
            # AND is actually one of the relevant (good) items within it -- i.e. the
            # assigned-ID match actually includes a true match, not just that SOME
            # gallery item (of any identity) happens to share the assigned ID. Equal to
            # whether this query is one of the (1)-consistency "mate matches" cases.
            good_in_match_check = good[is_id_match]
            r1[i] = bool(good_in_match_check.any())
            if n_match == 0:
                continue  # no gallery item shares the assigned ID; AP = 0 under this ranking
            good_in_match = good[is_id_match]
            good_in_rest = good[~is_id_match]
            # non-matched part of the ranking is fixed (arbitrary but consistent order);
            # precompute its contribution once.
            n_good_in_match = int(good_in_match.sum())
            ap_mc = np.empty(N_TIE_MC)
            for m in range(N_TIE_MC):
                perm = tie_rng.permutation(n_match)
                good_shuffled = good_in_match[perm]
                cum = np.cumsum(good_shuffled)
                ranks = np.arange(1, n_match + 1)
                prec_in_match = cum / ranks
                ap_from_match = float((prec_in_match * good_shuffled).sum())
                ap_from_rest = 0.0
                if n_good_in_match < n_good_total:
                    cum_after = np.cumsum(good_in_rest)
                    ranks_after = n_match + np.arange(1, len(good_in_rest) + 1)
                    prec_after = (n_good_in_match + cum_after) / ranks_after
                    ap_from_rest = float((prec_after * good_in_rest).sum())
                ap_mc[m] = (ap_from_match + ap_from_rest) / n_good_total
            aps[i] = ap_mc.mean()

        id_map = float(aps[valid].mean()) if n_valid else float('nan')
        id_r1 = float(r1[valid].mean()) if n_valid else float('nan')
        print(f"    (3) retrieval under exact ID match: mAP={100*id_map:.3f}%  R1(tie-aware)={100*id_r1:.3f}%  "
              f"n_valid={n_valid}", flush=True)

        out['per_layer'][Lname] = {
            'consistency': {'P_mate_match': p_mate, 'P_random_match': p_rand,
                             'diff': p_mate - p_rand, 'n_valid': n_valid},
            'distinctness': {
                'n_assigned_train_ids_used': n_assigned_ids_used,
                'mean_distinct_ood_per_train_id': float(distinct_counts.mean()),
                'median_distinct_ood_per_train_id': float(np.median(distinct_counts)),
                'max_distinct_ood_per_train_id': int(distinct_counts.max()),
                'collision_rate': collision_rate,
                'distinct_count_histogram': {str(k): int((distinct_counts == k).sum())
                                              for k in sorted(set(distinct_counts.tolist()))},
            },
            'id_match_retrieval': {'mAP': id_map, 'rank1_tie_aware': id_r1, 'n_valid': n_valid,
                                    'tie_policy': 'matched-ID gallery items tied at top, '
                                                  'interpolated precision within the tie block'},
        }

    outp = RESULTS / f'e44_id_level_snap_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    q_idx, g_idx = load_saved_bprime_sub_indices()
    run_arm('B_prime_sub', CKPT_MARKET, model_name='BoT-Market',
             train_dataset_name='market1501', eval_dataset_name='msmt17',
             q_idx=q_idx, g_idx=g_idx, batch=32)

    run_arm('C', CKPT_MSMT, model_name='BoT-MSMT',
            train_dataset_name='msmt17', eval_dataset_name='market1501',
            q_idx=None, g_idx=None, batch=32)


if __name__ == '__main__':
    main()
