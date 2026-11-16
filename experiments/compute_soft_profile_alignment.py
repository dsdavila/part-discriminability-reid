"""
Soft-profile alignment (Daniel via claude-chat, Channel 4 rev 660 §3).

The ID-level snap retrieval (rev 659) showed the nearest-training-neighbour
ARGMAX carries real signal (8.6-24.8x chance) but is a poor top-1 decision
(15-47% accuracy). This tests whether the FULL similarity profile (to every
training identity, not just the argmax) carries more usable signal than the
hard top-1 assignment discards.

Two measurements, per layer, per arm, no new forward passes beyond what the
snap mechanism already computes (the full query-to-training similarity
vector, before taking argmax):

  (1) recall@k: P(the mate's own top-1 assigned training ID appears among
      the QUERY's top-k most-similar training identities), swept over k.
      If recall climbs steeply with k, argmax (k=1) was discarding usable
      signal, and the climb's steepness is how much.

  (2) profile correlation: build a per-TRAINING-IDENTITY similarity profile
      (max similarity over that identity's training images), for the query
      and for its true gallery mate, and for a random different-identity
      gallery item. Spearman rank correlation between query's profile and
      mate's profile vs query's profile and random's profile. Two images
      of one person should correlate even when their argmax identity
      differs -- that is Daniel's "different alignment method" hypothesis,
      tested directly.

Writes /data/papers/low_dimensional_reid/results/e44_soft_profile_<arm>.json
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import (
    RNG_SEED, N_QUERY, N_GALLERY, load_backbone_and_bn,
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
    extract_gap_vectors_and_terminal, l2,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
K_SWEEP = [1, 2, 3, 5, 10, 20, 50]
N_RAND_DRAWS = 10


def identity_profile(sim_matrix, train_pids, unique_ids, id_to_cols):
    """sim_matrix: (N, N_train). Returns (N, n_unique_ids) = max similarity
    over each identity's training images, per sample."""
    N = sim_matrix.shape[0]
    profiles = np.zeros((N, len(unique_ids)), dtype=np.float32)
    for j, uid in enumerate(unique_ids):
        cols = id_to_cols[uid]
        profiles[:, j] = sim_matrix[:, cols].max(axis=1)
    return profiles


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32, train_pool_size=3000):
    print(f"\n=== soft-profile alignment, arm {arm_name} (model={model_name}) ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    train_pids = np.array([int(s.person_id) for s in train_samples])
    unique_ids = np.unique(train_pids)
    id_to_cols = {uid: np.where(train_pids == uid)[0] for uid in unique_ids}
    print(f"  train pool: {len(train_samples)} ({len(unique_ids)} distinct train IDs)", flush=True)

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

    out = {'rev': 660, 'arm': arm_name, 'model': model_name,
           'n_query': len(query_samples), 'n_gallery': len(gallery_samples),
           'train_pool_size': pool_size, 'n_distinct_train_ids': int(len(unique_ids)),
           'k_sweep': K_SWEEP, 'per_layer': {}}

    rng_rand = np.random.default_rng(RNG_SEED + 700)

    for Lname in LAYER_NAMES:
        print(f"\n  --- {Lname} ---", flush=True)
        q_n = l2(q_gap[Lname])
        g_n = l2(g_gap[Lname])
        tr_n = l2(tr_gap[Lname])
        q_sim = q_n @ tr_n.T  # (N_q, N_train)
        g_sim = g_n @ tr_n.T  # (N_g, N_train)

        q_argmax_id = train_pids[q_sim.argmax(axis=1)]
        g_argmax_id = train_pids[g_sim.argmax(axis=1)]

        q_order = np.argsort(-q_sim, axis=1)  # (N_q, N_train) descending training-image rank
        q_order_ids = train_pids[q_order]  # (N_q, N_train) identity at each rank

        n_q = len(q_pids)
        valid = np.zeros(n_q, dtype=bool)
        recall_at_k = {k: np.zeros(n_q, dtype=bool) for k in K_SWEEP}
        # random-item recall@k baseline (Channel 4 rev 662 §1): same k-sweep, same
        # random different-identity draws used for the rand_rho comparison, checking
        # whether a RANDOM item's own argmax identity also falls in the query's
        # top-k -- tests whether the climb is identity signal or just coverage of a
        # concentrated, overloaded set of occupied training IDs (rev 659 §2: only
        # ~110-175 of 710-913 train IDs are ever assigned to anything).
        recall_at_k_rand = {k: np.full(n_q, np.nan) for k in K_SWEEP}

        q_profile = identity_profile(q_sim, train_pids, unique_ids, id_to_cols)
        g_profile = identity_profile(g_sim, train_pids, unique_ids, id_to_cols)

        mate_rho = np.full(n_q, np.nan)
        rand_rho = np.full(n_q, np.nan)
        rand_rho_all = [None] * n_q  # per-query array of individual random rhos, for AUC

        for i in range(n_q):
            same_pid = g_pids == q_pids[i]
            junk = same_pid & (g_cams == q_cams[i])
            mates = np.where(same_pid & ~junk)[0]
            if len(mates) == 0:
                continue
            valid[i] = True
            # "any mate" convention, matching compute_id_level_snap_retrieval.py's
            # P(mate matches) exactly at k=1 -- most queries here have >1 mate
            # (B'-sub: 57% do, mean 2.3), so "first mate only" understates recall.
            mate_ids = set(g_argmax_id[mates].tolist())
            topk_ids_by_k = {k: set(q_order_ids[i, :k].tolist()) for k in K_SWEEP}
            for k in K_SWEEP:
                recall_at_k[k][i] = len(mate_ids & topk_ids_by_k[k]) > 0

            # profile correlation uses the single mate with the highest own-argmax
            # similarity to the query (closest mate), for a well-defined pairing.
            mate_sims = [g_sim[m].max() for m in mates]
            closest_mate = mates[int(np.argmax(mate_sims))]
            rho_mate, _ = spearmanr(q_profile[i], g_profile[closest_mate])
            mate_rho[i] = rho_mate

            diff_pid = np.where(g_pids != q_pids[i])[0]
            if len(diff_pid) > 0:
                draws = rng_rand.choice(diff_pid, size=min(N_RAND_DRAWS, len(diff_pid)), replace=False)
                rhos = np.array([spearmanr(q_profile[i], g_profile[d])[0] for d in draws])
                rand_rho[i] = float(np.nanmean(rhos))
                rand_rho_all[i] = rhos
                # random-item recall@k: for each random draw, does ITS OWN argmax
                # identity fall in the query's top-k (same k-sweep, same coverage test)
                draw_ids = g_argmax_id[draws]
                for k in K_SWEEP:
                    recall_at_k_rand[k][i] = float(np.isin(draw_ids, list(topk_ids_by_k[k])).mean())

        n_valid = int(valid.sum())
        print(f"    recall@k, mate vs random baseline (the gap is what matters):", flush=True)
        recall_summary = {}
        for k in K_SWEEP:
            r_mate = float(recall_at_k[k][valid].mean())
            r_rand = float(np.nanmean(recall_at_k_rand[k][valid]))
            recall_summary[str(k)] = {'mate': r_mate, 'random': r_rand, 'gap': r_mate - r_rand}
            print(f"      k={k:<3d}: mate={100*r_mate:.2f}%  random={100*r_rand:.2f}%  "
                  f"gap={100*(r_mate-r_rand):+.2f}pp", flush=True)

        mate_rho_mean = float(np.nanmean(mate_rho[valid]))
        rand_rho_mean = float(np.nanmean(rand_rho[valid]))

        # AUC / d' for mate-vs-random discrimination (Channel 4 rev 662 §2): per
        # query, fraction of this query's random draws that mate_rho exceeds,
        # averaged over queries -- the standard paired Mann-Whitney AUC estimator.
        auc_per_query = np.full(n_q, np.nan)
        for i in np.where(valid)[0]:
            if rand_rho_all[i] is not None and not np.isnan(mate_rho[i]):
                auc_per_query[i] = float((mate_rho[i] > rand_rho_all[i]).mean()
                                          + 0.5 * (mate_rho[i] == rand_rho_all[i]).mean())
        auc = float(np.nanmean(auc_per_query[valid]))
        # d' using pooled mate vs random distributions (not per-query paired)
        all_mate = mate_rho[valid]
        all_rand = np.concatenate([r for r in rand_rho_all if r is not None])
        pooled_sd = np.sqrt((np.nanvar(all_mate) + np.nanvar(all_rand)) / 2)
        d_prime = float((np.nanmean(all_mate) - np.nanmean(all_rand)) / pooled_sd) if pooled_sd > 0 else float('nan')

        print(f"    profile correlation: mate_rho={mate_rho_mean:+.4f}  "
              f"rand_rho={rand_rho_mean:+.4f}  diff={mate_rho_mean-rand_rho_mean:+.4f}  "
              f"AUC={auc:.4f}  d'={d_prime:.4f}", flush=True)

        out['per_layer'][Lname] = {
            'n_valid': n_valid,
            'recall_at_k': recall_summary,
            'profile_correlation': {
                'mate_rho_mean': mate_rho_mean, 'rand_rho_mean': rand_rho_mean,
                'diff': mate_rho_mean - rand_rho_mean,
                'auc_mate_vs_random': auc, 'd_prime': d_prime,
            },
        }

    outp = RESULTS / f'e44_soft_profile_{arm_name}.json'
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
