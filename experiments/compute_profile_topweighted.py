"""
Top-weighted profile comparison measures (Daniel via claude-chat, Channel 4
rev 664).

The soft-profile alignment's Spearman correlation (rev 660-663) weights
every one of ~710-913 training identities equally, diluting identity
signal that likely lives in the TOP of the similarity profile (the
training people the image actually resembles) rather than in the precise
ordering of the hundreds it resembles not at all -- the random baseline's
own high value (+0.46/+0.52) is a symptom of this dilution.

Three top-weighted measures, all computed on the SAME similarity matrices
already built for the soft-profile line (no new forward passes):

  (1) overlap@k: |top-k(query) INTERSECT top-k(other)| / k, swept over k.
  (2) RBO (rank-biased overlap, Webber et al. 2010): weights agreement at
      rank d by p^(d-1), so persistence p controls how fast the weight
      decays -- small p = top-heavy, large p = closer to uniform (Spearman-like).
  (3) softmax cosine: convert the similarity profile to a probability
      distribution via softmax at temperature T, then cosine the two
      distributions. Small T = sharper (more top-weighted).

Also runs BOTH max and mean identity-profile aggregation (Channel 4 rev
664 §3 -- the choice was unstated in the original soft-profile script,
which used max only).

Each measure reported as per-query paired Mann-Whitney AUC for
mate-vs-random discrimination (per 662 §2's correction), not a difference
of means.

Writes /data/papers/low_dimensional_reid/results/e44_profile_topweighted_<arm>.json
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
    extract_gap_vectors_and_terminal, l2,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer4']  # strongest layer per rev 663; extend to all 4 if useful
N_RAND_DRAWS = 10
K_SWEEP = [5, 10, 20, 50, 100]
RBO_P_SWEEP = [0.80, 0.90, 0.95, 0.98]
SOFTMAX_T_SWEEP = [0.01, 0.05, 0.1, 0.5]
AGGREGATIONS = ['max', 'mean']


def identity_profile(sim_matrix, train_pids, unique_ids, id_to_cols, agg='max'):
    N = sim_matrix.shape[0]
    profiles = np.zeros((N, len(unique_ids)), dtype=np.float32)
    for j, uid in enumerate(unique_ids):
        cols = id_to_cols[uid]
        if agg == 'max':
            profiles[:, j] = sim_matrix[:, cols].max(axis=1)
        else:
            profiles[:, j] = sim_matrix[:, cols].mean(axis=1)
    return profiles


def overlap_at_k(rank_a, rank_b, k):
    """rank_a, rank_b: (n_ids,) index arrays sorted by descending profile value."""
    return len(set(rank_a[:k].tolist()) & set(rank_b[:k].tolist())) / k


def rbo(rank_a, rank_b, p, max_depth=None):
    """Simple (non-extrapolated) RBO for two full rankings of the same item set."""
    n = len(rank_a) if max_depth is None else min(max_depth, len(rank_a))
    seen_a, seen_b = set(), set()
    overlap_sum = 0.0
    weight_sum = 0.0
    for d in range(1, n + 1):
        seen_a.add(int(rank_a[d - 1]))
        seen_b.add(int(rank_b[d - 1]))
        A_d = len(seen_a & seen_b) / d
        w = p ** (d - 1)
        overlap_sum += w * A_d
        weight_sum += w
    return (1 - p) * overlap_sum if weight_sum > 0 else float('nan')


def softmax_cosine(profile_a, profile_b, T):
    a = profile_a / T
    b = profile_b / T
    a = a - a.max()
    b = b - b.max()
    pa = np.exp(a); pa /= pa.sum()
    pb = np.exp(b); pb /= pb.sum()
    num = (pa * pb).sum()
    den = np.linalg.norm(pa) * np.linalg.norm(pb)
    return float(num / den) if den > 0 else float('nan')


def auc_from_lists(mate_vals, rand_val_lists):
    """mate_vals: (n_q,). rand_val_lists: list of arrays per query. Paired AUC."""
    per_query = []
    for mv, rv in zip(mate_vals, rand_val_lists):
        if rv is None or np.isnan(mv):
            continue
        per_query.append(float((mv > rv).mean() + 0.5 * (mv == rv).mean()))
    return float(np.mean(per_query)) if per_query else float('nan')


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32, train_pool_size=3000):
    print(f"\n=== top-weighted profile, arm {arm_name} (model={model_name}) ===", flush=True)
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

    out = {'rev': 664, 'arm': arm_name, 'model': model_name,
           'n_query': len(query_samples), 'n_gallery': len(gallery_samples),
           'train_pool_size': pool_size, 'n_distinct_train_ids': int(len(unique_ids)),
           'per_layer': {}}

    rng_rand = np.random.default_rng(RNG_SEED + 700)

    for Lname in LAYER_NAMES:
        print(f"\n  --- {Lname} ---", flush=True)
        q_n = l2(q_gap[Lname])
        g_n = l2(g_gap[Lname])
        tr_n = l2(tr_gap[Lname])
        q_sim = q_n @ tr_n.T
        g_sim = g_n @ tr_n.T

        layer_out = {}
        for agg in AGGREGATIONS:
            print(f"    [aggregation={agg}]", flush=True)
            q_profile = identity_profile(q_sim, train_pids, unique_ids, id_to_cols, agg=agg)
            g_profile = identity_profile(g_sim, train_pids, unique_ids, id_to_cols, agg=agg)
            q_rank = np.argsort(-q_profile, axis=1)  # (N_q, n_ids)
            g_rank = np.argsort(-g_profile, axis=1)

            n_q = len(q_pids)
            valid = np.zeros(n_q, dtype=bool)
            mate_overlap = {k: np.full(n_q, np.nan) for k in K_SWEEP}
            rand_overlap = {k: [None] * n_q for k in K_SWEEP}
            mate_rbo = {p: np.full(n_q, np.nan) for p in RBO_P_SWEEP}
            rand_rbo = {p: [None] * n_q for p in RBO_P_SWEEP}
            mate_sm = {T: np.full(n_q, np.nan) for T in SOFTMAX_T_SWEEP}
            rand_sm = {T: [None] * n_q for T in SOFTMAX_T_SWEEP}

            for i in range(n_q):
                same_pid = g_pids == q_pids[i]
                junk = same_pid & (g_cams == q_cams[i])
                mates = np.where(same_pid & ~junk)[0]
                if len(mates) == 0:
                    continue
                valid[i] = True
                mate_sims = [g_sim[m].max() for m in mates]
                mate = mates[int(np.argmax(mate_sims))]

                for k in K_SWEEP:
                    mate_overlap[k][i] = overlap_at_k(q_rank[i], g_rank[mate], k)
                for p in RBO_P_SWEEP:
                    mate_rbo[p][i] = rbo(q_rank[i], g_rank[mate], p, max_depth=200)
                for T in SOFTMAX_T_SWEEP:
                    mate_sm[T][i] = softmax_cosine(q_profile[i], g_profile[mate], T)

                diff_pid = np.where(g_pids != q_pids[i])[0]
                if len(diff_pid) > 0:
                    draws = rng_rand.choice(diff_pid, size=min(N_RAND_DRAWS, len(diff_pid)), replace=False)
                    for k in K_SWEEP:
                        rand_overlap[k][i] = np.array([overlap_at_k(q_rank[i], g_rank[d], k) for d in draws])
                    for p in RBO_P_SWEEP:
                        rand_rbo[p][i] = np.array([rbo(q_rank[i], g_rank[d], p, max_depth=200) for d in draws])
                    for T in SOFTMAX_T_SWEEP:
                        rand_sm[T][i] = np.array([softmax_cosine(q_profile[i], g_profile[d], T) for d in draws])

            agg_out = {'overlap_at_k': {}, 'rbo': {}, 'softmax_cosine': {}}
            print(f"      overlap@k AUC:", flush=True)
            for k in K_SWEEP:
                auc = auc_from_lists(mate_overlap[k][valid], [rand_overlap[k][i] for i in np.where(valid)[0]])
                agg_out['overlap_at_k'][str(k)] = auc
                print(f"        k={k:<4d}: AUC={auc:.4f}", flush=True)
            print(f"      RBO AUC:", flush=True)
            for p in RBO_P_SWEEP:
                auc = auc_from_lists(mate_rbo[p][valid], [rand_rbo[p][i] for i in np.where(valid)[0]])
                agg_out['rbo'][str(p)] = auc
                print(f"        p={p:<5g}: AUC={auc:.4f}", flush=True)
            print(f"      softmax-cosine AUC:", flush=True)
            for T in SOFTMAX_T_SWEEP:
                auc = auc_from_lists(mate_sm[T][valid], [rand_sm[T][i] for i in np.where(valid)[0]])
                agg_out['softmax_cosine'][str(T)] = auc
                print(f"        T={T:<5g}: AUC={auc:.4f}", flush=True)

            layer_out[agg] = agg_out
        out['per_layer'][Lname] = layer_out

    outp = RESULTS / f'e44_profile_topweighted_{arm_name}.json'
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
