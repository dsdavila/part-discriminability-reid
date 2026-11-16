"""
Rank histogram for the raw-embedding retrieval (claude-chat, Channel 4 rev
670 §1). Free follow-up: AUC=0.8826 and mAP=9.12% for B'-sub can both be
true only if the rank distribution is bimodal (mAP ~= E[1/rank] != 1/E[rank],
so a few buried queries crater mAP while AUC, an average over *pairs* not
ranks, stays high). This reports the actual per-query rank distribution on
the SAME gallery subsample (N_GALLERY=5000) used throughout this line, so
mAP/AUC/rank-histogram are directly comparable to each other.

Per query: rank = 1 + (number of non-mate gallery items with strictly
higher similarity than the best-matching mate), i.e. the standard best-case
rank. AP computed in the usual CMC/mAP convention over this same gallery.

Writes /data/papers/low_dimensional_reid/results/e44_raw_embedding_rank_histogram_<arm>.json
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/reid-tk')
sys.path.insert(0, '/workspace/part-discriminability-reid/experiments')
from reid_tk.datasets import get_dataset
from compute_e_snap_a import (
    RNG_SEED, N_QUERY, N_GALLERY, load_backbone_and_bn,
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
    extract_gap_vectors_and_terminal,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
N_RAND_DRAWS = 10
RANK_BINS = [(1, 1), (2, 5), (6, 20), (21, 50), (51, 200), (201, 1000), (1001, None)]


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32):
    print(f"\n=== raw-embedding rank histogram, arm {arm_name} ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)
    rng = np.random.default_rng(RNG_SEED)
    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
    if q_idx is None:
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
    query_samples = [q_all[int(i)] for i in q_idx]
    gallery_samples = [g_all[int(i)] for i in g_idx]
    print(f"  query: {len(query_samples)}  gallery: {len(gallery_samples)}", flush=True)

    _, q_term, q_pids, q_cams = extract_gap_vectors_and_terminal(backbone, bn, query_samples, batch=batch, mode='test', tag='query')
    _, g_term, g_pids, g_cams = extract_gap_vectors_and_terminal(backbone, bn, gallery_samples, batch=batch, mode='test', tag='gallery')

    sims = q_term @ g_term.T  # (N_q, N_g), terminal embeddings already L2-normalized
    n_q = len(q_pids)
    n_g = len(g_pids)

    rng_rand = np.random.default_rng(RNG_SEED + 700)
    auc_per_query = []
    best_mate_ranks = []
    ap_per_query = []

    for i in range(n_q):
        same_pid = g_pids == q_pids[i]
        junk = same_pid & (g_cams == q_cams[i])
        mates = np.where(same_pid & ~junk)[0]
        if len(mates) == 0:
            continue
        valid_g = ~junk  # exclude junk from ranking universe, standard protocol
        row_sims = sims[i]

        # AUC (unchanged from compute_raw_embedding_auc.py, for cross-check)
        mate_sim = row_sims[mates].max()
        diff_pid = np.where(g_pids != q_pids[i])[0]
        if len(diff_pid) > 0:
            draws = rng_rand.choice(diff_pid, size=min(N_RAND_DRAWS, len(diff_pid)), replace=False)
            rand_sims = row_sims[draws]
            auc_per_query.append(float((mate_sim > rand_sims).mean() + 0.5 * (mate_sim == rand_sims).mean()))

        # Rank + AP over the full valid gallery (standard mAP convention)
        ranked_idx = np.argsort(-row_sims[valid_g])
        valid_idx_map = np.where(valid_g)[0]
        ranked_global = valid_idx_map[ranked_idx]
        is_mate = same_pid[ranked_global] & ~junk[ranked_global]

        hits = np.where(is_mate)[0]  # 0-indexed positions of mates in ranked list
        if len(hits) == 0:
            continue
        best_mate_ranks.append(int(hits[0]) + 1)  # 1-indexed best-case rank

        n_mates_total = len(hits)
        precisions = (np.arange(len(hits)) + 1) / (hits + 1)
        ap_per_query.append(float(precisions.mean()))

    auc = float(np.mean(auc_per_query))
    mAP = float(np.mean(ap_per_query))
    ranks = np.array(best_mate_ranks)

    hist = {}
    for lo, hi in RANK_BINS:
        if hi is None:
            key = f'{lo}+'
            cnt = int((ranks >= lo).sum())
        else:
            key = f'{lo}-{hi}'
            cnt = int(((ranks >= lo) & (ranks <= hi)).sum())
        hist[key] = cnt

    frac_rank1 = float((ranks == 1).mean())
    frac_rank_le5 = float((ranks <= 5).mean())
    frac_rank_gt1000 = float((ranks > 1000).mean())
    median_rank = float(np.median(ranks))

    print(f"  n_valid_queries={len(ranks)}  AUC={auc:.4f}  mAP={mAP:.4f}  "
          f"median_rank={median_rank:.0f}  frac_rank1={frac_rank1:.3f}  "
          f"frac_rank<=5={frac_rank_le5:.3f}  frac_rank>1000={frac_rank_gt1000:.3f}", flush=True)
    print(f"  rank histogram (best-mate rank, n_gallery={n_g}):", flush=True)
    for k, v in hist.items():
        print(f"    {k:>10s}: {v:5d}  ({100*v/len(ranks):.1f}%)", flush=True)

    out = {
        'rev': 671, 'arm': arm_name, 'model': model_name,
        'n_query': len(query_samples), 'n_gallery': len(gallery_samples),
        'n_valid_queries': int(len(ranks)),
        'auc': auc, 'mAP': mAP,
        'median_rank': median_rank,
        'frac_rank1': frac_rank1, 'frac_rank_le5': frac_rank_le5,
        'frac_rank_gt1000': frac_rank_gt1000,
        'rank_histogram': hist,
        'best_mate_ranks': ranks.tolist(),
    }
    outp = RESULTS / f'e44_raw_embedding_rank_histogram_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"  saved {outp}", flush=True)
    return out


def main():
    q_idx, g_idx = load_saved_bprime_sub_indices()
    out_b = run_arm('B_prime_sub', CKPT_MARKET, 'BoT-Market', 'market1501', 'msmt17', q_idx=q_idx, g_idx=g_idx)
    out_c = run_arm('C', CKPT_MSMT, 'BoT-MSMT', 'msmt17', 'market1501', q_idx=None, g_idx=None)
    print(f"\nSUMMARY: B'-sub AUC={out_b['auc']:.4f} mAP={out_b['mAP']:.4f} median_rank={out_b['median_rank']:.0f}  |  "
          f"C AUC={out_c['auc']:.4f} mAP={out_c['mAP']:.4f} median_rank={out_c['median_rank']:.0f}")


if __name__ == '__main__':
    main()
