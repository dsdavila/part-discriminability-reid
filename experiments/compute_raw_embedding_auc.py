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

def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32):
    print(f"\n=== raw-embedding pairwise AUC, arm {arm_name} ===", flush=True)
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

    _, q_term, q_pids, q_cams = extract_gap_vectors_and_terminal(backbone, bn, query_samples, batch=batch, mode='test', tag='query')
    _, g_term, g_pids, g_cams = extract_gap_vectors_and_terminal(backbone, bn, gallery_samples, batch=batch, mode='test', tag='gallery')

    sims = q_term @ g_term.T  # (N_q, N_g), terminal embeddings already L2-normalized
    n_q = len(q_pids)
    rng_rand = np.random.default_rng(RNG_SEED + 700)
    auc_per_query = []
    for i in range(n_q):
        same_pid = g_pids == q_pids[i]
        junk = same_pid & (g_cams == q_cams[i])
        mates = np.where(same_pid & ~junk)[0]
        if len(mates) == 0:
            continue
        mate_sim = sims[i, mates].max()  # closest mate, consistent with other scripts
        diff_pid = np.where(g_pids != q_pids[i])[0]
        if len(diff_pid) == 0:
            continue
        draws = rng_rand.choice(diff_pid, size=min(N_RAND_DRAWS, len(diff_pid)), replace=False)
        rand_sims = sims[i, draws]
        auc_per_query.append(float((mate_sim > rand_sims).mean() + 0.5*(mate_sim == rand_sims).mean()))
    auc = float(np.mean(auc_per_query))
    print(f"  raw embedding pairwise AUC (mate vs random, n={len(auc_per_query)}): {auc:.5f}", flush=True)
    return auc

q_idx, g_idx = load_saved_bprime_sub_indices()
auc_b = run_arm('B_prime_sub', CKPT_MARKET, 'BoT-Market', 'market1501', 'msmt17', q_idx=q_idx, g_idx=g_idx)
auc_c = run_arm('C', CKPT_MSMT, 'BoT-MSMT', 'msmt17', 'market1501', q_idx=None, g_idx=None)

print(f"\nSUMMARY: B'-sub raw-embedding AUC={auc_b:.5f}  C raw-embedding AUC={auc_c:.5f}")
