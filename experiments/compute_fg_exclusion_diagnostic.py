"""
Diagnostic requested at Channel 4 rev 641 §2: for each layer/arm, (1) the
fraction of all cells excluded by the foreground restriction, and (2) what
fraction of the UNRESTRICTED top-k divergent set and bottom-k aligned set
was already foreground, BEFORE any restriction was applied. Decides whether
the foreground-matched control at layer4 did anything.

No gallery / nearest-matching needed -- only query-side residual ranking
and query-side foreground masks.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset
sys.path.insert(0, '/workspace/part-discriminability-reid/experiments')
from compute_e_snap_a import RNG_SEED, N_QUERY, N_GALLERY, load_backbone_and_bn, CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices
from compute_e_snap_a_cell_agreement import extract_layer_maps, per_cell_residual, TRAIN_POOL_SIZE_CELL_DIAGNOSTIC, FRACS
from compute_e_snap_a_cell_agreement_fg import person_masks_at, LAYER_HW

ARMS = [
    ('B_prime_sub', CKPT_MARKET, 'market1501', 'msmt17'),
    ('C', CKPT_MSMT, 'msmt17', 'market1501'),
]

for arm_name, ckpt, train_ds_name, eval_ds_name in ARMS:
    print(f"\n=== {arm_name} ===", flush=True)
    train_ds_obj = get_dataset(train_ds_name, sample=False)
    eval_ds_obj = get_dataset(eval_ds_name, sample=False)
    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(TRAIN_POOL_SIZE_CELL_DIAGNOSTIC, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]

    if arm_name == 'B_prime_sub':
        q_idx, _ = load_saved_bprime_sub_indices()
    else:
        rng2 = np.random.default_rng(RNG_SEED)
        q_all = [s for s in eval_ds_obj.query if not s.is_junk]
        q_idx = rng2.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    query_samples = [q_all[int(i)] for i in q_idx]

    backbone, bn = load_backbone_and_bn(ckpt)
    seg_model = None

    for Lname in ['layer1', 'layer2', 'layer3', 'layer4']:
        H, W = LAYER_HW[Lname]
        tr_maps = extract_layer_maps(backbone, train_samples, Lname, batch=32, mode='train', tag='train')
        q_maps, q_pids, q_cams = extract_layer_maps(backbone, query_samples, Lname, batch=32, mode='test', tag='query')
        train_mean_cell = tr_maps.mean(axis=0)
        q_resid = per_cell_residual(q_maps, train_mean_cell)  # (N, n_cells)
        q_fg, seg_model = person_masks_at(query_samples, H, W, seg_model=seg_model)
        q_fg_flat = q_fg.reshape(len(query_samples), -1)
        n_cells = q_resid.shape[1]
        excluded_frac = 1.0 - float(q_fg_flat.mean())

        rank_order = np.argsort(-q_resid, axis=1)  # unrestricted, descending residual
        results = {}
        for f in FRACS:
            k = max(1, int(round(f * n_cells)))
            top_idx = rank_order[:, :k]
            bot_idx = rank_order[:, -k:]
            top_fg_frac = np.array([q_fg_flat[j, top_idx[j]].mean() for j in range(len(query_samples))]).mean()
            bot_fg_frac = np.array([q_fg_flat[j, bot_idx[j]].mean() for j in range(len(query_samples))]).mean()
            results[f] = (top_fg_frac, bot_fg_frac)
        print(f"  {Lname} (n_cells={n_cells}): overall cells excluded by FG restriction = {excluded_frac:.3f}", flush=True)
        for f, (top_fg, bot_fg) in results.items():
            print(f"    f={f:.2f}: divergent-set FG fraction (pre-restriction) = {top_fg:.3f}   "
                  f"aligned-set FG fraction (pre-restriction) = {bot_fg:.3f}", flush=True)
        del tr_maps, q_maps
