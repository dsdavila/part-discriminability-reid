"""
E-align: does alignment-to-training predict retrieval AP along one continuum
(training -> in-domain held-out -> OOD), rather than in-domain held-out being
a null against which OOD looks anomalous?

Spec: Channel 4 rev 635 §2-3 (claude-chat, relaying Daniel). Supersedes the
cell-agreement gate (632 §4) as the primary line; that diagnostic's result
(Channel 4 rev 636 §1) is corroborating context here, not a separate gate.

Six (population, model) rows, two models:
  Market model (BoT-Market): train (LOO), A' (Market test, in-domain), B' (MSMT test, OOD)
  MSMT model   (BoT-MSMT):   train (LOO), D  (MSMT test, in-domain),   C  (Market test, OOD)

Per layer L in {1,2,3,4}, per sample: per-cell distance-to-nearest-training
(1 - max cosine similarity against the training bank at that layer; LOO for
the training population itself -- self-match excluded). Z-normalise each
layer's distances against that layer's LOO-training median/IQR BEFORE any
cross-layer aggregation (training and test populations share the same model,
so the same z-transform applies to all three).

Four alignment scalars per sample (the "two free choices"):
  cell aggregation: mean over all cells, OR top-k-divergent mean (k swept,
                    FRACS = [0.01, 0.05, 0.10, 0.25], same grid as I-1/P2)
  layer aggregation: mean over the 4 layers, OR max over the 4 layers
                      (argmax = the per-sample "onset layer")

Pre-registered (635 §3): top-k-over-cells x max-over-layers is predicted to
win, because that is the localised, onset-driven quantity the
displacement-field story is about.

Dependent variable: per-query AP from the standard baseline (unsnapped,
terminal-embedding) retrieval run, persisted here since this is the whole
experiment's y-axis (635 §3 note).

Also computed at near-zero marginal cost (635 §4 item 2):
  - anchor trajectory: per-layer argmax-training-index sequence per sample
    (free -- already computed as part of the per-cell matching)
  - per-sample onset layer: argmax layer of the per-layer mean-cell distance
  - spatial propagation: NOT implemented in this pass (needs the explicit
    receptive-field downsampling map between layers; flagged as follow-up)

Writes /data/papers/low_dimensional_reid/results/e44_e_align_<model>.json
(one file per model, covering all three of its populations).
"""
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import (
    RESULTS, device, RNG_SEED, N_QUERY, N_GALLERY,
    load_backbone_and_bn, DStest, DStrain,
    CKPT_MARKET, CKPT_MSMT, per_query_ap, l2,
)
from compute_e_snap_a_cell_agreement import extract_layer_maps, CHUNK_CELLS

TRAIN_POOL_SIZE = 3000  # same declared memory-bounded compromise as the cell-agreement script
FRACS = [0.01, 0.05, 0.10, 0.25]
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']


def per_cell_distance_batched(test_maps_np, train_maps_np, loo=False, chunk_cells=CHUNK_CELLS):
    """Returns dist: (n_cells, N_test) float32 = 1 - max cosine similarity
    against the training bank, and nn_idx: (n_cells, N_test) int32 = argmax
    training index (the 'anchor trajectory' ingredient, free here).

    loo=True means test_maps_np IS train_maps_np (same array, same order):
    self-similarity is masked to -inf before the max, giving the
    leave-one-out nearest-OTHER-training-sample distance.
    """
    N_test, C, H, W = test_maps_np.shape
    N_train = train_maps_np.shape[0]
    n_cells = H * W
    test_flat = test_maps_np.reshape(N_test, C, n_cells).transpose(2, 0, 1)
    train_flat = train_maps_np.reshape(N_train, C, n_cells).transpose(2, 0, 1)
    test_flat = test_flat / (np.linalg.norm(test_flat, axis=2, keepdims=True) + 1e-12)
    train_flat = train_flat / (np.linalg.norm(train_flat, axis=2, keepdims=True) + 1e-12)
    dist = np.zeros((n_cells, N_test), dtype=np.float32)
    nn_idx = np.zeros((n_cells, N_test), dtype=np.int32)
    for start in range(0, n_cells, chunk_cells):
        end = min(start + chunk_cells, n_cells)
        t_chunk = torch.from_numpy(np.ascontiguousarray(test_flat[start:end])).to(device)
        tr_chunk = torch.from_numpy(np.ascontiguousarray(train_flat[start:end])).to(device)
        with torch.no_grad():
            sims = torch.bmm(t_chunk, tr_chunk.transpose(1, 2))  # (chunk, N_test, N_train)
            if loo:
                diag = torch.arange(N_test, device=device)
                sims[:, diag, diag] = -2.0  # below any real cosine sim
            max_sim, idx = sims.max(dim=2)
        dist[start:end] = (1.0 - max_sim.cpu().numpy())
        nn_idx[start:end] = idx.cpu().numpy()
        del t_chunk, tr_chunk, sims, max_sim, idx
    return dist, nn_idx


def robust_z(x, ref_median, ref_iqr):
    """x normalised against a reference median/IQR; IQR floor avoids /0."""
    return (x - ref_median) / max(ref_iqr, 1e-6)


def cell_aggregate(dist_cells, frac):
    """dist_cells: (n_cells, N). Returns (N,) mean distance over the top-frac
    most divergent (highest-distance) cells per sample. frac=None means all
    cells (plain mean)."""
    n_cells, N = dist_cells.shape
    if frac is None:
        return dist_cells.mean(axis=0)
    k = max(1, int(round(frac * n_cells)))
    # top-k per sample (column): partial sort along axis 0
    top_k_vals = np.partition(dist_cells, -k, axis=0)[-k:]
    return top_k_vals.mean(axis=0)


def process_model(model_name, ckpt, train_dataset_name, in_domain_dataset_name,
                   ood_dataset_name, in_domain_label, ood_label,
                   q_idx_ood=None, g_idx_ood=None, batch=32,
                   train_pool_size=TRAIN_POOL_SIZE):
    print(f"\n=== E-align model {model_name} ===", flush=True)
    backbone, bn = load_backbone_and_bn(ckpt)

    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    in_domain_ds_obj = get_dataset(in_domain_dataset_name, sample=False)
    ood_ds_obj = get_dataset(ood_dataset_name, sample=False) if ood_dataset_name != in_domain_dataset_name else in_domain_ds_obj

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    print(f"  train pool for cell-bank: {len(train_samples)} / {len(train_all)} "
          f"(declared subsample, memory-bounded)", flush=True)

    # in-domain test population: full non-junk query set (no fixed-size subsample --
    # this is the "held-out" reference, want it representative; cap at N_QUERY for
    # parity with the OOD comparison and tractable compute)
    in_q_all = [s for s in in_domain_ds_obj.query if not s.is_junk]
    in_g_all = [s for s in in_domain_ds_obj.gallery if not s.is_junk]
    rng2 = np.random.default_rng(RNG_SEED + 777)
    in_q_idx = rng2.choice(len(in_q_all), size=min(N_QUERY, len(in_q_all)), replace=False)
    in_g_idx = rng2.choice(len(in_g_all), size=min(N_GALLERY, len(in_g_all)), replace=False)
    in_query_samples = [in_q_all[int(i)] for i in in_q_idx]
    in_gallery_samples = [in_g_all[int(i)] for i in in_g_idx]
    print(f"  in-domain ({in_domain_label}): query={len(in_query_samples)} "
          f"gallery={len(in_gallery_samples)}", flush=True)

    ood_q_all = [s for s in ood_ds_obj.query if not s.is_junk]
    ood_g_all = [s for s in ood_ds_obj.gallery if not s.is_junk]
    if q_idx_ood is None:
        rng3 = np.random.default_rng(RNG_SEED)
        q_idx_ood = rng3.choice(len(ood_q_all), size=min(N_QUERY, len(ood_q_all)), replace=False)
        g_idx_ood = rng3.choice(len(ood_g_all), size=min(N_GALLERY, len(ood_g_all)), replace=False)
    ood_query_samples = [ood_q_all[int(i)] for i in q_idx_ood]
    ood_gallery_samples = [ood_g_all[int(i)] for i in g_idx_ood]
    print(f"  OOD ({ood_label}): query={len(ood_query_samples)} "
          f"gallery={len(ood_gallery_samples)}", flush=True)

    out = {
        'rev': 636,
        'model': model_name,
        'train_dataset': train_dataset_name,
        'in_domain_label': in_domain_label, 'ood_label': ood_label,
        'params': {
            'train_pool_size_DECLARED_SUBSAMPLE': pool_size,
            'train_pool_full_set_size': len(train_all),
            'N_QUERY': N_QUERY, 'N_GALLERY': N_GALLERY,
            'RNG_SEED': RNG_SEED, 'fracs': FRACS,
        },
        'populations': {},
    }

    # Per-sample containers, filled per layer below, keyed by population name.
    pop_names = ['train', in_domain_label, ood_label]
    cell_dist_by_layer = {p: {L: None for L in LAYER_NAMES} for p in pop_names}
    nn_idx_by_layer = {p: {L: None for L in LAYER_NAMES} for p in pop_names}
    train_median_iqr = {}

    for Lname in LAYER_NAMES:
        print(f"\n  --- layer {Lname} ---", flush=True)
        tr_maps = extract_layer_maps(backbone, train_samples, Lname, batch=batch,
                                      mode='train', tag='train')
        in_q_maps, in_q_pids, in_q_cams = extract_layer_maps(
            backbone, in_query_samples, Lname, batch=batch, mode='test', tag='in_q')
        ood_q_maps, ood_q_pids, ood_q_cams = extract_layer_maps(
            backbone, ood_query_samples, Lname, batch=batch, mode='test', tag='ood_q')

        # LOO on training itself
        tr_dist, tr_nn = per_cell_distance_batched(tr_maps, tr_maps, loo=True)
        median = float(np.median(tr_dist))
        q1, q3 = np.percentile(tr_dist, [25, 75])
        iqr = float(q3 - q1)
        train_median_iqr[Lname] = (median, iqr)
        print(f"    train-LOO dist: median={median:.5f}  IQR={iqr:.5f}", flush=True)

        in_q_dist, in_q_nn = per_cell_distance_batched(in_q_maps, tr_maps)
        ood_q_dist, ood_q_nn = per_cell_distance_batched(ood_q_maps, tr_maps)

        # Query-side is the alignment-vs-AP unit of analysis (the dependent
        # variable is per-QUERY AP, so the feature is the query's own
        # alignment). Gallery-side alignment is not computed here -- not
        # needed for the M1/M2/M3 regression as specced.
        cell_dist_by_layer['train'][Lname] = tr_dist
        cell_dist_by_layer[in_domain_label][Lname] = in_q_dist
        cell_dist_by_layer[ood_label][Lname] = ood_q_dist
        nn_idx_by_layer['train'][Lname] = tr_nn
        nn_idx_by_layer[in_domain_label][Lname] = in_q_nn
        nn_idx_by_layer[ood_label][Lname] = ood_q_nn

        del tr_maps, in_q_maps, ood_q_maps
        gc.collect()

    # Baseline (unsnapped) retrieval AP -- the dependent variable.
    from compute_e_snap_a import extract_gap_vectors_and_terminal
    print(f"\n  --- baseline retrieval (terminal embeddings) ---", flush=True)
    _, in_q_term, in_q_pids2, in_q_cams2 = extract_gap_vectors_and_terminal(
        backbone, bn, in_query_samples, batch=batch, mode='test', tag='in_q_term')
    _, in_g_term, in_g_pids2, in_g_cams2 = extract_gap_vectors_and_terminal(
        backbone, bn, in_gallery_samples, batch=batch, mode='test', tag='in_g_term')
    _, ood_q_term, ood_q_pids2, ood_q_cams2 = extract_gap_vectors_and_terminal(
        backbone, bn, ood_query_samples, batch=batch, mode='test', tag='ood_q_term')
    _, ood_g_term, ood_g_pids2, ood_g_cams2 = extract_gap_vectors_and_terminal(
        backbone, bn, ood_gallery_samples, batch=batch, mode='test', tag='ood_g_term')

    in_aps, in_valid, in_r1 = per_query_ap(in_q_term, in_q_pids2, in_q_cams2,
                                           in_g_term, in_g_pids2, in_g_cams2)
    ood_aps, ood_valid, ood_r1 = per_query_ap(ood_q_term, ood_q_pids2, ood_q_cams2,
                                              ood_g_term, ood_g_pids2, ood_g_cams2)
    in_map = float(in_aps[in_valid].mean()) if in_valid.any() else float('nan')
    ood_map = float(ood_aps[ood_valid].mean()) if ood_valid.any() else float('nan')
    print(f"  {in_domain_label} baseline mAP={100*in_map:.3f}%  n_valid={int(in_valid.sum())}", flush=True)
    print(f"  {ood_label} baseline mAP={100*ood_map:.3f}%  n_valid={int(ood_valid.sum())}", flush=True)

    # Assemble per-sample alignment scalars for the two test populations.
    for pop_label, q_dist_layers, aps, valid, pids_, cams_ in [
        (in_domain_label, cell_dist_by_layer[in_domain_label], in_aps, in_valid, in_q_pids2, in_q_cams2),
        (ood_label, cell_dist_by_layer[ood_label], ood_aps, ood_valid, ood_q_pids2, ood_q_cams2),
    ]:
        n_cells_by_layer = {}
        z_mean_by_layer = {}  # (N,) per layer, mean-over-all-cells, z-normed
        z_topk_by_layer = {f: {} for f in FRACS}  # per frac, per layer, (N,)
        onset_layer_mean = None
        for Lname in LAYER_NAMES:
            dist_cells = q_dist_layers[Lname]  # (n_cells, N)
            n_cells_by_layer[Lname] = dist_cells.shape[0]
            median, iqr = train_median_iqr[Lname]
            z_all = robust_z(cell_aggregate(dist_cells, None), median, iqr)
            z_mean_by_layer[Lname] = z_all
            for f in FRACS:
                z_topk_by_layer[f][Lname] = robust_z(cell_aggregate(dist_cells, f), median, iqr)

        layer_arr_mean = np.stack([z_mean_by_layer[L] for L in LAYER_NAMES], axis=1)  # (N, 4)
        scalar_meanCell_meanLayer = layer_arr_mean.mean(axis=1)
        scalar_meanCell_maxLayer = layer_arr_mean.max(axis=1)
        onset_layer_idx = layer_arr_mean.argmax(axis=1)  # 0..3 -> layer1..layer4

        topk_scalars = {}
        for f in FRACS:
            layer_arr_topk = np.stack([z_topk_by_layer[f][L] for L in LAYER_NAMES], axis=1)
            topk_scalars[f] = {
                'meanLayer': layer_arr_topk.mean(axis=1),
                'maxLayer': layer_arr_topk.max(axis=1),
                'onset_layer_idx': layer_arr_topk.argmax(axis=1),
            }

        pop_out = {
            'n_cells_by_layer': n_cells_by_layer,
            'baseline_mAP': float(aps[valid].mean()) if valid.any() else float('nan'),
            'n_valid': int(valid.sum()),
            'per_query': {
                'AP': aps.tolist(),
                'valid': valid.tolist(),
                'pid': pids_.tolist(),
                'cam': cams_.tolist(),
            },
            'alignment': {
                'meanCell_meanLayer': scalar_meanCell_meanLayer.tolist(),
                'meanCell_maxLayer': scalar_meanCell_maxLayer.tolist(),
                'meanCell_onsetLayerIdx': onset_layer_idx.tolist(),
                'topK': {
                    f'f{f:.2f}': {
                        'meanLayer': topk_scalars[f]['meanLayer'].tolist(),
                        'maxLayer': topk_scalars[f]['maxLayer'].tolist(),
                        'onsetLayerIdx': topk_scalars[f]['onset_layer_idx'].tolist(),
                    } for f in FRACS
                },
            },
        }
        out['populations'][pop_label] = pop_out

        # Quick Spearman preview (pre-registered combo: topK f=0.01, maxLayer)
        from scipy.stats import spearmanr
        v = valid
        primary = np.array(topk_scalars[0.01]['maxLayer'])
        rho_primary, p_primary = spearmanr(primary[v], aps[v])
        rho_mean, p_mean = spearmanr(scalar_meanCell_meanLayer[v], aps[v])
        print(f"  [{pop_label}] Spearman(AP, alignment): "
              f"primary(topK0.01 x maxLayer)={rho_primary:+.4f} (p={p_primary:.2e})  "
              f"baseline(meanCell x meanLayer)={rho_mean:+.4f} (p={p_mean:.2e})", flush=True)

    out['populations']['train'] = {
        'n_cells_by_layer': {L: int(cell_dist_by_layer['train'][L].shape[0]) for L in LAYER_NAMES},
        'loo_median_iqr_by_layer': {L: {'median': train_median_iqr[L][0], 'iqr': train_median_iqr[L][1]}
                                     for L in LAYER_NAMES},
        'note': 'training population has no AP (not a query/gallery role); '
                'its LOO distances define the z-normalisation reference only.',
    }

    outp = RESULTS / f'e44_e_align_{model_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['market', 'msmt', 'both'], default='market')
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    if args.model in ('market', 'both'):
        process_model('BoT-Market', CKPT_MARKET, train_dataset_name='market1501',
                       in_domain_dataset_name='market1501', ood_dataset_name='msmt17',
                       in_domain_label='A_prime', ood_label='B_prime',
                       batch=args.batch)

    if args.model in ('msmt', 'both'):
        process_model('BoT-MSMT', CKPT_MSMT, train_dataset_name='msmt17',
                       in_domain_dataset_name='msmt17', ood_dataset_name='market1501',
                       in_domain_label='D', ood_label='C',
                       batch=args.batch)


if __name__ == '__main__':
    main()
