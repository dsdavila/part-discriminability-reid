"""
Amplification vs spreading: Daniel's natural experiment (Channel 4 rev 646c §3-4).

BoT ResNet-50 (last_stride=1) grid sizes: layer1 64x32, layer2 32x16,
layer3 16x8, layer4 16x8. Transitions layer1->layer2 and layer2->layer3 both
downsample 4x (stride-2 conv); layer3->layer4 does NOT downsample at all.

If total divergence grows across a transition, that growth could be genuine
nonlinear amplification OR an artefact of stride-2 pooling combining an
anomalous cell with 3 "normal" neighbours (spreads the anomaly over more of
the smaller map while diluting its per-cell magnitude). The L3->L4
transition has no such pooling, so growth there CANNOT be a spreading
artefact -- it isolates genuine amplification for free.

Per transition, per arm, tracks (per Daniel's prediction table, 646c §4):
  (a) total z-scored divergence summed over cells
  (b) count of cells above a z-threshold (spatial extent)
  (c) mean magnitude among cells above threshold (per-cell intensity)

Threshold: the 90th percentile of the TRAINING population's own z-distribution
at that layer, so ~10% of training cells exceed it by construction --
layer-comparable by design, not an arbitrary absolute cutoff.

z-normalisation: per-cell distance-to-nearest-LOO-training-sample
(Channel 4 rev 635-640, validated machinery), z'd against that layer's
LOO-training median/IQR, exactly as in compute_e_align.py.

Writes /data/papers/low_dimensional_reid/results/e44_amplification_vs_spreading_<arm>.json
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
)
from compute_e_snap_a_cell_agreement import extract_layer_maps, TRAIN_POOL_SIZE_CELL_DIAGNOSTIC
from compute_e_align import per_cell_distance_batched, robust_z

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
LAYER_HW = {'layer1': (64, 32), 'layer2': (32, 16), 'layer3': (16, 8), 'layer4': (16, 8)}
THRESH_PCTILE = 90


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32, train_pool_size=TRAIN_POOL_SIZE_CELL_DIAGNOSTIC):
    print(f"\n=== amplification-vs-spreading, arm {arm_name} (model={model_name}) ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]

    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    if q_idx is None:
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
    query_samples = [q_all[int(i)] for i in q_idx]
    print(f"  train pool: {len(train_samples)}  query: {len(query_samples)}", flush=True)

    out = {'rev': 650, 'arm': arm_name, 'model': model_name,
           'thresh_percentile': THRESH_PCTILE, 'n_query': len(query_samples),
           'per_layer': {}}

    for Lname in LAYER_NAMES:
        print(f"\n  --- {Lname} ---", flush=True)
        tr_maps = extract_layer_maps(backbone, train_samples, Lname, batch=batch,
                                      mode='train', tag='train')
        q_maps, q_pids, q_cams = extract_layer_maps(backbone, query_samples, Lname,
                                                      batch=batch, mode='test', tag='query')

        tr_dist, _ = per_cell_distance_batched(tr_maps, tr_maps, loo=True)  # (n_cells, N_train)
        median = float(np.median(tr_dist))
        q1, q3 = np.percentile(tr_dist, [25, 75])
        iqr = float(q3 - q1)
        thresh = float(np.percentile(tr_dist, THRESH_PCTILE))
        thresh_z = (thresh - median) / max(iqr, 1e-6)

        q_dist, _ = per_cell_distance_batched(q_maps, tr_maps)  # (n_cells, N_query)
        q_z = robust_z(q_dist, median, iqr)  # (n_cells, N_query)
        n_cells = q_z.shape[0]

        total_z_per_sample = q_z.sum(axis=0)  # sum over cells, per sample
        above = q_z > thresh_z
        count_above_per_sample = above.sum(axis=0).astype(float)
        # mean magnitude among cells above threshold, per sample (nan if none above)
        mag_above_per_sample = np.full(q_z.shape[1], np.nan)
        for i in range(q_z.shape[1]):
            if above[:, i].any():
                mag_above_per_sample[i] = q_z[above[:, i], i].mean()

        total_mean = float(total_z_per_sample.mean())
        mean_per_cell = total_mean / n_cells  # controls for raw cell-count differences across layers
        count_mean = float(count_above_per_sample.mean())
        mag_mean = float(np.nanmean(mag_above_per_sample))
        frac_above_mean = float(count_above_per_sample.mean() / n_cells)

        print(f"    n_cells={n_cells}  thresh_z(90th pctile of train)={thresh_z:.3f}", flush=True)
        print(f"    total z-divergence (mean over samples): {total_mean:.3f}  "
              f"(mean-per-cell: {mean_per_cell:.5f})", flush=True)
        print(f"    cells above threshold (mean count / fraction): {count_mean:.2f} / {frac_above_mean:.4f}", flush=True)
        print(f"    mean magnitude among above-threshold cells: {mag_mean:.3f}", flush=True)

        out['per_layer'][Lname] = {
            'n_cells': int(n_cells),
            'train_median': median, 'train_iqr': iqr, 'thresh_z': thresh_z,
            'total_z_divergence_mean': total_mean,
            'mean_z_divergence_per_cell': mean_per_cell,
            'count_above_thresh_mean': count_mean,
            'fraction_above_thresh_mean': frac_above_mean,
            'magnitude_above_thresh_mean': mag_mean,
        }
        del tr_maps, q_maps

    # Transition deltas, the actual discriminating numbers
    print(f"\n  --- transitions ---", flush=True)
    transitions = [('layer1', 'layer2'), ('layer2', 'layer3'), ('layer3', 'layer4')]
    out['transitions'] = {}
    for a, b in transitions:
        da = out['per_layer'][a]
        db = out['per_layer'][b]
        d_total = db['total_z_divergence_mean'] - da['total_z_divergence_mean']
        d_mean_percell = db['mean_z_divergence_per_cell'] - da['mean_z_divergence_per_cell']
        d_frac = db['fraction_above_thresh_mean'] - da['fraction_above_thresh_mean']
        d_mag = db['magnitude_above_thresh_mean'] - da['magnitude_above_thresh_mean']
        downsamples = (a, b) != ('layer3', 'layer4')
        print(f"    {a}->{b} (downsamples={downsamples}): "
              f"d(total)={d_total:+.3f}  d(mean/cell)={d_mean_percell:+.5f}  "
              f"d(extent frac)={d_frac:+.4f}  d(magnitude)={d_mag:+.3f}", flush=True)
        out['transitions'][f'{a}_to_{b}'] = {
            'downsamples': downsamples,
            'delta_total_z_divergence': d_total,
            'delta_mean_z_divergence_per_cell': d_mean_percell,
            'delta_fraction_above_thresh': d_frac,
            'delta_magnitude_above_thresh': d_mag,
        }

    outp = RESULTS / f'e44_amplification_vs_spreading_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    q_idx, _ = load_saved_bprime_sub_indices()
    run_arm('B_prime_sub', CKPT_MARKET, model_name='BoT-Market',
             train_dataset_name='market1501', eval_dataset_name='msmt17',
             q_idx=q_idx, batch=32)

    run_arm('C', CKPT_MSMT, model_name='BoT-MSMT',
            train_dataset_name='msmt17', eval_dataset_name='market1501',
            q_idx=None, batch=32)


if __name__ == '__main__':
    main()
