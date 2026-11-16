"""
Camera test (Daniel via claude-chat, Channel 4 rev 676 section 2a / rev 677
item 1): "At layer1, does the matched training image share the query's
camera more often than chance?" Proposed explanation for low early-layer
identity-agreement: GAP over layer1 averages 2,048 cells, ~45% background,
so the descriptor may be closer to "average colour/edge energy under this
camera's illumination" than a person descriptor.

Operationalization note (flagged in rev 678): literal camera-ID equality is
well-posed only for the IN-DOMAIN arms (A', D), where train and eval share
one dataset's camera numbering. For the OOD arms (B'-sub, C), train and
eval are different datasets with unrelated camera IDs, so ID equality is
vacuous by construction. The cross-dataset analog that tests the same
hypothesis without requiring IDs to align: does query camera predict WHICH
training camera gets matched, above chance (contingency / chi-square test,
Cramer's V as effect size, between query-camera and matched-train-camera
labels).

Runs all four layers (not just layer1), all four arms.

Writes /data/papers/low_dimensional_reid/results/e44_camera_test.json
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2_contingency
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import (
    RNG_SEED, N_QUERY, N_GALLERY, load_backbone_and_bn,
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
    extract_gap_vectors_and_terminal, snap_indices,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']


def run_arm_indomain(label, model_ckpt, dataset_name, q_idx=None, g_idx=None,
                      batch=32, train_pool_size=3000):
    """A' / D: literal camera-ID match (train and eval share camera space)."""
    print(f"\n=== camera test (in-domain): {label} ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    ds_obj = get_dataset(dataset_name, sample=False)
    rng = np.random.default_rng(RNG_SEED)
    train_all = ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    train_cams = np.array([int(s.camera_id) for s in train_samples])

    q_all = [s for s in ds_obj.query if not s.is_junk]
    g_all = [s for s in ds_obj.gallery if not s.is_junk]
    if q_idx is None:
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
    query_samples = [q_all[int(i)] for i in q_idx]
    gallery_samples = [g_all[int(i)] for i in g_idx]

    tr_gap, _ = extract_gap_vectors_and_terminal(backbone, bn, train_samples, batch=batch,
                                                   mode='train', tag='train')
    q_gap, _, q_pids, q_cams = extract_gap_vectors_and_terminal(backbone, bn, query_samples,
                                                                  batch=batch, mode='test', tag='query')

    # chance level: training pool's own marginal frequency of each camera id
    cam_counts = {int(c): int((train_cams == c).sum()) for c in np.unique(train_cams)}
    cam_freq = {c: n / pool_size for c, n in cam_counts.items()}

    per_layer = {}
    for Lname in LAYER_NAMES:
        q_nn = snap_indices(q_gap[Lname], tr_gap[Lname])
        matched_cam = train_cams[q_nn]
        match = (matched_cam == q_cams)
        rate = float(match.mean())
        chance = float(np.mean([cam_freq.get(int(c), 0.0) for c in q_cams]))
        print(f"    {Lname}: rate={100*rate:.2f}%  chance={100*chance:.2f}%  "
              f"ratio={rate/chance if chance > 0 else float('nan'):.2f}  n={len(q_cams)}", flush=True)
        per_layer[Lname] = {'rate': rate, 'chance': chance,
                             'ratio': rate / chance if chance > 0 else float('nan'),
                             'n_query': len(q_cams)}
    return {'label': label, 'mode': 'literal_match', 'n_train_cameras': len(cam_counts),
            'train_pool_size': pool_size, 'per_layer': per_layer}


def run_arm_crossdomain(label, model_ckpt, train_dataset_name, eval_dataset_name,
                         q_idx=None, g_idx=None, batch=32, train_pool_size=3000):
    """B'-sub / C: contingency test (query camera vs matched-train camera)."""
    print(f"\n=== camera test (cross-domain): {label} ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)
    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    train_cams = np.array([int(s.camera_id) for s in train_samples])

    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
    if q_idx is None:
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
    query_samples = [q_all[int(i)] for i in q_idx]

    tr_gap, _ = extract_gap_vectors_and_terminal(backbone, bn, train_samples, batch=batch,
                                                   mode='train', tag='train')
    q_gap, _, q_pids, q_cams = extract_gap_vectors_and_terminal(backbone, bn, query_samples,
                                                                  batch=batch, mode='test', tag='query')

    n_q_cams = len(np.unique(q_cams))
    n_tr_cams = len(np.unique(train_cams))
    per_layer = {}
    for Lname in LAYER_NAMES:
        q_nn = snap_indices(q_gap[Lname], tr_gap[Lname])
        matched_cam = train_cams[q_nn]
        # contingency table: rows = query camera, cols = matched train camera
        q_cam_vals = np.unique(q_cams)
        tr_cam_vals = np.unique(matched_cam)
        table = np.zeros((len(q_cam_vals), len(tr_cam_vals)), dtype=np.int64)
        for i, qc in enumerate(q_cam_vals):
            for j, tc in enumerate(tr_cam_vals):
                table[i, j] = int(((q_cams == qc) & (matched_cam == tc)).sum())
        # drop zero rows/cols for a valid chi2 test
        table = table[table.sum(axis=1) > 0]
        table = table[:, table.sum(axis=0) > 0]
        n = table.sum()
        if table.shape[0] > 1 and table.shape[1] > 1 and n > 0:
            chi2, p, dof, _ = chi2_contingency(table)
            k = min(table.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n * k))) if k > 0 else float('nan')
        else:
            chi2, p, cramers_v = float('nan'), float('nan'), float('nan')
        print(f"    {Lname}: chi2={chi2:.2f}  p={p:.4g}  Cramers_V={cramers_v:.4f}  "
              f"n={int(n)}  table_shape={table.shape}", flush=True)
        per_layer[Lname] = {'chi2': float(chi2), 'p_value': float(p),
                             'cramers_v': cramers_v, 'n': int(n),
                             'table_shape': list(table.shape)}
    return {'label': label, 'mode': 'contingency', 'n_query_cameras': n_q_cams,
            'n_train_cameras': n_tr_cams, 'train_pool_size': pool_size, 'per_layer': per_layer}


def main():
    out = {'rev': 678, 'arms': {}}

    out['arms']['A_prime'] = run_arm_indomain('A_prime', CKPT_MARKET, 'market1501')
    out['arms']['D'] = run_arm_indomain('D', CKPT_MSMT, 'msmt17')

    q_idx, g_idx = load_saved_bprime_sub_indices()
    out['arms']['B_prime_sub'] = run_arm_crossdomain('B_prime_sub', CKPT_MARKET,
                                                       'market1501', 'msmt17',
                                                       q_idx=q_idx, g_idx=g_idx)
    out['arms']['C'] = run_arm_crossdomain('C', CKPT_MSMT, 'msmt17', 'market1501')

    outp = RESULTS / 'e44_camera_test.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp}", flush=True)


if __name__ == '__main__':
    main()
