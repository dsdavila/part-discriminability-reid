"""
Foreground-matched control for the cell-level divergent-vs-aligned agreement
diagnostic (compute_e_snap_a_cell_agreement.py).

Spec: Channel 4 rev 637 §1 (claude-chat). The original diagnostic's
"divergent cell" selection (top-k by residual-to-train-mean, ||x - mu_train||)
is the same quantity that was previously shown to be a person detector
(97.5% / 94.7% accuracy at layer4 against a 54% base rate). The original
result's two signatures -- contrast growing monotonically with depth, and
C's contrast exceeding B'-sub's at every layer -- are BOTH exactly what a
foreground-detection confound predicts (receptive fields sharpen fg/bg
separation with depth; Market crops are cleaner than MSMT's, so C's
foreground detection works better and earlier).

Control: restrict BOTH the divergent-cell set and the aligned-cell set to
FOREGROUND cells only (DeepLabV3 person segmentation, same model as the
precheck1c line, pooled to each layer's (H, W) grid at a 0.5 fill
threshold), then re-run the exact same mate-vs-random agreement statistic
within that restriction. If the divergent > aligned contrast survives, the
original result (Channel 4 rev 636 §1) stands; if it collapses to ~parity,
it is a rediscovered person detector.

Writes /data/papers/low_dimensional_reid/results/e44_e_snap_a_cellagree_fg_<arm>.json
"""
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fnn
import torchvision.models.segmentation as seg
from torch.utils.data import DataLoader
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import (
    RESULTS, device, RNG_SEED, N_QUERY, N_GALLERY,
    load_backbone_and_bn, DStest, DStrain, TRANSFORM,
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
)
from compute_e_snap_a_cell_agreement import (
    extract_layer_maps, per_cell_nearest_index_batched, per_cell_residual,
    TRAIN_POOL_SIZE_CELL_DIAGNOSTIC, FRACS,
)

LAYER_HW = {'layer1': (64, 32), 'layer2': (32, 16), 'layer3': (16, 8), 'layer4': (16, 8)}


@torch.no_grad()
def person_masks_at(samples, H, W, batch=16, seg_model=None):
    """Returns (N, H, W) bool array: cell is majority-foreground (>=0.5 fill)
    by DeepLabV3 person segmentation, pooled to the (H, W) grid. Reuses the
    criterion from compute_precheck1c.py's person_masks."""
    if seg_model is None:
        seg_model = seg.deeplabv3_resnet50(
            weights=seg.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1).to(device).eval()
    PERSON = 15
    ldr = DataLoader(DStrain(samples), batch_size=batch, num_workers=0, pin_memory=False)
    masks = []
    for imgs in ldr:
        out = seg_model(imgs.to(device))['out'].argmax(dim=1)
        is_person = (out == PERSON).float()
        cell = Fnn.adaptive_avg_pool2d(is_person.unsqueeze(1), (H, W)).squeeze(1)
        cell_fg = (cell >= 0.5).cpu().numpy().astype(bool)
        masks.append(cell_fg)
    return np.concatenate(masks, axis=0), seg_model


def compute_cell_agreement_fg(arm_name, model_ckpt, model_name, train_dataset_name,
                               eval_dataset_name, q_idx, g_idx,
                               layer_names=('layer1', 'layer2', 'layer3', 'layer4'),
                               train_pool_size=TRAIN_POOL_SIZE_CELL_DIAGNOSTIC,
                               batch=32, fracs=tuple(FRACS)):
    print(f"\n=== FG-matched cell-agreement arm {arm_name} (model={model_name}) ===", flush=True)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]

    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
    query_samples = [q_all[int(i)] for i in q_idx]
    gallery_samples = [g_all[int(i)] for i in g_idx]
    print(f"  query: {len(query_samples)}  gallery: {len(gallery_samples)}", flush=True)

    backbone, bn = load_backbone_and_bn(model_ckpt)
    seg_model = None

    out = {
        'rev': 638,
        'arm': arm_name,
        'control': 'foreground-matched (Channel 4 rev 637 §1)',
        'model_ckpt': str(model_ckpt),
        'params': {
            'train_pool_size_DECLARED_SUBSAMPLE': pool_size,
            'train_pool_full_set_size': len(train_all),
            'N_QUERY': len(query_samples), 'N_GALLERY': len(gallery_samples),
            'RNG_SEED': RNG_SEED, 'fracs': list(fracs),
        },
        'per_layer': {},
    }

    for Lname in layer_names:
        print(f"\n  --- layer {Lname} ---", flush=True)
        H, W = LAYER_HW[Lname]
        tr_maps = extract_layer_maps(backbone, train_samples, Lname, batch=batch,
                                      mode='train', tag='train')
        q_maps, q_pids, q_cams = extract_layer_maps(backbone, query_samples, Lname,
                                                      batch=batch, mode='test', tag='query')
        g_maps, g_pids, g_cams = extract_layer_maps(backbone, gallery_samples, Lname,
                                                      batch=batch, mode='test', tag='gallery')

        print(f"    computing person masks at ({H}x{W})...", flush=True)
        q_fg, seg_model = person_masks_at(query_samples, H, W, seg_model=seg_model)
        q_fg_flat = q_fg.reshape(len(query_samples), -1)
        fg_frac = float(q_fg_flat.mean())
        print(f"    query person-area fraction: {fg_frac:.3f}", flush=True)

        train_mean_cell = tr_maps.mean(axis=0)
        q_resid = per_cell_residual(q_maps, train_mean_cell)

        print(f"    computing per-cell nearest-training-index (query)...", flush=True)
        q_nn = per_cell_nearest_index_batched(q_maps, tr_maps)
        print(f"    computing per-cell nearest-training-index (gallery)...", flush=True)
        g_nn = per_cell_nearest_index_batched(g_maps, tr_maps)

        n_cells = q_nn.shape[0]
        del tr_maps, q_maps, g_maps
        gc.collect()

        n_q = len(q_pids)
        mate_agree = np.zeros((n_cells, n_q), dtype=np.float32)
        rand_agree = np.zeros((n_cells, n_q), dtype=np.float32)
        valid = np.zeros(n_q, dtype=bool)
        for i in range(n_q):
            same_pid = g_pids == q_pids[i]
            junk = same_pid & (g_cams == q_cams[i])
            mates = np.where(same_pid & ~junk)[0]
            if len(mates) == 0:
                continue
            valid[i] = True
            mate_agree[:, i] = (g_nn[:, mates] == q_nn[:, i:i + 1]).mean(axis=1)
            diff_pid = np.where(g_pids != q_pids[i])[0]
            rand_agree[:, i] = (g_nn[:, diff_pid] == q_nn[:, i:i + 1]).mean(axis=1)

        valid_idx = np.where(valid)[0]
        mate_agree_v = mate_agree[:, valid_idx]
        rand_agree_v = rand_agree[:, valid_idx]
        q_resid_v = q_resid[valid_idx]
        q_fg_v = q_fg_flat[valid_idx]  # (n_valid, n_cells) bool
        n_valid = len(valid_idx)

        layer_out = {
            'n_valid': n_valid, 'n_cells': int(n_cells),
            'foreground_fraction_mean': fg_frac,
            'unrestricted_all_cells': {
                'P_mate': float(mate_agree_v.mean()),
                'P_rand': float(rand_agree_v.mean()),
            },
            'k_sweep_foreground_matched': {},
        }

        # Residual rank restricted to foreground cells only: set background
        # cells' residual to -inf so they never enter the top-k (divergent)
        # or bottom-k (aligned) selection -- both sides draw from foreground only.
        q_resid_fg_masked = np.where(q_fg_v, q_resid_v, -np.inf)
        rank_order_fg = np.argsort(-q_resid_fg_masked, axis=1)

        for f in fracs:
            k = max(1, int(round(f * n_cells)))
            # Per-sample, clip k to that sample's available foreground cell count
            n_fg_per_sample = q_fg_v.sum(axis=1)
            mate_top_vals = []
            rand_top_vals = []
            mate_bot_vals = []
            rand_bot_vals = []
            n_skipped = 0
            for j in range(n_valid):
                avail = int(n_fg_per_sample[j])
                kj = min(k, avail)
                if kj == 0:
                    n_skipped += 1
                    continue
                top_idx = rank_order_fg[j, :kj]          # most divergent FG cells
                bot_idx = rank_order_fg[j, avail - kj:avail]  # least divergent FG cells (still FG)
                mate_top_vals.append(mate_agree_v[top_idx, j].mean())
                rand_top_vals.append(rand_agree_v[top_idx, j].mean())
                mate_bot_vals.append(mate_agree_v[bot_idx, j].mean())
                rand_bot_vals.append(rand_agree_v[bot_idx, j].mean())
            mate_top = np.array(mate_top_vals); rand_top = np.array(rand_top_vals)
            mate_bot = np.array(mate_bot_vals); rand_bot = np.array(rand_bot_vals)
            divergent_signal = float(mate_top.mean() - rand_top.mean())
            aligned_signal = float(mate_bot.mean() - rand_bot.mean())
            print(f"    f={f:.2f} (k={k}, n_skipped={n_skipped}/{n_valid}): "
                  f"FG-divergent mate-rand={divergent_signal:+.5f}  "
                  f"FG-aligned mate-rand={aligned_signal:+.5f}  "
                  f"{'DIVERGENT > ALIGNED (survives)' if divergent_signal > aligned_signal else 'divergent <= aligned (collapses)'}",
                  flush=True)
            layer_out['k_sweep_foreground_matched'][f'f{f:.2f}'] = {
                'k_requested': k, 'n_valid_scored': int(len(mate_top_vals)),
                'n_skipped_insufficient_fg': n_skipped,
                'P_mate_divergent_fg': float(mate_top.mean()),
                'P_rand_divergent_fg': float(rand_top.mean()),
                'P_mate_minus_rand_divergent_fg': divergent_signal,
                'P_mate_aligned_fg': float(mate_bot.mean()),
                'P_rand_aligned_fg': float(rand_bot.mean()),
                'P_mate_minus_rand_aligned_fg': aligned_signal,
            }

        out['per_layer'][Lname] = layer_out

    outp = RESULTS / f'e44_e_snap_a_cellagree_fg_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['B_prime_sub', 'C', 'both'], default='B_prime_sub')
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    if args.arm in ('B_prime_sub', 'both'):
        q_idx, g_idx = load_saved_bprime_sub_indices()
        compute_cell_agreement_fg('B_prime_sub', CKPT_MARKET, model_name='BoT-Market',
                                   train_dataset_name='market1501', eval_dataset_name='msmt17',
                                   q_idx=q_idx, g_idx=g_idx, batch=args.batch)

    if args.arm in ('C', 'both'):
        rng = np.random.default_rng(RNG_SEED)
        eval_ds_obj = get_dataset('market1501', sample=False)
        q_all = [s for s in eval_ds_obj.query if not s.is_junk]
        g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
        compute_cell_agreement_fg('C', CKPT_MSMT, model_name='BoT-MSMT',
                                   train_dataset_name='msmt17', eval_dataset_name='market1501',
                                   q_idx=q_idx, g_idx=g_idx, batch=args.batch)


if __name__ == '__main__':
    main()
