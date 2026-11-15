"""
Cell-level divergent-vs-aligned agreement diagnostic.

Spec: Channel 4 rev 632 §4 (claude-chat, correcting the invalid gate in 631 §5).

Gates variant (b). Per layer L, per arm, compares:

    P(query cell and its true gallery mate's cell snap to the same training
      cell) vs P(query cell and a different-identity gallery item's cell do),

over (a) ALL cells, and (b) the top-k most DIVERGENT cells per query (ranked
by per-cell residual-to-train-mean -- the same statistic P2 uses), contrasted
against (c) the bottom-k most ALIGNED cells at the same k. "Snap to the same
training cell" means: nearest training-image patch at that same (h, w) grid
position, by cosine similarity (the per-cell matching basis already built
for the I-1 line).

The decisive signature (632 §4): mate-agreement exceeding random-agreement
MORE on divergent cells than on aligned cells. If divergent-cell agreement is
at or below aligned-cell agreement, the displacement field is not
identity-bearing and variant (b) will fail for a reason unrelated to GAP.

Declared pool-size compromise (Channel 4 rev 634): the full per-cell training
bank at layer1 for the full training set would be ~63GB (MSMT) / ~27GB
(Market) -- infeasible in RAM. Uses N_TRAIN_POOL=3000 (the SAME subsample
size that was an undeclared bug in revs 629-630), now a declared,
memory-bounded choice for THIS diagnostic only. The whole-image GAP matching
in compute_e_snap_a.py uses the full training pool; this script does not,
and says so in its own output.

Writes /data/papers/low_dimensional_reid/results/e44_e_snap_a_cellagree_<arm>.json
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
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
)

TRAIN_POOL_SIZE_CELL_DIAGNOSTIC = 3000  # declared compromise, see module docstring
FRACS = [0.01, 0.05, 0.10, 0.25]
CHUNK_CELLS = 16


@torch.no_grad()
def extract_layer_maps(backbone, samples, layer_name, batch=32, mode='test', tag=''):
    """Full backbone forward pass; keeps only the requested layer's raw map.
    Runs once per layer (4x redundant compute across the sweep) to avoid
    holding all four layers' raw activations in memory simultaneously."""
    is_train = (mode == 'train')
    ds = DStrain(samples) if is_train else DStest(samples)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    layer_idx = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}[layer_name]
    maps = []
    pids = []; cams = []
    n_done = 0
    for item in ldr:
        if is_train:
            imgs = item.to(device, non_blocking=True)
        else:
            imgs, p, c = item
            imgs = imgs.to(device, non_blocking=True)
            pids.append(p.numpy()); cams.append(c.numpy())
        outs = backbone(imgs)
        m = outs[layer_idx]
        maps.append(m.cpu().numpy().astype(np.float32))
        n_done += imgs.shape[0]
        if n_done % 1024 == 0 or n_done == len(ds):
            print(f"      {tag} ({layer_name}) extract: {n_done}/{len(ds)}", flush=True)
    maps = np.concatenate(maps, axis=0)
    if is_train:
        return maps
    return maps, np.concatenate(pids), np.concatenate(cams)


def per_cell_nearest_index_batched(test_maps_np, train_maps_np, chunk_cells=CHUNK_CELLS):
    """Returns nn_idx: (n_cells, N_test) int32 -- nearest training-image index
    per cell per test item, by cosine similarity, chunked over cells on GPU."""
    N_test, C, H, W = test_maps_np.shape
    N_train = train_maps_np.shape[0]
    n_cells = H * W
    test_flat = test_maps_np.reshape(N_test, C, n_cells).transpose(2, 0, 1)
    train_flat = train_maps_np.reshape(N_train, C, n_cells).transpose(2, 0, 1)
    test_flat = test_flat / (np.linalg.norm(test_flat, axis=2, keepdims=True) + 1e-12)
    train_flat = train_flat / (np.linalg.norm(train_flat, axis=2, keepdims=True) + 1e-12)
    nn_idx = np.zeros((n_cells, N_test), dtype=np.int32)
    for start in range(0, n_cells, chunk_cells):
        end = min(start + chunk_cells, n_cells)
        t_chunk = torch.from_numpy(np.ascontiguousarray(test_flat[start:end])).to(device)
        tr_chunk = torch.from_numpy(np.ascontiguousarray(train_flat[start:end])).to(device)
        with torch.no_grad():
            sims = torch.bmm(t_chunk, tr_chunk.transpose(1, 2))
            idx = sims.argmax(dim=2).cpu().numpy()
        nn_idx[start:end] = idx
        del t_chunk, tr_chunk, sims
    return nn_idx


def per_cell_residual(maps_np, train_mean_cell):
    """maps_np: (N, C, H, W); train_mean_cell: (C, H, W). Returns (N, n_cells)."""
    diff = maps_np - train_mean_cell[None]
    resid = np.linalg.norm(diff, axis=1)
    N = maps_np.shape[0]
    return resid.reshape(N, -1)


def compute_cell_agreement(arm_name, model_ckpt, model_name, train_dataset_name,
                            eval_dataset_name, q_idx, g_idx,
                            layer_names=('layer1', 'layer2', 'layer3', 'layer4'),
                            train_pool_size=TRAIN_POOL_SIZE_CELL_DIAGNOSTIC,
                            batch=32, fracs=tuple(FRACS)):
    print(f"\n=== cell-agreement arm {arm_name} (model={model_name}, "
          f"train={train_dataset_name}, eval={eval_dataset_name}) ===", flush=True)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    print(f"  train pool for cell-bank: {len(train_samples)} / {len(train_all)} "
          f"(DECLARED subsample, memory-bounded; see module docstring)", flush=True)

    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
    query_samples = [q_all[int(i)] for i in q_idx]
    gallery_samples = [g_all[int(i)] for i in g_idx]
    print(f"  query: {len(query_samples)}  gallery: {len(gallery_samples)}", flush=True)

    backbone, bn = load_backbone_and_bn(model_ckpt)
    print(f"  backbone loaded: {model_ckpt.name}", flush=True)

    out = {
        'rev': 634,
        'arm': arm_name,
        'model_ckpt': str(model_ckpt),
        'train_dataset': train_dataset_name,
        'eval_dataset': eval_dataset_name,
        'diagnostic': 'cell-level divergent-vs-aligned agreement (Channel 4 rev 632 §4)',
        'params': {
            'N_QUERY': len(query_samples), 'N_GALLERY': len(gallery_samples),
            'train_pool_size_DECLARED_SUBSAMPLE': pool_size,
            'train_pool_full_set_size': len(train_all),
            'pool_is_subsampled': pool_size < len(train_all),
            'RNG_SEED': RNG_SEED, 'fracs': list(fracs),
        },
        'per_layer': {},
    }

    for Lname in layer_names:
        print(f"\n  --- layer {Lname} ---", flush=True)
        tr_maps = extract_layer_maps(backbone, train_samples, Lname, batch=batch,
                                      mode='train', tag='train')
        q_maps, q_pids, q_cams = extract_layer_maps(backbone, query_samples, Lname,
                                                      batch=batch, mode='test', tag='query')
        g_maps, g_pids, g_cams = extract_layer_maps(backbone, gallery_samples, Lname,
                                                      batch=batch, mode='test', tag='gallery')

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
        n_valid = len(valid_idx)

        P_mate_all = float(mate_agree_v.mean())
        P_rand_all = float(rand_agree_v.mean())
        print(f"    ALL cells: P_mate={P_mate_all:.5f}  P_rand={P_rand_all:.5f}  "
              f"diff={P_mate_all - P_rand_all:+.5f}  n_valid={n_valid}", flush=True)

        layer_out = {
            'n_valid': n_valid, 'n_cells': int(n_cells),
            'P_mate_all_cells': P_mate_all,
            'P_rand_all_cells': P_rand_all,
            'P_mate_minus_rand_all_cells': P_mate_all - P_rand_all,
            'k_sweep': {},
        }

        rank_order = np.argsort(-q_resid_v, axis=1)  # descending residual per query

        for f in fracs:
            k = max(1, int(round(f * n_cells)))
            top_idx = rank_order[:, :k]
            bot_idx = rank_order[:, -k:]
            mate_top = np.array([mate_agree_v[top_idx[j], j].mean() for j in range(n_valid)])
            rand_top = np.array([rand_agree_v[top_idx[j], j].mean() for j in range(n_valid)])
            mate_bot = np.array([mate_agree_v[bot_idx[j], j].mean() for j in range(n_valid)])
            rand_bot = np.array([rand_agree_v[bot_idx[j], j].mean() for j in range(n_valid)])
            divergent_signal = float(mate_top.mean() - rand_top.mean())
            aligned_signal = float(mate_bot.mean() - rand_bot.mean())
            print(f"    f={f:.2f} (k={k}): divergent mate-rand={divergent_signal:+.5f}  "
                  f"aligned mate-rand={aligned_signal:+.5f}  "
                  f"{'DIVERGENT > ALIGNED' if divergent_signal > aligned_signal else 'divergent <= aligned'}",
                  flush=True)
            layer_out['k_sweep'][f'f{f:.2f}'] = {
                'k': k,
                'P_mate_divergent': float(mate_top.mean()),
                'P_rand_divergent': float(rand_top.mean()),
                'P_mate_minus_rand_divergent': divergent_signal,
                'P_mate_aligned': float(mate_bot.mean()),
                'P_rand_aligned': float(rand_bot.mean()),
                'P_mate_minus_rand_aligned': aligned_signal,
            }

        out['per_layer'][Lname] = layer_out

    outp = RESULTS / f'e44_e_snap_a_cellagree_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['B_prime_sub', 'C', 'both'], default='B_prime_sub')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--train-pool-size', type=int, default=TRAIN_POOL_SIZE_CELL_DIAGNOSTIC)
    args = ap.parse_args()

    if args.arm in ('B_prime_sub', 'both'):
        q_idx, g_idx = load_saved_bprime_sub_indices()
        compute_cell_agreement('B_prime_sub', CKPT_MARKET, model_name='BoT-Market',
                                train_dataset_name='market1501', eval_dataset_name='msmt17',
                                q_idx=q_idx, g_idx=g_idx, batch=args.batch,
                                train_pool_size=args.train_pool_size)

    if args.arm in ('C', 'both'):
        rng = np.random.default_rng(RNG_SEED)
        eval_ds_obj = get_dataset('market1501', sample=False)
        q_all = [s for s in eval_ds_obj.query if not s.is_junk]
        g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
        compute_cell_agreement('C', CKPT_MSMT, model_name='BoT-MSMT',
                                train_dataset_name='msmt17', eval_dataset_name='market1501',
                                q_idx=q_idx, g_idx=g_idx, batch=args.batch,
                                train_pool_size=args.train_pool_size)


if __name__ == '__main__':
    main()
