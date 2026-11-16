"""
Stripe-wise pooling and Gram/channel-covariance descriptors (Daniel via
claude-chat, Channel 4 rev 676 sections 2c/2d, rev 677 items 10/11).
Companions to the camera test (679) and foreground-masking control (682) on
why early-layer identity-agreement is low.

(c) Stripe-wise pooling: split each layer's (C, H, W) map into 3 equal
    horizontal bands (head/torso/legs), mean-pool within each band, use the
    concatenated (3C,) descriptor for the snap. Preserves vertical structure
    GAP flattens -- all 4 layers.

(d) Gram / channel-covariance: per image, G = X X^T / (H*W) where X is the
    (C, H*W) flattened map -- a translation-invariant second-order
    descriptor (keeps co-occurring texture/colour, discards *where*).
    Upper-triangular (incl. diagonal) flattened as the snap vector. Early
    layers only (layer1, layer2) per spec and to keep the C*(C+1)/2
    dimensionality bounded (32,896 / 131,328).

Both rerun the same image-vs-identity (1)/(2)/(3) decomposition as 674/682,
reporting p_total_id_match vs random, for direct comparison to GAP.

Writes /data/papers/low_dimensional_reid/results/e44_stripe_gram_<arm>.json
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
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices, l2,
)
from compute_e_snap_a_cell_agreement import extract_layer_maps

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
GRAM_LAYERS = ['layer1', 'layer2']
N_BANDS = 3
N_RAND_DRAWS = 20


def stripe_descriptor(maps):
    """maps: (N, C, H, W). Returns (N, N_BANDS*C): mean-pool within each of
    N_BANDS equal horizontal bands, concatenated."""
    N, C, H, W = maps.shape
    edges = np.linspace(0, H, N_BANDS + 1).astype(int)
    bands = []
    for b in range(N_BANDS):
        h0, h1 = edges[b], edges[b + 1]
        bands.append(maps[:, :, h0:h1, :].mean(axis=(2, 3)))
    return np.concatenate(bands, axis=1)


def gram_descriptor(maps):
    """maps: (N, C, H, W). Returns (N, C*(C+1)/2): upper-triangular
    (incl. diagonal) of the per-image channel-covariance G = X X^T / (H*W)."""
    N, C, H, W = maps.shape
    X = maps.reshape(N, C, H * W)
    iu = np.triu_indices(C)
    out = np.zeros((N, len(iu[0])), dtype=np.float32)
    for i in range(N):
        G = (X[i] @ X[i].T) / (H * W)
        out[i] = G[iu]
    return out


def snap_indices_np(q_vec, train_vec):
    q = l2(q_vec); t = l2(train_vec)
    return (q @ t.T).argmax(axis=1)


def decompose(q_nn, g_nn, q_pids, q_cams, g_pids, g_cams, train_pids, rng_rand):
    n_q = len(q_pids)
    same_image = np.zeros(n_q, dtype=bool)
    same_id_diff_image = np.zeros(n_q, dtype=bool)
    valid = np.zeros(n_q, dtype=bool)
    rand_same_id = np.full(n_q, np.nan)
    for i in range(n_q):
        same_pid = g_pids == q_pids[i]
        junk = same_pid & (g_cams == q_cams[i])
        mates = np.where(same_pid & ~junk)[0]
        if len(mates) == 0:
            continue
        valid[i] = True
        q_img = q_nn[i]
        mate_imgs = g_nn[mates]
        if np.any(mate_imgs == q_img):
            same_image[i] = True
        elif np.any(train_pids[mate_imgs] == train_pids[q_img]):
            same_id_diff_image[i] = True
        diff_pid_gallery = np.where(g_pids != q_pids[i])[0]
        if len(diff_pid_gallery) > 0:
            draws = rng_rand.choice(diff_pid_gallery, size=min(N_RAND_DRAWS, len(diff_pid_gallery)), replace=False)
            draw_imgs = g_nn[draws]
            rand_same_id[i] = float((train_pids[draw_imgs] == train_pids[q_img]).mean())
    valid_n = int(valid.sum())
    p_same_image = float(same_image[valid].mean())
    p_same_id_diff = float(same_id_diff_image[valid].mean())
    p_total = p_same_image + p_same_id_diff
    r_same_id = float(np.nanmean(rand_same_id[valid]))
    return p_total, r_same_id, valid_n


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32, train_pool_size=3000):
    print(f"\n=== stripe/gram pooling, arm {arm_name} (model={model_name}) ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    train_pids = np.array([int(s.person_id) for s in train_samples])

    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
    if q_idx is None:
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
    query_samples = [q_all[int(i)] for i in q_idx]
    gallery_samples = [g_all[int(i)] for i in g_idx]
    print(f"  train pool: {len(train_samples)}  query: {len(query_samples)}  gallery: {len(gallery_samples)}", flush=True)

    out = {'rev': 676, 'arm': arm_name, 'model': model_name, 'n_bands': N_BANDS,
           'stripe': {}, 'gram': {}}
    rng_rand = np.random.default_rng(RNG_SEED + 800)

    for Lname in LAYER_NAMES:
        tr_maps = extract_layer_maps(backbone, train_samples, Lname, batch=batch, mode='train', tag='train')
        q_maps, q_pids, q_cams = extract_layer_maps(backbone, query_samples, Lname, batch=batch, mode='test', tag='query')
        g_maps, g_pids, g_cams = extract_layer_maps(backbone, gallery_samples, Lname, batch=batch, mode='test', tag='gallery')

        tr_stripe = stripe_descriptor(tr_maps)
        q_stripe = stripe_descriptor(q_maps)
        g_stripe = stripe_descriptor(g_maps)
        q_nn = snap_indices_np(q_stripe, tr_stripe)
        g_nn = snap_indices_np(g_stripe, tr_stripe)
        p_total, r_same, n_valid = decompose(q_nn, g_nn, q_pids, q_cams, g_pids, g_cams, train_pids, rng_rand)
        print(f"    {Lname} STRIPE: (1)+(2)={100*p_total:.2f}%  random={100*r_same:.3f}%  n_valid={n_valid}", flush=True)
        out['stripe'][Lname] = {'p_total_id_match': p_total, 'random_same_id': r_same, 'n_valid': n_valid}

        if Lname in GRAM_LAYERS:
            tr_gram = gram_descriptor(tr_maps)
            q_gram = gram_descriptor(q_maps)
            g_gram = gram_descriptor(g_maps)
            q_nn_g = snap_indices_np(q_gram, tr_gram)
            g_nn_g = snap_indices_np(g_gram, tr_gram)
            p_total_g, r_same_g, n_valid_g = decompose(q_nn_g, g_nn_g, q_pids, q_cams, g_pids, g_cams, train_pids, rng_rand)
            print(f"    {Lname} GRAM:   (1)+(2)={100*p_total_g:.2f}%  random={100*r_same_g:.3f}%  n_valid={n_valid_g}  dim={tr_gram.shape[1]}", flush=True)
            out['gram'][Lname] = {'p_total_id_match': p_total_g, 'random_same_id': r_same_g,
                                   'n_valid': n_valid_g, 'descriptor_dim': int(tr_gram.shape[1])}

        del tr_maps, q_maps, g_maps

    outp = RESULTS / f'e44_stripe_gram_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    q_idx, g_idx = load_saved_bprime_sub_indices()
    run_arm('B_prime_sub', CKPT_MARKET, 'BoT-Market', 'market1501', 'msmt17', q_idx=q_idx, g_idx=g_idx)
    run_arm('C', CKPT_MSMT, 'BoT-MSMT', 'msmt17', 'market1501', q_idx=None, g_idx=None)


if __name__ == '__main__':
    main()
