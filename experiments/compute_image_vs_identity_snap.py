"""
Image-level decomposition of the ID-level snap (Daniel via claude-chat,
Channel 4 rev 665).

"As a really dumb test, you just pick the unique train_id_pic_id that the
OOD sample most aligns to." Decomposes 659's consistency number into two
mechanisms: is the snap locking onto a specific training PHOTO (view-locked)
or onto the training PERSON across photos (person-locked)?

Per layer, per arm, per query-mate pair, three mutually exclusive outcomes:
  (1) same training image      -- query and mate argmax to the identical
                                   training photo index
  (2) different image, same ID -- different training photos, same identity
  (3) different identity       -- failure

(1)+(2) sum to 659's P(mate matches). If (1)~(2), the mapping is
view-locked; if (2)>>(1), it generalises across pose/lighting -- the
person-locked reading.

Measured random-pair baseline for (1) and (1)+(2), same convention as
662 §1's recall@k random baseline, since hub concentration means chance
agreement on the exact image is far above the naive 1/N_train.

Writes /data/papers/low_dimensional_reid/results/e44_image_vs_identity_<arm>.json
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
    extract_gap_vectors_and_terminal, snap_indices,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
N_RAND_DRAWS = 20


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32, train_pool_size=3000):
    print(f"\n=== image-vs-identity snap, arm {arm_name} (model={model_name}) ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    train_pids = np.array([int(s.person_id) for s in train_samples])
    print(f"  train pool: {len(train_samples)}", flush=True)

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

    out = {'rev': 665, 'arm': arm_name, 'model': model_name,
           'n_query': len(query_samples), 'n_gallery': len(gallery_samples),
           'train_pool_size': pool_size, 'per_layer': {}}

    rng_rand = np.random.default_rng(RNG_SEED + 800)

    for Lname in LAYER_NAMES:
        print(f"\n  --- {Lname} ---", flush=True)
        q_nn = snap_indices(q_gap[Lname], tr_gap[Lname])  # training IMAGE index
        g_nn = snap_indices(g_gap[Lname], tr_gap[Lname])

        n_q = len(q_pids)
        same_image = np.zeros(n_q, dtype=bool)
        same_id_diff_image = np.zeros(n_q, dtype=bool)
        diff_id = np.zeros(n_q, dtype=bool)
        valid = np.zeros(n_q, dtype=bool)
        rand_same_image = np.full(n_q, np.nan)
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
            else:
                diff_id[i] = True

            diff_pid_gallery = np.where(g_pids != q_pids[i])[0]
            if len(diff_pid_gallery) > 0:
                draws = rng_rand.choice(diff_pid_gallery, size=min(N_RAND_DRAWS, len(diff_pid_gallery)),
                                         replace=False)
                draw_imgs = g_nn[draws]
                rand_same_image[i] = float((draw_imgs == q_img).mean())
                rand_same_id[i] = float((train_pids[draw_imgs] == train_pids[q_img]).mean())

        n_valid = int(valid.sum())
        p_same_image = float(same_image[valid].mean())
        p_same_id_diff = float(same_id_diff_image[valid].mean())
        p_diff_id = float(diff_id[valid].mean())
        p_total_id_match = p_same_image + p_same_id_diff  # should equal 659's P(mate matches)

        r_same_image = float(np.nanmean(rand_same_image[valid]))
        r_same_id = float(np.nanmean(rand_same_id[valid]))

        print(f"    (1) same training image:        {100*p_same_image:.2f}%  (random: {100*r_same_image:.3f}%)", flush=True)
        print(f"    (2) diff image, same identity:   {100*p_same_id_diff:.2f}%", flush=True)
        print(f"    (3) different identity:          {100*p_diff_id:.2f}%", flush=True)
        print(f"    (1)+(2) total ID match:          {100*p_total_id_match:.2f}%  (random: {100*r_same_id:.3f}%)  "
              f"[cross-check vs rev 659's P(mate matches)]", flush=True)
        ratio = p_same_id_diff / p_same_image if p_same_image > 0 else float('inf')
        print(f"    (2)/(1) ratio: {ratio:.2f}  {'PERSON-LOCKED (generalizes across photos)' if ratio > 2 else 'VIEW-LOCKED (mostly same-photo matches)' if ratio < 0.5 else 'mixed'}", flush=True)

        out['per_layer'][Lname] = {
            'n_valid': n_valid,
            'p_same_image': p_same_image, 'p_same_id_diff_image': p_same_id_diff,
            'p_diff_identity': p_diff_id, 'p_total_id_match': p_total_id_match,
            'random_same_image': r_same_image, 'random_same_id': r_same_id,
            'ratio_diffimg_to_sameimg': ratio,
        }

    outp = RESULTS / f'e44_image_vs_identity_{arm_name}.json'
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
