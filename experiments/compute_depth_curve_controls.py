"""
Three controls on the rising "mapping worked" depth curve (Daniel via
claude-chat, Channel 4 rev 673). The image-vs-identity (1)+(2) total ID-match
rate rises with depth on both OOD arms (B'-sub 5.5->4.7->7.7->15.5%;
C 23.1->24.1->34.0->47.2%). "Averaging out early-layer patch noise" is
already dead by arithmetic (layer1 GAP averages MORE cells, 2048 vs
layer4's 128). Three surviving explanations, one control each:

  (a) Learned invariance   -> in-domain arms (A', D) should show the SAME
                               rising curve if it's just the training
                               objective, not anything OOD-specific.
  (b) Pooling artefact      -> an ImageNet-init (never trained for re-ID)
                               network on the SAME OOD arms should show the
                               same curve if it's architectural.
  (c) Dimension             -> random-projecting every layer's GAP vector
                               to a common dim (256, layer1's native size)
                               before the cosine-NN snap should flatten the
                               curve if it's a |cos|~1/sqrt(d) effect.

All three reuse the exact same decomposition as
compute_image_vs_identity_snap.py (rev 665): per query-mate pair, snap to
nearest training IMAGE, classify as same-image / diff-image-same-ID /
diff-identity. The curve in question is p_total_id_match = (1)+(2) per layer.

Writes /data/papers/low_dimensional_reid/results/e44_depth_curve_controls.json
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import (
    RNG_SEED, N_QUERY, N_GALLERY, device, BoTBackbone4L, load_backbone_and_bn,
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
    extract_gap_vectors_and_terminal, snap_indices, l2,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
LAYER_DIMS = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
N_RAND_DRAWS = 20
PROJECT_DIM = 256


def load_imagenet_backbone():
    """Same architecture (layer4 stride override) as BoTBackbone4L, ImageNet
    weights instead of trained -- the (b) pooling-artefact null model."""
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    backbone = BoTBackbone4L().to(device).eval()
    backbone.stem_conv.load_state_dict(resnet.conv1.state_dict())
    backbone.stem_bn.load_state_dict(resnet.bn1.state_dict())
    backbone.layer1.load_state_dict(resnet.layer1.state_dict())
    backbone.layer2.load_state_dict(resnet.layer2.state_dict())
    backbone.layer3.load_state_dict(resnet.layer3.state_dict())
    backbone.layer4.load_state_dict(resnet.layer4.state_dict())
    return backbone


def dummy_bn():
    """extract_gap_vectors_and_terminal always computes a terminal embedding;
    this measure never uses it, so an untrained BN is fine as a placeholder."""
    return nn.BatchNorm1d(2048, affine=True, track_running_stats=True).to(device).eval()


def random_project(X, R):
    return X @ R


def make_projection_matrices(seed):
    rng = np.random.default_rng(seed)
    mats = {}
    for Lname in LAYER_NAMES:
        d = LAYER_DIMS[Lname]
        R = rng.standard_normal((d, PROJECT_DIM)).astype(np.float32) / np.sqrt(d)
        mats[Lname] = R
    return mats


def run_measure(label, backbone, bn, train_dataset_name, eval_dataset_name,
                 q_idx=None, g_idx=None, batch=32, train_pool_size=3000,
                 project_mats=None):
    print(f"\n=== depth-curve control: {label} (train={train_dataset_name}, eval={eval_dataset_name}, "
          f"projected={project_mats is not None}) ===", flush=True)
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

    out = {'rev': 673, 'label': label, 'train_dataset': train_dataset_name, 'eval_dataset': eval_dataset_name,
           'n_query': len(query_samples), 'n_gallery': len(gallery_samples),
           'train_pool_size': pool_size, 'projected': project_mats is not None,
           'per_layer': {}}

    rng_rand = np.random.default_rng(RNG_SEED + 800)

    for Lname in LAYER_NAMES:
        tr_vec = tr_gap[Lname]
        q_vec = q_gap[Lname]
        g_vec = g_gap[Lname]
        if project_mats is not None:
            R = project_mats[Lname]
            tr_vec = random_project(tr_vec, R)
            q_vec = random_project(q_vec, R)
            g_vec = random_project(g_vec, R)

        q_nn = snap_indices(q_vec, tr_vec)
        g_nn = snap_indices(g_vec, tr_vec)

        n_q = len(q_pids)
        same_image = np.zeros(n_q, dtype=bool)
        same_id_diff_image = np.zeros(n_q, dtype=bool)
        diff_id = np.zeros(n_q, dtype=bool)
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
            else:
                diff_id[i] = True

            diff_pid_gallery = np.where(g_pids != q_pids[i])[0]
            if len(diff_pid_gallery) > 0:
                draws = rng_rand.choice(diff_pid_gallery, size=min(N_RAND_DRAWS, len(diff_pid_gallery)),
                                         replace=False)
                draw_imgs = g_nn[draws]
                rand_same_id[i] = float((train_pids[draw_imgs] == train_pids[q_img]).mean())

        n_valid = int(valid.sum())
        p_same_image = float(same_image[valid].mean())
        p_same_id_diff = float(same_id_diff_image[valid].mean())
        p_total_id_match = p_same_image + p_same_id_diff
        r_same_id = float(np.nanmean(rand_same_id[valid]))

        print(f"    {Lname}: (1)+(2) total ID match={100*p_total_id_match:.2f}%  "
              f"(1)={100*p_same_image:.2f}%  (2)={100*p_same_id_diff:.2f}%  "
              f"random={100*r_same_id:.3f}%  n_valid={n_valid}", flush=True)

        out['per_layer'][Lname] = {
            'n_valid': n_valid,
            'p_same_image': p_same_image, 'p_same_id_diff_image': p_same_id_diff,
            'p_total_id_match': p_total_id_match, 'random_same_id': r_same_id,
        }

    return out


def main():
    results = {}

    # (a) Learned-invariance control: in-domain arms, same measure.
    out_a_prime = run_measure('A_prime', *load_backbone_and_bn(CKPT_MARKET),
                               train_dataset_name='market1501', eval_dataset_name='market1501')
    out_d = run_measure('D', *load_backbone_and_bn(CKPT_MSMT),
                         train_dataset_name='msmt17', eval_dataset_name='msmt17')
    results['A_prime'] = out_a_prime
    results['D'] = out_d

    # (b) Pooling-artefact control: ImageNet-init backbone, same OOD arms.
    imagenet_backbone = load_imagenet_backbone()
    bn_dummy = dummy_bn()
    q_idx, g_idx = load_saved_bprime_sub_indices()
    out_b_in = run_measure('B_prime_sub_IN', imagenet_backbone, bn_dummy,
                            train_dataset_name='market1501', eval_dataset_name='msmt17',
                            q_idx=q_idx, g_idx=g_idx)
    out_c_in = run_measure('C_IN', imagenet_backbone, bn_dummy,
                            train_dataset_name='msmt17', eval_dataset_name='market1501')
    results['B_prime_sub_IN'] = out_b_in
    results['C_IN'] = out_c_in

    # (c) Dimension control: common-dim random projection, same OOD arms, trained backbones.
    proj_mats = make_projection_matrices(seed=RNG_SEED + 900)
    out_b_proj = run_measure('B_prime_sub_proj', *load_backbone_and_bn(CKPT_MARKET),
                              train_dataset_name='market1501', eval_dataset_name='msmt17',
                              q_idx=q_idx, g_idx=g_idx, project_mats=proj_mats)
    out_c_proj = run_measure('C_proj', *load_backbone_and_bn(CKPT_MSMT),
                              train_dataset_name='msmt17', eval_dataset_name='market1501',
                              project_mats=proj_mats)
    results['B_prime_sub_proj'] = out_b_proj
    results['C_proj'] = out_c_proj

    outp = RESULTS / 'e44_depth_curve_controls.json'
    outp.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {outp}", flush=True)

    print("\n=== SUMMARY: (1)+(2) total ID match by layer ===", flush=True)
    print(f"{'label':20s}  layer1  layer2  layer3  layer4", flush=True)
    for label, out in results.items():
        vals = [100 * out['per_layer'][L]['p_total_id_match'] for L in LAYER_NAMES]
        print(f"{label:20s}  " + "  ".join(f"{v:5.2f}%" for v in vals), flush=True)


if __name__ == '__main__':
    main()
