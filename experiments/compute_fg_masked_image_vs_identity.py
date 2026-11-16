"""
Foreground-masked pooling control (Daniel via claude-chat, Channel 4 rev
676 section 2b / rev 677 item 9). Companion to the camera test (679):
removes the ~45% background term from GAP pooling at every layer (existing
DeepLabV3 person-segmentation criterion, same as the cell-agreement-fg
control) and re-runs the image-vs-identity "mapping worked" measure
((1)+(2) total ID match per layer) to see whether background exclusion
alone closes the early-layer gap, independent of the camera-association
result.

DeepLabV3 runs ONCE per image at input resolution (256x128), then the
per-pixel mask is average-pooled down to each layer's own (H, W) grid and
thresholded at 0.5 fill -- avoids 4x redundant segmentation passes.

Writes /data/papers/low_dimensional_reid/results/e44_fg_masked_image_vs_identity_<arm>.json
"""
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
    RNG_SEED, N_QUERY, N_GALLERY, device, load_backbone_and_bn,
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
    DStest, DStrain, l2,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
LAYER_HW = {'layer1': (64, 32), 'layer2': (32, 16), 'layer3': (16, 8), 'layer4': (16, 8)}
N_RAND_DRAWS = 20
SEG_INPUT_HW = (256, 128)
PERSON_CLASS = 15


@torch.no_grad()
def extract_maps_and_masks(backbone, samples, batch=32, mode='test', tag='', seg_model=None):
    """Returns per-layer (N, C, H, W) float maps AND per-layer (N, H, W) bool
    foreground masks (shared single DeepLabV3 pass, pooled per layer), plus
    pids/cams for test mode."""
    if seg_model is None:
        seg_model = seg.deeplabv3_resnet50(
            weights=seg.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1).to(device).eval()
    is_train = (mode == 'train')
    ds = DStrain(samples) if is_train else DStest(samples)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    maps_chunks = {L: [] for L in LAYER_NAMES}
    mask_chunks = {L: [] for L in LAYER_NAMES}
    pid_chunks = []; cam_chunks = []
    n_done = 0
    for item in ldr:
        if is_train:
            imgs = item.to(device, non_blocking=True)
        else:
            imgs, p, c = item
            imgs = imgs.to(device, non_blocking=True)
            pid_chunks.append(p.numpy()); cam_chunks.append(c.numpy())
        l1, l2m, l3, l4 = backbone(imgs)
        for L, m in zip(LAYER_NAMES, [l1, l2m, l3, l4]):
            maps_chunks[L].append(m.cpu().numpy().astype(np.float32))

        seg_out = seg_model(imgs)['out'].argmax(dim=1)
        is_person = (seg_out == PERSON_CLASS).float().unsqueeze(1)  # (B,1,256,128)
        for L in LAYER_NAMES:
            H, W = LAYER_HW[L]
            cell = Fnn.adaptive_avg_pool2d(is_person, (H, W)).squeeze(1)
            mask_chunks[L].append((cell >= 0.5).cpu().numpy().astype(bool))

        n_done += imgs.shape[0]
        if n_done % 512 == 0 or n_done == len(ds):
            print(f"      {tag} extract: {n_done}/{len(ds)}", flush=True)

    maps = {L: np.concatenate(v, axis=0) for L, v in maps_chunks.items()}
    masks = {L: np.concatenate(v, axis=0) for L, v in mask_chunks.items()}
    if is_train:
        return maps, masks, seg_model
    return maps, masks, np.concatenate(pid_chunks), np.concatenate(cam_chunks), seg_model


def masked_gap(maps, masks):
    """maps: (N, C, H, W). masks: (N, H, W) bool. Returns (N, C) mean over
    foreground cells; falls back to full GAP when a sample has none."""
    N, C, H, W = maps.shape
    out = np.zeros((N, C), dtype=np.float32)
    flat_maps = maps.reshape(N, C, H * W)
    flat_masks = masks.reshape(N, H * W)
    for i in range(N):
        m = flat_masks[i]
        if m.sum() == 0:
            out[i] = flat_maps[i].mean(axis=1)
        else:
            out[i] = flat_maps[i][:, m].mean(axis=1)
    return out


def snap_indices_np(q_vec, train_vec):
    q = l2(q_vec); t = l2(train_vec)
    return (q @ t.T).argmax(axis=1)


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=24, train_pool_size=3000):
    print(f"\n=== fg-masked image-vs-identity, arm {arm_name} (model={model_name}) ===", flush=True)
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

    tr_maps, tr_masks, seg_model = extract_maps_and_masks(backbone, train_samples, batch=batch, mode='train', tag='train')
    q_maps, q_masks, q_pids, q_cams, seg_model = extract_maps_and_masks(backbone, query_samples, batch=batch, mode='test', tag='query', seg_model=seg_model)
    g_maps, g_masks, g_pids, g_cams, seg_model = extract_maps_and_masks(backbone, gallery_samples, batch=batch, mode='test', tag='gallery', seg_model=seg_model)

    out = {'rev': 676, 'arm': arm_name, 'model': model_name, 'per_layer': {}}
    rng_rand = np.random.default_rng(RNG_SEED + 800)

    for Lname in LAYER_NAMES:
        tr_fg_frac = float(tr_masks[Lname].mean())
        tr_vec = masked_gap(tr_maps[Lname], tr_masks[Lname])
        q_vec = masked_gap(q_maps[Lname], q_masks[Lname])
        g_vec = masked_gap(g_maps[Lname], g_masks[Lname])

        q_nn = snap_indices_np(q_vec, tr_vec)
        g_nn = snap_indices_np(g_vec, tr_vec)

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

        n_valid = int(valid.sum())
        p_same_image = float(same_image[valid].mean())
        p_same_id_diff = float(same_id_diff_image[valid].mean())
        p_total = p_same_image + p_same_id_diff
        r_same_id = float(np.nanmean(rand_same_id[valid]))

        print(f"    {Lname}: fg_frac(train)={100*tr_fg_frac:.1f}%  "
              f"(1)+(2) total ID match={100*p_total:.2f}%  random={100*r_same_id:.3f}%  n_valid={n_valid}", flush=True)

        out['per_layer'][Lname] = {
            'n_valid': n_valid, 'train_fg_cell_fraction': tr_fg_frac,
            'p_same_image': p_same_image, 'p_same_id_diff_image': p_same_id_diff,
            'p_total_id_match': p_total, 'random_same_id': r_same_id,
        }

    outp = RESULTS / f'e44_fg_masked_image_vs_identity_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    q_idx, g_idx = load_saved_bprime_sub_indices()
    out_b = run_arm('B_prime_sub', CKPT_MARKET, 'BoT-Market', 'market1501', 'msmt17', q_idx=q_idx, g_idx=g_idx)
    out_c = run_arm('C', CKPT_MSMT, 'BoT-MSMT', 'msmt17', 'market1501', q_idx=None, g_idx=None)

    print("\n=== SUMMARY: (1)+(2) total ID match, fg-masked vs original ===", flush=True)
    orig = {
        'B_prime_sub': [5.52, 4.70, 7.73, 15.47],
        'C': [23.11, 24.13, 33.95, 47.24],
    }
    for label, out in [('B_prime_sub', out_b), ('C', out_c)]:
        masked_vals = [100 * out['per_layer'][L]['p_total_id_match'] for L in LAYER_NAMES]
        print(f"{label:15s}  masked: " + "  ".join(f"{v:5.2f}%" for v in masked_vals) +
              "   |  original: " + "  ".join(f"{v:5.2f}%" for v in orig[label]), flush=True)


if __name__ == '__main__':
    main()
