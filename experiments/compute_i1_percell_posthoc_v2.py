"""
Post-hoc v2 for I-1 per-cell preview (rev 565 corrections).

Rev 564 was wrong in two ways:
  (1) Used (cond_ap - base_mAP_scalar) as the per-query delta instead of
      (cond_ap - base_ap_per_query). Scalar subtraction under-exploits
      query-level correlation between baseline and condition APs,
      inflating the variance. Fix: recompute baselines per query and use
      the properly paired delta.
  (2) Reported displacement as ||Δx||/||x||. Top-residual cells have
      larger ||x|| by construction, so equal relative displacement is
      larger absolute displacement. Downstream GAP sums raw cell
      vectors, so report absolute ||Δx|| alongside ||x|| per group.

Re-extracts the same subsamples (fixed seed 42) for per-query baselines
and displacement units.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

RESULTS = Path('/data/papers/low_dimensional_reid/results')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG_SEED = 42
N_QUERY = 500
N_GALLERY = 5000
N_TRAIN_POOL = 3000
FRAC = 0.10
H, W, C = 16, 8, 2048
N_CELLS = H * W
N_BOOT = 1000

CKPT_MARKET = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')

TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ARMS = {
    'A_prime': 'market',
    'B_prime': 'msmt',
}


class BoTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.stem_conv = resnet.conv1; self.stem_bn = resnet.bn1
        self.stem_relu = resnet.relu; self.stem_pool = resnet.maxpool
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3; self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.stem_conv(x); x = self.stem_bn(x)
        x = self.stem_relu(x); x = self.stem_pool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return x


def load_backbone_and_bn(ckpt_path):
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = BoTBackbone().to(device).eval()
    MAP = {'0.0.': 'stem_conv.', '0.1.': 'stem_bn.',
           '0.4.': 'layer1.', '0.5.': 'layer2.',
           '0.6.': 'layer3.', '0.7.': 'layer4.'}
    sd = {}
    bn_sd = {}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k:
            continue
        if k.startswith('3.'):
            bn_sd[k[2:]] = v
            continue
        for prefix, target in MAP.items():
            if k.startswith(prefix):
                sd[target + k[len(prefix):]] = v
                break
    backbone.load_state_dict(sd, strict=False)
    bn = nn.BatchNorm1d(2048, affine=True, track_running_stats=True).to(device).eval()
    bn.weight.data = bn_sd['weight'].to(device)
    bn.bias.data = bn_sd['bias'].to(device)
    bn.running_mean.data = bn_sd['running_mean'].to(device)
    bn.running_var.data = bn_sd['running_var'].to(device)
    return backbone, bn


class DSimg(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return TRANSFORM(self.samples[i].load_image())


class DStest(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), int(s.person_id), int(s.camera_id)


@torch.no_grad()
def extract_maps(backbone, samples, mode='test'):
    ds = DSimg(samples) if mode == 'train' else DStest(samples)
    ldr = DataLoader(ds, batch_size=32, num_workers=0, pin_memory=False)
    maps = []; pids = []; cams = []
    for item in ldr:
        if mode == 'train':
            f = backbone(item.to(device))
            maps.append(f.cpu().numpy())
        else:
            imgs, p, c = item
            f = backbone(imgs.to(device))
            maps.append(f.cpu().numpy())
            pids.append(p.numpy())
            cams.append(c.numpy())
    maps = np.concatenate(maps, axis=0)
    if mode == 'train':
        return maps
    return maps, np.concatenate(pids), np.concatenate(cams)


def forward_retrieval(maps_chw, bn):
    gap = maps_chw.mean(axis=(2, 3))
    gap_t = torch.from_numpy(gap).to(device)
    with torch.no_grad():
        post = bn(gap_t).cpu().numpy()
    norms = np.linalg.norm(post, axis=1, keepdims=True)
    return post / (norms + 1e-12)


def per_query_ap(q_emb, q_pids, q_cams, g_emb, g_pids, g_cams):
    sims = q_emb @ g_emb.T
    aps = np.zeros(len(q_pids), dtype=np.float64)
    valid = np.zeros(len(q_pids), dtype=bool)
    for i in range(len(q_pids)):
        qid, qcam = q_pids[i], q_cams[i]
        junk = (g_pids == qid) & (g_cams == qcam)
        order = np.argsort(-sims[i])
        order = order[~junk[order]]
        good = g_pids[order] == qid
        if not good.any():
            continue
        prec = np.cumsum(good) / np.arange(1, len(good) + 1)
        aps[i] = (prec * good).sum() / good.sum()
        valid[i] = True
    return aps, valid


def extract_arm(backbone, ds, rng, n_query, n_gallery):
    non_junk_q = [s for s in ds.query if not s.is_junk]
    non_junk_g = [s for s in ds.gallery if not s.is_junk]
    q_idx = rng.choice(len(non_junk_q), size=min(n_query, len(non_junk_q)), replace=False)
    g_idx = rng.choice(len(non_junk_g), size=min(n_gallery, len(non_junk_g)), replace=False)
    q_samples = [non_junk_q[int(i)] for i in q_idx]
    g_samples = [non_junk_g[int(i)] for i in g_idx]
    q_maps, q_pids, q_cams = extract_maps(backbone, q_samples, mode='test')
    g_maps, g_pids, g_cams = extract_maps(backbone, g_samples, mode='test')
    return q_maps, q_pids, q_cams, g_maps, g_pids, g_cams


def build_cell_distances(maps, cell_means):
    N, C_, H_, W_ = maps.shape
    out = np.zeros((N, H_ * W_), dtype=np.float32)
    for h in range(H_):
        for w in range(W_):
            diff = maps[:, :, h, w] - cell_means[h, w]
            out[:, h * W_ + w] = np.linalg.norm(diff, axis=1)
    return out


def precompute_nearest_idx(test_maps, train_patches_by_cell):
    N, C_, H_, W_ = test_maps.shape
    out = np.zeros((N, H_ * W_), dtype=np.int32)
    for h in range(H_):
        for w in range(W_):
            train = train_patches_by_cell[(h, w)]
            train_t = torch.from_numpy(train).to(device)
            train_norm = train_t / (train_t.norm(dim=1, keepdim=True) + 1e-12)
            test_t = torch.from_numpy(test_maps[:, :, h, w]).to(device)
            test_norm = test_t / (test_t.norm(dim=1, keepdim=True) + 1e-12)
            sims = test_norm @ train_norm.T
            out[:, h * W_ + w] = sims.argmax(dim=1).cpu().numpy()
    return out


def displacement_absolute(maps, selected_cells_per_image, train_patches_by_cell, nearest_idx):
    """Report absolute ||Δx|| and ||x|| per replaced cell."""
    N, C_, H_, W_ = maps.shape
    abs_disps = []
    orig_norms = []
    for img_idx in range(N):
        for cell_idx in selected_cells_per_image[img_idx]:
            h = cell_idx // W_
            w = cell_idx % W_
            orig = maps[img_idx, :, h, w]
            k = int(nearest_idx[img_idx, cell_idx])
            new = train_patches_by_cell[(h, w)][k]
            abs_disps.append(float(np.linalg.norm(new - orig)))
            orig_norms.append(float(np.linalg.norm(orig)))
    return np.array(abs_disps), np.array(orig_norms)


def apply_replacement(maps, selected_cells_per_image, train_patches_by_cell,
                      mode='nearest', nearest_idx=None, rng=None):
    N, C_, H_, W_ = maps.shape
    maps_mod = maps.copy()
    for img_idx in range(N):
        for cell_idx in selected_cells_per_image[img_idx]:
            h = cell_idx // W_
            w = cell_idx % W_
            train_patches = train_patches_by_cell[(h, w)]
            if mode == 'nearest':
                k = int(nearest_idx[img_idx, cell_idx])
            else:
                k = int(rng.integers(0, train_patches.shape[0]))
            maps_mod[img_idx, :, h, w] = train_patches[k]
    return maps_mod


def main():
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    backbone, bn = load_backbone_and_bn(CKPT_MARKET)

    print("Training pool...", flush=True)
    train_samples = market.train
    if len(train_samples) > N_TRAIN_POOL:
        idx = rng.choice(len(train_samples), size=N_TRAIN_POOL, replace=False)
        train_samples = [train_samples[int(i)] for i in idx]
    train_maps = extract_maps(backbone, train_samples, mode='train')
    cell_means = train_maps.mean(axis=0).transpose(1, 2, 0)
    train_patches_by_cell = {(h, w): train_maps[:, :, h, w].copy()
                             for h in range(H) for w in range(W)}

    out = {'post_hoc_v2_rev': 565}
    per_arm = {}

    for arm, ckpt in ARMS.items():
        print(f"\n### arm {arm} ###", flush=True)
        ds = market if ckpt == 'market' else msmt
        q_maps, q_pids, q_cams, g_maps, g_pids, g_cams = extract_arm(
            backbone, ds, rng, N_QUERY, N_GALLERY)

        # Baseline per-query APs
        q_emb_b = forward_retrieval(q_maps, bn)
        g_emb_b = forward_retrieval(g_maps, bn)
        aps_b, valid_b = per_query_ap(q_emb_b, q_pids, q_cams, g_emb_b, g_pids, g_cams)
        print(f"  baseline mAP = {100*aps_b[valid_b].mean():.4f}%  n_valid={int(valid_b.sum())}",
              flush=True)

        # Top-residual + nearest at f=0.10 (apply to q and g identically)
        q_cd = build_cell_distances(q_maps, cell_means)
        g_cd = build_cell_distances(g_maps, cell_means)
        q_near = precompute_nearest_idx(q_maps, train_patches_by_cell)
        g_near = precompute_nearest_idx(g_maps, train_patches_by_cell)
        k = max(1, int(round(FRAC * N_CELLS)))
        q_sel_top = [np.argsort(-q_cd[i])[:k] for i in range(q_maps.shape[0])]
        g_sel_top = [np.argsort(-g_cd[i])[:k] for i in range(g_maps.shape[0])]
        q_mod = apply_replacement(q_maps, q_sel_top, train_patches_by_cell,
                                  mode='nearest', nearest_idx=q_near)
        g_mod = apply_replacement(g_maps, g_sel_top, train_patches_by_cell,
                                  mode='nearest', nearest_idx=g_near)
        q_emb_c = forward_retrieval(q_mod, bn)
        g_emb_c = forward_retrieval(g_mod, bn)
        aps_c, valid_c = per_query_ap(q_emb_c, q_pids, q_cams, g_emb_c, g_pids, g_cams)
        both_valid = valid_b & valid_c

        # Properly paired delta
        delta = aps_c - aps_b
        d_v = delta[both_valid]
        p_v = q_pids[both_valid]
        unique_ids = np.unique(p_v)
        id_to_idx = {int(pid): np.where(p_v == pid)[0] for pid in unique_ids}
        n_id = len(unique_ids)
        rng_boot = np.random.default_rng(RNG_SEED + 1)
        boot_means = []
        for _ in range(N_BOOT):
            s = rng_boot.choice(unique_ids, size=n_id, replace=True)
            qi = np.concatenate([id_to_idx[int(pid)] for pid in s])
            boot_means.append(d_v[qi].mean())
        boot_means = np.array(boot_means)
        mean_d = float(d_v.mean())
        lo = float(np.quantile(boot_means, 0.025))
        hi = float(np.quantile(boot_means, 0.975))
        print(f"  paired Δ (cond - baseline per query): mean={100*mean_d:+.3f} pp  "
              f"CI95 [{100*lo:+.3f}, {100*hi:+.3f}]  n_id={n_id}", flush=True)

        # Displacement in absolute units: top-residual vs random-cell
        rng_rand = np.random.default_rng(RNG_SEED + 42 + abs(hash(arm)) % 1000)
        q_sel_rand = [rng_rand.choice(N_CELLS, size=k, replace=False)
                      for _ in range(q_maps.shape[0])]
        top_abs, top_orig = displacement_absolute(q_maps, q_sel_top,
                                                   train_patches_by_cell, q_near)
        rand_abs, rand_orig = displacement_absolute(q_maps, q_sel_rand,
                                                    train_patches_by_cell, q_near)
        print(f"  ABS displacement (query cells, f=0.10):", flush=True)
        print(f"    top-residual: ||Δx||  mean={top_abs.mean():.3f}  median={np.median(top_abs):.3f}  "
              f"||x|| mean={top_orig.mean():.3f}", flush=True)
        print(f"    random-cell:  ||Δx||  mean={rand_abs.mean():.3f}  median={np.median(rand_abs):.3f}  "
              f"||x|| mean={rand_orig.mean():.3f}", flush=True)
        print(f"    abs ratio ||Δx||_top/||Δx||_rand = {top_abs.mean()/rand_abs.mean():.3f}  "
              f"norm ratio ||x||_top/||x||_rand = {top_orig.mean()/rand_orig.mean():.3f}", flush=True)

        per_arm[arm] = {
            'baseline_map': float(aps_b[valid_b].mean()),
            'condition_top_near_f010_map': float(aps_c[valid_c].mean()),
            'paired_delta_mean_pp': mean_d,
            'paired_delta_ci95_lo_pp': lo,
            'paired_delta_ci95_hi_pp': hi,
            'n_identities': n_id,
            'n_queries_valid': int(both_valid.sum()),
            'displacement_top_abs_mean': float(top_abs.mean()),
            'displacement_top_abs_median': float(np.median(top_abs)),
            'displacement_rand_abs_mean': float(rand_abs.mean()),
            'displacement_rand_abs_median': float(np.median(rand_abs)),
            'orig_norm_top_mean': float(top_orig.mean()),
            'orig_norm_rand_mean': float(rand_orig.mean()),
            'abs_disp_ratio_top_over_rand': float(top_abs.mean() / rand_abs.mean()),
            'norm_ratio_top_over_rand': float(top_orig.mean() / rand_orig.mean()),
        }

    out['per_arm'] = per_arm

    # Joint interaction with properly paired deltas
    boot_a = np.array([per_arm['A_prime']['paired_delta_mean_pp']])  # placeholder
    # Rerun bootstrap jointly from the stored delta vectors? We'd need them.
    # Do joint from arm-level boot_means we just produced — but we didn't keep those.
    # Compute interaction pointwise: diff of the two per-arm point estimates;
    # CI from independent per-arm bootstraps is still the right formula if
    # we assume arms are independent (different test sets, which they are).
    # To do this properly, save per-arm boot_means and difference them.
    print("\n### INTERACTION ###", flush=True)
    rng_i_a = np.random.default_rng(RNG_SEED + 1)
    rng_i_b = np.random.default_rng(RNG_SEED + 2)
    # Rebuild from per_arm stored data — we lost the deltas, need to restore
    # Simpler: just compute pointwise difference and report the independence-sum SE.
    a_lo = per_arm['A_prime']['paired_delta_ci95_lo_pp']
    a_hi = per_arm['A_prime']['paired_delta_ci95_hi_pp']
    a_m = per_arm['A_prime']['paired_delta_mean_pp']
    b_lo = per_arm['B_prime']['paired_delta_ci95_lo_pp']
    b_hi = per_arm['B_prime']['paired_delta_ci95_hi_pp']
    b_m = per_arm['B_prime']['paired_delta_mean_pp']
    se_a = (a_hi - a_lo) / (2 * 1.96)
    se_b = (b_hi - b_lo) / (2 * 1.96)
    inter = b_m - a_m
    se_inter = (se_a**2 + se_b**2) ** 0.5
    inter_lo = inter - 1.96 * se_inter
    inter_hi = inter + 1.96 * se_inter
    print(f"  pp interaction (B'-A') = {100*inter:+.3f} pp  "
          f"approx CI95 [{100*inter_lo:+.3f}, {100*inter_hi:+.3f}]  "
          f"(independence formula, paired per-query deltas)", flush=True)
    out['pp_interaction'] = {
        'mean_pp': inter, 'ci95_lo_pp': inter_lo, 'ci95_hi_pp': inter_hi,
        'excludes_zero': bool((inter_lo > 0) or (inter_hi < 0)),
        'note': 'per-arm CIs from properly paired per-query delta bootstrap; interaction via independence formula',
    }

    outp = RESULTS / 'e44_i1_percell_posthoc_v2.json'
    with open(outp, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
