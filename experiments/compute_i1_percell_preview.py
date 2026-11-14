"""
I-1 per-cell preview v2 (rev 554 amended pre-registration, post rev 553).

Per-cell tail replacement on layer4 feature map, acting at the location
I-1 is specced for (intermediate feature maps, NOT GAP-pooled).

Applied identically to BOTH queries and gallery (the deployable test-time
case; any asymmetry puts q/g in differently-transformed spaces).

For each test image:
  1. Extract layer4 feature map (C=2048, H=16, W=8).
  2. Per cell (h, w): distance to training-mean at that cell.
  3. Rank cells by distance; select top f ∈ {0.01, 0.05, 0.10, 0.25}.
  4. Replace selected cells per condition.
  5. Forward modified map through GAP + BNNeck + L2.
  6. Compute per-query AP for paired analysis vs baseline.
  7. Report paired delta and paired 95% CI.

FOUR CONDITIONS per f per arm:
  - top-residual + NEAREST training patch   (the clamp as specced)
  - top-residual + RANDOM training patch    (lane-snap control: nearest vs random patch)
  - random-cell  + NEAREST training patch   (ranking control: which cells to touch)
  - baseline (unmodified)

Pre-registered hypotheses (rev 554 amendments):
  1. B' paired delta at the best f is >= 0.5 pp AND A' paired delta at
     matched f is <= 0 pp -> per-cell I-1 reduces OOD cost without
     denoising in-domain.
  2a. B' top-residual+nearest ~ random-cell+nearest -> RANKING does no work.
  2b. B' top-residual+nearest ~ top-residual+random-patch -> NEAREST-PATCH
      SEMANTIC MATCH does no work; replacement reduces to perturbing an
      outlier cell toward the training bulk.
  3. A' paired delta >= 0.5 pp at any f -> dataset-agnostic denoise.
  4. All conditions null at all f on both arms -> CENTROID-RANKED per-cell
     variant at f <= 0.25 retired on mAP. Does NOT retire the kNN/local-hull
     variant (rev 508 spec) or other f; those remain open per Daniel's
     one-null-does-not-kill-family rule.

Primary endpoint: paired delta at f_best = argmax_f(top-residual+nearest
delta on B'), with paired bootstrap 95% CI. All other cells of the 4x4x2
grid are descriptive.

Source file: e44_i1_percell_preview.json.
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
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

RESULTS = Path('/data/papers/low_dimensional_reid/results')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG_SEED = 42
N_QUERY = 500
N_GALLERY = 5000
N_TRAIN_POOL = 3000
FRACS = [0.01, 0.05, 0.10, 0.25]
H, W, C = 16, 8, 2048
N_CELLS = H * W  # 128
N_BOOT = 1000

CKPT_MARKET = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')

TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


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


class DStest(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), int(s.person_id), int(s.camera_id)


class DStrain(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return TRANSFORM(self.samples[i].load_image())


@torch.no_grad()
def extract_maps(backbone, samples, batch=32, mode='test'):
    if mode == 'train':
        ds = DStrain(samples)
    else:
        ds = DStest(samples)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    maps = []; pids = []; cams = []
    print(f"    extracting {len(ds)} items...", flush=True)
    for item in ldr:
        if mode == 'train':
            f = backbone(item.to(device, non_blocking=True))
            maps.append(f.cpu().numpy())
        else:
            imgs, p, c = item
            f = backbone(imgs.to(device, non_blocking=True))
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
    """Returns (ap_per_query, valid_mask) — AP for each query (0 if no match)."""
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


def paired_bootstrap_ci_id_clustered(deltas, valid_mask, pids, n_boot=N_BOOT, seed=0):
    """Paired bootstrap clustered by identity (rev 555 §2): resample identities
    with replacement, take all of each drawn identity's queries. Same-identity
    queries share gallery neighbourhoods and often cameras, so query-level
    resampling under-states correlation and narrows CIs."""
    rng = np.random.default_rng(seed)
    d = deltas[valid_mask]
    p = pids[valid_mask]
    if len(d) == 0:
        return float('nan'), float('nan'), float('nan'), 0
    unique_ids = np.unique(p)
    id_to_idx = {int(pid): np.where(p == pid)[0] for pid in unique_ids}
    n_id = len(unique_ids)
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_ids, size=n_id, replace=True)
        q_idx = np.concatenate([id_to_idx[int(pid)] for pid in sampled])
        boot_means.append(d[q_idx].mean())
    boot_means = np.array(boot_means)
    return (float(d.mean()),
            float(np.quantile(boot_means, 0.025)),
            float(np.quantile(boot_means, 0.975)),
            int(n_id))


def build_cell_distances(maps, cell_means):
    N, C_, H_, W_ = maps.shape
    out = np.zeros((N, H_ * W_), dtype=np.float32)
    for h in range(H_):
        for w in range(W_):
            diff = maps[:, :, h, w] - cell_means[h, w]
            out[:, h * W_ + w] = np.linalg.norm(diff, axis=1)
    return out


def select_top_cells(cell_distances, frac):
    N = cell_distances.shape[0]
    k = max(1, int(round(frac * N_CELLS)))
    return [np.argsort(-cell_distances[i])[:k] for i in range(N)]


def select_random_cells(N, frac, rng):
    k = max(1, int(round(frac * N_CELLS)))
    return [rng.choice(N_CELLS, size=k, replace=False) for _ in range(N)]


def precompute_nearest_idx(test_maps, train_patches_by_cell):
    """One matmul per cell position, GPU-accelerated.
    Returns (N_test, H*W) int array: nearest training-patch index per (image, cell).
    Depends only on test features and training pool, not on f or condition.
    """
    N, C_, H_, W_ = test_maps.shape
    out = np.zeros((N, H_ * W_), dtype=np.int32)
    for h in range(H_):
        for w in range(W_):
            train = train_patches_by_cell[(h, w)]  # (K, C)
            train_t = torch.from_numpy(train).to(device)
            train_norm = train_t / (train_t.norm(dim=1, keepdim=True) + 1e-12)
            test = test_maps[:, :, h, w]  # (N, C)
            test_t = torch.from_numpy(test).to(device)
            test_norm = test_t / (test_t.norm(dim=1, keepdim=True) + 1e-12)
            sims = test_norm @ train_norm.T  # (N, K)
            best = sims.argmax(dim=1).cpu().numpy()  # (N,)
            out[:, h * W_ + w] = best
    return out


def apply_replacement(maps, selected_cells_per_image, train_patches_by_cell,
                      mode='nearest', nearest_idx=None, rng=None):
    """mode ∈ {'nearest', 'random_patch'}.
    nearest: look up pre-computed nearest_idx[image, cell] -> training patch.
    random_patch: uniformly random training patch at (h, w)."""
    N, C_, H_, W_ = maps.shape
    maps_mod = maps.copy()
    for img_idx in range(N):
        for cell_idx in selected_cells_per_image[img_idx]:
            h = cell_idx // W_
            w = cell_idx % W_
            train_patches = train_patches_by_cell[(h, w)]  # (K, C)
            if mode == 'nearest':
                k = int(nearest_idx[img_idx, cell_idx])
            else:  # random_patch
                k = int(rng.integers(0, train_patches.shape[0]))
            maps_mod[img_idx, :, h, w] = train_patches[k]
    return maps_mod


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


def main():
    print(f"device: {device}", flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    print("Loading Market model + BNNeck...", flush=True)
    backbone, bn = load_backbone_and_bn(CKPT_MARKET)

    print("\n### market training pool ###", flush=True)
    train_samples = market.train
    if len(train_samples) > N_TRAIN_POOL:
        idx = rng.choice(len(train_samples), size=N_TRAIN_POOL, replace=False)
        train_samples = [train_samples[int(i)] for i in idx]
    train_maps = extract_maps(backbone, train_samples, mode='train')
    print(f"  train pool shape: {train_maps.shape}", flush=True)
    cell_means = train_maps.mean(axis=0).transpose(1, 2, 0)
    train_patches_by_cell = {(h, w): train_maps[:, :, h, w].copy()
                             for h in range(H) for w in range(W)}

    all_out = {}
    for arm, ds in [('A_prime', market), ('B_prime', msmt)]:
        print(f"\n### arm {arm} ###", flush=True)
        q_maps, q_pids, q_cams, g_maps, g_pids, g_cams = extract_arm(
            backbone, ds, rng, N_QUERY, N_GALLERY)

        # Baseline
        q_emb_b = forward_retrieval(q_maps, bn)
        g_emb_b = forward_retrieval(g_maps, bn)
        aps_b, valid = per_query_ap(q_emb_b, q_pids, q_cams, g_emb_b, g_pids, g_cams)
        base_map = float(aps_b[valid].mean()) if valid.any() else float('nan')
        print(f"  baseline mAP = {100*base_map:.4f}%  (n_valid={int(valid.sum())}/{len(valid)})",
              flush=True)
        arm_out = {'baseline_map': base_map,
                   'n_valid_queries': int(valid.sum()),
                   'conditions': {}}

        q_cd = build_cell_distances(q_maps, cell_means)
        g_cd = build_cell_distances(g_maps, cell_means)

        print("  precomputing nearest-patch indices (once per arm)...", flush=True)
        q_nearest = precompute_nearest_idx(q_maps, train_patches_by_cell)
        g_nearest = precompute_nearest_idx(g_maps, train_patches_by_cell)
        print(f"  nearest-patch cache built: q={q_nearest.shape}, g={g_nearest.shape}",
              flush=True)

        for f in FRACS:
            f_out = {'fraction': f, 'cells_per_image': max(1, int(round(f * N_CELLS)))}

            # Selections
            q_sel_top = select_top_cells(q_cd, f)
            g_sel_top = select_top_cells(g_cd, f)
            q_sel_rand = select_random_cells(q_maps.shape[0], f, rng)
            g_sel_rand = select_random_cells(g_maps.shape[0], f, rng)

            # Four conditions
            conditions = [
                ('top_near', q_sel_top, g_sel_top, 'nearest'),
                ('top_rand', q_sel_top, g_sel_top, 'random_patch'),
                ('rand_near', q_sel_rand, g_sel_rand, 'nearest'),
            ]
            for name, q_sel, g_sel, patch_mode in conditions:
                q_mod = apply_replacement(q_maps, q_sel, train_patches_by_cell,
                                          mode=patch_mode, nearest_idx=q_nearest, rng=rng)
                g_mod = apply_replacement(g_maps, g_sel, train_patches_by_cell,
                                          mode=patch_mode, nearest_idx=g_nearest, rng=rng)
                q_emb = forward_retrieval(q_mod, bn)
                g_emb = forward_retrieval(g_mod, bn)
                aps, valid_c = per_query_ap(q_emb, q_pids, q_cams, g_emb, g_pids, g_cams)
                both_valid = valid & valid_c
                map_cond = float(aps[valid_c].mean()) if valid_c.any() else float('nan')
                deltas = aps - aps_b
                delta_mean, delta_lo, delta_hi, n_id = paired_bootstrap_ci_id_clustered(
                    deltas, both_valid, q_pids,
                    seed=RNG_SEED + abs(hash(name)) % 1000)
                f_out[name] = {
                    'mAP': map_cond,
                    'paired_delta_mean': delta_mean,
                    'paired_delta_ci_lo95_id_clustered': delta_lo,
                    'paired_delta_ci_hi95_id_clustered': delta_hi,
                    'delta_from_baseline_mAP': map_cond - base_map,
                    'n_valid_queries': int(both_valid.sum()),
                    'n_valid_identities': n_id,
                    'per_query_ap': aps.tolist(),
                    'per_query_valid': valid_c.tolist(),
                }
                print(f"  f={f:.2f} {name:>10s}: mAP={100*map_cond:.4f}%  "
                      f"paired_Δ={100*delta_mean:+.3f} pp "
                      f"[{100*delta_lo:+.3f}, {100*delta_hi:+.3f}] (n_id={n_id})",
                      flush=True)
            arm_out['conditions'][f"f{f:.2f}"] = f_out

        # Primary endpoint (rev 555 §1): FIXED at f=0.10, pre-registered
        primary_f = "f0.10"
        arm_out['primary_endpoint'] = {
            'f_prespecified': primary_f,
            'condition': 'top_near',
            'paired_delta_mean': arm_out['conditions'][primary_f]['top_near']['paired_delta_mean'],
            'paired_delta_ci_lo95_id_clustered': arm_out['conditions'][primary_f]['top_near']['paired_delta_ci_lo95_id_clustered'],
            'paired_delta_ci_hi95_id_clustered': arm_out['conditions'][primary_f]['top_near']['paired_delta_ci_hi95_id_clustered'],
        }
        # Argmax (descriptive, labelled selected-on-outcome)
        best_f = max(arm_out['conditions'].keys(),
                     key=lambda fk: arm_out['conditions'][fk]['top_near']['paired_delta_mean'])
        arm_out['selected_on_outcome'] = {
            'f_argmax_top_near': best_f,
            'paired_delta_mean': arm_out['conditions'][best_f]['top_near']['paired_delta_mean'],
            'note': 'selected after seeing outcomes; CI under-covers',
        }
        pe = arm_out['primary_endpoint']
        print(f"  PRIMARY (prespec f=0.10 top_near): paired Δ = "
              f"{100*pe['paired_delta_mean']:+.3f} pp "
              f"[{100*pe['paired_delta_ci_lo95_id_clustered']:+.3f}, "
              f"{100*pe['paired_delta_ci_hi95_id_clustered']:+.3f}]", flush=True)
        print(f"  SELECTED (argmax f = {best_f}): "
              f"paired Δ = {100*arm_out['selected_on_outcome']['paired_delta_mean']:+.3f} pp",
              flush=True)

        all_out[arm] = arm_out

    outp = RESULTS / 'e44_i1_percell_preview.json'
    with open(outp, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
