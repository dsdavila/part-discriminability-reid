"""
Post-hoc v3 for I-1 per-cell preview (rev 567 asks).

Two things:
  (1) Spearman rank correlation between ||x - mu_train|| and ||x||
      across cells, per arm, on the saved subsample. If ρ ≈ 1 then
      "top-residual" is "high-norm" wearing a label, and the NAE
      framework (Chen et al. CVPR 2020) explains the result as the
      norm being a foregroundness proxy rather than identity info.
  (2) Relative-to-baseline interaction with baselines resampled
      inside each identity-clustered bootstrap draw. Rev 566's
      delta-method held baselines fixed; this exposes the real
      width.
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
from scipy.stats import spearmanr
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

ARMS = {'A_prime': 'market', 'B_prime': 'msmt'}


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
    sd = {}; bn_sd = {}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k: continue
        if k.startswith('3.'):
            bn_sd[k[2:]] = v; continue
        for prefix, target in MAP.items():
            if k.startswith(prefix):
                sd[target + k[len(prefix):]] = v; break
    backbone.load_state_dict(sd, strict=False)
    bn = nn.BatchNorm1d(2048, affine=True, track_running_stats=True).to(device).eval()
    bn.weight.data = bn_sd['weight'].to(device)
    bn.bias.data = bn_sd['bias'].to(device)
    bn.running_mean.data = bn_sd['running_mean'].to(device)
    bn.running_var.data = bn_sd['running_var'].to(device)
    return backbone, bn


class DSimg(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return TRANSFORM(self.samples[i].load_image())


class DStest(Dataset):
    def __init__(self, samples): self.samples = [s for s in samples if not s.is_junk]
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
            f = backbone(item.to(device)); maps.append(f.cpu().numpy())
        else:
            imgs, p, c = item
            f = backbone(imgs.to(device))
            maps.append(f.cpu().numpy()); pids.append(p.numpy()); cams.append(c.numpy())
    maps = np.concatenate(maps, axis=0)
    if mode == 'train': return maps
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
    aps = np.zeros(len(q_pids)); valid = np.zeros(len(q_pids), dtype=bool)
    for i in range(len(q_pids)):
        qid, qcam = q_pids[i], q_cams[i]
        junk = (g_pids == qid) & (g_cams == qcam)
        order = np.argsort(-sims[i]); order = order[~junk[order]]
        good = g_pids[order] == qid
        if not good.any(): continue
        prec = np.cumsum(good) / np.arange(1, len(good) + 1)
        aps[i] = (prec * good).sum() / good.sum(); valid[i] = True
    return aps, valid


def extract_arm(backbone, ds, rng, n_query, n_gallery):
    non_junk_q = [s for s in ds.query if not s.is_junk]
    non_junk_g = [s for s in ds.gallery if not s.is_junk]
    q_idx = rng.choice(len(non_junk_q), size=min(n_query, len(non_junk_q)), replace=False)
    g_idx = rng.choice(len(non_junk_g), size=min(n_gallery, len(non_junk_g)), replace=False)
    q = [non_junk_q[int(i)] for i in q_idx]
    g = [non_junk_g[int(i)] for i in g_idx]
    qm, qp, qc = extract_maps(backbone, q, mode='test')
    gm, gp, gc = extract_maps(backbone, g, mode='test')
    return qm, qp, qc, gm, gp, gc


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
    cell_means = train_maps.mean(axis=0).transpose(1, 2, 0)  # (H, W, C)

    out = {'post_hoc_v3_rev': 567}

    per_arm = {}
    for arm, ckpt in ARMS.items():
        print(f"\n### arm {arm} ###", flush=True)
        ds = market if ckpt == 'market' else msmt
        q_maps, q_pids, q_cams, g_maps, g_pids, g_cams = extract_arm(
            backbone, ds, rng, N_QUERY, N_GALLERY)

        # (1) Spearman ρ between ||x - mu_train|| and ||x|| across cells
        q_dist = np.zeros((q_maps.shape[0], N_CELLS))
        q_norm = np.zeros((q_maps.shape[0], N_CELLS))
        for h in range(H):
            for w in range(W):
                diff = q_maps[:, :, h, w] - cell_means[h, w]
                q_dist[:, h * W + w] = np.linalg.norm(diff, axis=1)
                q_norm[:, h * W + w] = np.linalg.norm(q_maps[:, :, h, w], axis=1)
        # Flatten all cells across all images
        rho_all, _ = spearmanr(q_dist.ravel(), q_norm.ravel())
        # Also per image (mean of per-image spearman)
        per_img_rho = []
        for i in range(q_maps.shape[0]):
            r, _ = spearmanr(q_dist[i], q_norm[i])
            if not np.isnan(r):
                per_img_rho.append(r)
        print(f"  Spearman ρ (all-cells, flattened): {rho_all:.4f}", flush=True)
        print(f"  Spearman ρ (per-image, mean of {len(per_img_rho)} images): "
              f"{np.mean(per_img_rho):.4f}  median={np.median(per_img_rho):.4f}", flush=True)

        # Baseline per-query APs
        q_emb_b = forward_retrieval(q_maps, bn)
        g_emb_b = forward_retrieval(g_maps, bn)
        aps_b, valid_b = per_query_ap(q_emb_b, q_pids, q_cams, g_emb_b, g_pids, g_cams)

        # Top-residual + nearest at f=0.10
        from compute_i1_percell_preview import (precompute_nearest_idx, build_cell_distances,
                                                apply_replacement)
        q_cd = build_cell_distances(q_maps, cell_means)
        g_cd = build_cell_distances(g_maps, cell_means)
        train_patches_by_cell = {(h, w): train_maps[:, :, h, w].copy()
                                 for h in range(H) for w in range(W)}
        q_near = precompute_nearest_idx(q_maps, train_patches_by_cell)
        g_near = precompute_nearest_idx(g_maps, train_patches_by_cell)
        k = max(1, int(round(FRAC * N_CELLS)))
        q_sel = [np.argsort(-q_cd[i])[:k] for i in range(q_maps.shape[0])]
        g_sel = [np.argsort(-g_cd[i])[:k] for i in range(g_maps.shape[0])]
        q_mod = apply_replacement(q_maps, q_sel, train_patches_by_cell, mode='nearest',
                                  nearest_idx=q_near)
        g_mod = apply_replacement(g_maps, g_sel, train_patches_by_cell, mode='nearest',
                                  nearest_idx=g_near)
        q_emb_c = forward_retrieval(q_mod, bn)
        g_emb_c = forward_retrieval(g_mod, bn)
        aps_c, valid_c = per_query_ap(q_emb_c, q_pids, q_cams, g_emb_c, g_pids, g_cams)
        both = valid_b & valid_c

        # (2) Relative-interaction bootstrap: resample identities WITH baselines inside
        aps_b_v = aps_b[both]; aps_c_v = aps_c[both]; pids_v = q_pids[both]
        unique_ids = np.unique(pids_v)
        id_to_idx = {int(pid): np.where(pids_v == pid)[0] for pid in unique_ids}
        n_id = len(unique_ids)
        rng_boot = np.random.default_rng(RNG_SEED + 100)
        boot_rel = []  # (cond_map / base_map - 1) per arm
        boot_pp = []
        boot_base = []
        for _ in range(N_BOOT):
            s = rng_boot.choice(unique_ids, size=n_id, replace=True)
            qi = np.concatenate([id_to_idx[int(pid)] for pid in s])
            b_map = aps_b_v[qi].mean()
            c_map = aps_c_v[qi].mean()
            boot_base.append(b_map)
            boot_pp.append(c_map - b_map)
            boot_rel.append((c_map - b_map) / b_map if b_map > 1e-12 else np.nan)

        per_arm[arm] = {
            'spearman_dist_vs_norm_all_cells': float(rho_all),
            'spearman_dist_vs_norm_per_image_mean': float(np.mean(per_img_rho)),
            'spearman_dist_vs_norm_per_image_median': float(np.median(per_img_rho)),
            'baseline_map': float(aps_b_v.mean()),
            'condition_map': float(aps_c_v.mean()),
            'paired_delta_pp_mean': float(aps_c_v.mean() - aps_b_v.mean()),
            'paired_delta_rel_mean': float((aps_c_v.mean() - aps_b_v.mean()) / aps_b_v.mean()),
            'boot_pp_mean': float(np.mean(boot_pp)),
            'boot_pp_ci95_lo': float(np.quantile(boot_pp, 0.025)),
            'boot_pp_ci95_hi': float(np.quantile(boot_pp, 0.975)),
            'boot_rel_mean': float(np.nanmean(boot_rel)),
            'boot_rel_ci95_lo': float(np.nanquantile(boot_rel, 0.025)),
            'boot_rel_ci95_hi': float(np.nanquantile(boot_rel, 0.975)),
            'n_identities': n_id,
            'n_queries_valid': int(both.sum()),
        }
        p = per_arm[arm]
        print(f"  baseline={100*p['baseline_map']:.3f}%  cond={100*p['condition_map']:.3f}%  "
              f"n_q={p['n_queries_valid']}  n_id={n_id}", flush=True)
        print(f"  pp Δ bootstrap: mean={100*p['boot_pp_mean']:+.3f} pp  "
              f"CI95 [{100*p['boot_pp_ci95_lo']:+.3f}, {100*p['boot_pp_ci95_hi']:+.3f}]", flush=True)
        print(f"  rel Δ bootstrap (baselines resampled): "
              f"mean={100*p['boot_rel_mean']:+.3f}%  "
              f"CI95 [{100*p['boot_rel_ci95_lo']:+.3f}%, {100*p['boot_rel_ci95_hi']:+.3f}%]",
              flush=True)

        # Save raw per-bootstrap arrays for cross-arm joint
        per_arm[arm]['_boot_pp'] = boot_pp
        per_arm[arm]['_boot_rel'] = boot_rel

    # Cross-arm interaction via paired-across-arms bootstraps (independent)
    a_pp = np.array(per_arm['A_prime']['_boot_pp'])
    b_pp = np.array(per_arm['B_prime']['_boot_pp'])
    a_rel = np.array(per_arm['A_prime']['_boot_rel'])
    b_rel = np.array(per_arm['B_prime']['_boot_rel'])
    # Independent bootstraps -> pair them randomly for the diff
    diff_pp = b_pp - a_pp
    diff_rel = b_rel - a_rel
    print(f"\n### INTERACTION (independent bootstraps, baselines resampled) ###", flush=True)
    print(f"  pp: mean={100*np.mean(diff_pp):+.3f} pp  "
          f"CI95 [{100*np.quantile(diff_pp, 0.025):+.3f}, "
          f"{100*np.quantile(diff_pp, 0.975):+.3f}]", flush=True)
    rel_mask = ~np.isnan(diff_rel)
    print(f"  rel: mean={100*np.nanmean(diff_rel):+.3f}%  "
          f"CI95 [{100*np.nanquantile(diff_rel, 0.025):+.3f}%, "
          f"{100*np.nanquantile(diff_rel, 0.975):+.3f}%]", flush=True)

    out['per_arm'] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                      for k, v in per_arm.items()}
    out['interaction_resampled_baselines'] = {
        'pp_mean': float(np.mean(diff_pp)),
        'pp_ci95_lo': float(np.quantile(diff_pp, 0.025)),
        'pp_ci95_hi': float(np.quantile(diff_pp, 0.975)),
        'rel_mean': float(np.nanmean(diff_rel)),
        'rel_ci95_lo': float(np.nanquantile(diff_rel, 0.025)),
        'rel_ci95_hi': float(np.nanquantile(diff_rel, 0.975)),
        'note': 'independent per-arm identity-clustered bootstraps; baselines resampled inside each draw',
    }

    outp = RESULTS / 'e44_i1_percell_posthoc_v3.json'
    with open(outp, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
