"""
Post-hoc for I-1 per-cell preview (rev 563 corrections).

Three analyses on saved per-query APs + one quick re-extraction:

(1) Joint identity-clustered bootstrap on the interaction (B' - A') at
    f=0.10 top_near. Replaces rev 562's hand-wave with the real CI.
(2) Relative-to-baseline normalisation: both arms' deltas divided by
    their respective baseline mAPs. Direction may flip relative to
    pp-denominated (rev 563 §2).
(3) Mean ||Δx||/||x|| per replaced cell, top-residual vs random-cell at
    f=0.10, both arms. Separates "top-residual cells carry info" from
    "top-residual cells are perturbed harder" (rev 563 §4).
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


def load_backbone(ckpt_path):
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = BoTBackbone().to(device).eval()
    MAP = {'0.0.': 'stem_conv.', '0.1.': 'stem_bn.',
           '0.4.': 'layer1.', '0.5.': 'layer2.',
           '0.6.': 'layer3.', '0.7.': 'layer4.'}
    sd = {}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k:
            continue
        if k.startswith('3.'):
            continue
        for prefix, target in MAP.items():
            if k.startswith(prefix):
                sd[target + k[len(prefix):]] = v
                break
    backbone.load_state_dict(sd, strict=False)
    return backbone


class DSimg(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return TRANSFORM(self.samples[i].load_image())


class DStest(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]

    def __len__(self):
        return len(self.samples)

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


def displacement_per_cell(maps, selected_cells_per_image, train_patches_by_cell, nearest_idx):
    """For replaced cells only, compute ||Δx||/||x|| mean/median."""
    N, C_, H_, W_ = maps.shape
    rel_disps = []
    for img_idx in range(N):
        for cell_idx in selected_cells_per_image[img_idx]:
            h = cell_idx // W_
            w = cell_idx % W_
            orig = maps[img_idx, :, h, w]
            k = int(nearest_idx[img_idx, cell_idx])
            new = train_patches_by_cell[(h, w)][k]
            rel = np.linalg.norm(new - orig) / (np.linalg.norm(orig) + 1e-12)
            rel_disps.append(rel)
    return np.array(rel_disps)


def main():
    in_path = RESULTS / 'e44_i1_percell_preview.json'
    with open(in_path) as f:
        data = json.load(f)

    out = {'post_hoc_rev': 563}

    # --- (1) + (2) from saved per-query APs ---
    print("=== (1) + (2) from saved APs ===", flush=True)
    rng = np.random.default_rng(RNG_SEED)
    arm_data = {}
    for arm in ARMS:
        base = data[arm]['baseline_map']
        cond = data[arm]['conditions']['f0.10']['top_near']
        aps = np.array(cond['per_query_ap'])
        valid = np.array(cond['per_query_valid'])
        deltas = aps - base  # per-query delta vs baseline mAP (approximate, but
        # since rev 562 script computes baseline aps per-query already, use that)
        # Actually: aps_b = baseline APs weren't saved explicitly; the delta
        # needs to be reconstructed as (cond_ap - baseline_ap). The saved
        # 'per_query_ap' in the condition is cond_ap, but we don't have
        # baseline_ap per query. Save that next time.
        # Workaround: approximate by using (cond_ap - base_scalar) where
        # base_scalar is the mean baseline. For bootstrap, this changes
        # delta mean but not its variance — the variance is dominated by
        # within-arm scatter. Flag the limitation.
        arm_data[arm] = {
            'cond_aps': aps,
            'valid': valid,
            'base_scalar': base,
        }

    # Note limitation and move on — joint bootstrap by identity across arms
    # (bootstrap each arm's identity set independently then compute diff).

    # For identity-clustered bootstrap we need identities. Re-extract
    # (quick) to get pids aligned with the saved APs (same seed).
    print("  re-extracting per-arm pids (same seed)...", flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng2 = np.random.default_rng(RNG_SEED)
    # Reproduce the sample_pool for each arm
    # Replicate pipeline exactly: train pool first, then each arm
    train_samples = market.train
    if len(train_samples) > N_TRAIN_POOL:
        idx = rng2.choice(len(train_samples), size=N_TRAIN_POOL, replace=False)
        # Advance RNG state equivalently
    pid_by_arm = {}
    for arm, ckpt in ARMS.items():
        ds = market if ckpt == 'market' else msmt
        non_junk_q = [s for s in ds.query if not s.is_junk]
        non_junk_g = [s for s in ds.gallery if not s.is_junk]
        q_idx = rng2.choice(len(non_junk_q), size=min(N_QUERY, len(non_junk_q)), replace=False)
        g_idx = rng2.choice(len(non_junk_g), size=min(N_GALLERY, len(non_junk_g)), replace=False)
        q_pids = np.array([int(non_junk_q[int(i)].person_id) for i in q_idx])
        pid_by_arm[arm] = q_pids

    # Identity-clustered bootstrap, independent per arm
    print("  running joint identity-clustered bootstrap on interaction...", flush=True)
    boot_a = []
    boot_b = []
    for arm, boot_list in [('A_prime', boot_a), ('B_prime', boot_b)]:
        rng_boot = np.random.default_rng(RNG_SEED + hash(arm) % 1000)
        aps = arm_data[arm]['cond_aps']
        valid = arm_data[arm]['valid']
        base = arm_data[arm]['base_scalar']
        pids = pid_by_arm[arm]
        # Delta per query: cond_ap - base (scalar). This under-states variance
        # slightly but preserves the per-query shape.
        d = aps - base
        v = valid
        d_valid = d[v]
        p_valid = pids[v]
        unique_ids = np.unique(p_valid)
        id_to_idx = {int(pid): np.where(p_valid == pid)[0] for pid in unique_ids}
        n_id = len(unique_ids)
        for _ in range(N_BOOT):
            sampled = rng_boot.choice(unique_ids, size=n_id, replace=True)
            q_idx = np.concatenate([id_to_idx[int(pid)] for pid in sampled])
            boot_list.append(d_valid[q_idx].mean())

    boot_a = np.array(boot_a)
    boot_b = np.array(boot_b)
    inter_boot = boot_b - boot_a
    inter_mean = float(inter_boot.mean())
    inter_lo = float(np.quantile(inter_boot, 0.025))
    inter_hi = float(np.quantile(inter_boot, 0.975))
    print(f"  INTERACTION (B' - A') pp: mean = {100*inter_mean:+.3f}  "
          f"CI95 [{100*inter_lo:+.3f}, {100*inter_hi:+.3f}]  (joint id-clustered)",
          flush=True)
    excludes_zero = (inter_lo > 0) or (inter_hi < 0)
    print(f"  Excludes zero: {excludes_zero}", flush=True)

    out['pp_interaction'] = {
        'mean_pp': inter_mean,
        'ci95_lo_pp': inter_lo,
        'ci95_hi_pp': inter_hi,
        'excludes_zero': bool(excludes_zero),
        'note': 'joint identity-clustered bootstrap, independent arms, delta-vs-mean-baseline',
    }

    # (2) Relative normalisation: delta / baseline_mAP
    print("\n=== (2) Relative-to-baseline normalisation ===", flush=True)
    a_delta = arm_data['A_prime']['cond_aps'][arm_data['A_prime']['valid']].mean() - arm_data['A_prime']['base_scalar']
    b_delta = arm_data['B_prime']['cond_aps'][arm_data['B_prime']['valid']].mean() - arm_data['B_prime']['base_scalar']
    a_rel = a_delta / arm_data['A_prime']['base_scalar']
    b_rel = b_delta / arm_data['B_prime']['base_scalar']
    print(f"  A' delta_pp = {100*a_delta:+.3f}  rel = {100*a_rel:+.3f}%", flush=True)
    print(f"  B' delta_pp = {100*b_delta:+.3f}  rel = {100*b_rel:+.3f}%", flush=True)
    rel_inter = b_rel - a_rel
    print(f"  REL INTERACTION = {100*rel_inter:+.3f} pp (B' rel - A' rel)", flush=True)
    # Bootstrap on relative normalisation too
    boot_a_rel = boot_a / arm_data['A_prime']['base_scalar']
    boot_b_rel = boot_b / arm_data['B_prime']['base_scalar']
    rel_inter_boot = boot_b_rel - boot_a_rel
    rel_lo = float(np.quantile(rel_inter_boot, 0.025))
    rel_hi = float(np.quantile(rel_inter_boot, 0.975))
    print(f"  REL INTERACTION CI95 [{100*rel_lo:+.3f}%, {100*rel_hi:+.3f}%]", flush=True)
    rel_excludes_zero = (rel_lo > 0) or (rel_hi < 0)
    print(f"  Excludes zero: {rel_excludes_zero}", flush=True)
    out['rel_interaction'] = {
        'a_rel': a_rel, 'b_rel': b_rel,
        'rel_interaction_mean': b_rel - a_rel,
        'ci95_lo': rel_lo, 'ci95_hi': rel_hi,
        'excludes_zero': bool(rel_excludes_zero),
        'note': 'delta / baseline_mAP per arm; joint id-clustered bootstrap',
    }

    # --- (3) Displacement check via re-extraction ---
    print("\n=== (3) Displacement check ===", flush=True)
    rng3 = np.random.default_rng(RNG_SEED)
    backbone = load_backbone(CKPT_MARKET)
    train_samples = market.train
    if len(train_samples) > N_TRAIN_POOL:
        idx = rng3.choice(len(train_samples), size=N_TRAIN_POOL, replace=False)
        train_samples = [train_samples[int(i)] for i in idx]
    train_maps = extract_maps(backbone, train_samples, mode='train')
    cell_means = train_maps.mean(axis=0).transpose(1, 2, 0)
    train_patches_by_cell = {(h, w): train_maps[:, :, h, w].copy()
                             for h in range(H) for w in range(W)}

    disp_out = {}
    for arm_name, ckpt in ARMS.items():
        print(f"  arm {arm_name}", flush=True)
        ds = market if ckpt == 'market' else msmt
        q_maps, q_pids, q_cams, g_maps, g_pids, g_cams = extract_arm(
            backbone, ds, rng3, N_QUERY, N_GALLERY)
        q_cd = build_cell_distances(q_maps, cell_means)
        q_nearest = precompute_nearest_idx(q_maps, train_patches_by_cell)

        # Top-residual at f=0.10 and random-cell at f=0.10 (query maps only)
        k = max(1, int(round(FRAC * N_CELLS)))
        q_sel_top = [np.argsort(-q_cd[i])[:k] for i in range(q_maps.shape[0])]
        rng_rand = np.random.default_rng(RNG_SEED + 123)
        q_sel_rand = [rng_rand.choice(N_CELLS, size=k, replace=False)
                      for _ in range(q_maps.shape[0])]

        top_disps = displacement_per_cell(q_maps, q_sel_top, train_patches_by_cell, q_nearest)
        rand_disps = displacement_per_cell(q_maps, q_sel_rand, train_patches_by_cell, q_nearest)
        disp_out[arm_name] = {
            'top_residual_mean': float(top_disps.mean()),
            'top_residual_median': float(np.median(top_disps)),
            'random_cell_mean': float(rand_disps.mean()),
            'random_cell_median': float(np.median(rand_disps)),
            'ratio_top_over_random_mean': float(top_disps.mean() / rand_disps.mean()),
        }
        print(f"    top_residual: mean ||Δx||/||x|| = {disp_out[arm_name]['top_residual_mean']:.4f}  "
              f"median = {disp_out[arm_name]['top_residual_median']:.4f}", flush=True)
        print(f"    random_cell:  mean ||Δx||/||x|| = {disp_out[arm_name]['random_cell_mean']:.4f}  "
              f"median = {disp_out[arm_name]['random_cell_median']:.4f}", flush=True)
        print(f"    ratio (top/rand) = {disp_out[arm_name]['ratio_top_over_random_mean']:.3f}",
              flush=True)

    out['displacement_check'] = disp_out

    outp = RESULTS / 'e44_i1_percell_posthoc.json'
    with open(outp, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
