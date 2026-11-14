"""
Combined I-1 / I-5 experiment job, addressing all four concerns from
rev 535/536:

(1) Radial ablation (rev 536 §1): hold direction fixed, set ||x-mu|| to
    a constant, push through BN, recompute same-cam post-BN cos d'. The
    drop IS the radial contribution to retrieval.

(2) Clamp x BN x cos retrieval cost (rev 536 §5 + rev 535 §5): apply
    the radius sweep from rev 534, then push through BN+L2, measure
    same-cam post-BN cos d' and cross-cam post-BN cos d'. Gives the
    actual retrieval cost of the clamp operation.

(3) Displacement check (rev 535 §3): per arm per radius, report
    ||Delta_x|| / ||x|| on clamped features and induced mean |Delta_cos|
    on evaluated pairs. Distinguishes "clamp is a null operation" from
    "clamp denoised".

(4) Pre-BN norm distribution vs BN running statistics (rev 536 §4):
    per arm, compare pre-BN norm distribution to BN layer's running
    mean mu_BN. Checks the conjecture that B' sits off mu_BN more than
    A' does.

Shared machinery: pre-BN GAP features + BN layer + per-arm pool.
Reference center for clamp: per-model training-set mean (same as
rev 534). Radius set: {p50, p75, p90, p95, p99} of training distances.
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
MAX_IDS = 400
MIN_IMGS = 2
MAX_IMGS_PER_ID = 6
PCTILES = [50, 75, 90, 95, 99]

CKPT_MARKET = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')
CKPT_MSMT   = Path('/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth')

TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ARMS = {
    'A_prime': {'ckpt': 'market', 'test_set': 'market'},
    'B_prime': {'ckpt': 'market', 'test_set': 'msmt'},
    'C':       {'ckpt': 'msmt',   'test_set': 'market'},
    'D':       {'ckpt': 'msmt',   'test_set': 'msmt'},
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
        return x.mean(dim=(2, 3))


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
def extract_train(backbone, samples, batch=64):
    ds = DSimg(samples)
    print(f"      extracting {len(ds)} training items...", flush=True)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    out = []
    for imgs in ldr:
        f = backbone(imgs.to(device, non_blocking=True))
        out.append(f.cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def extract_test(backbone, samples, batch=64):
    ds = DStest(samples)
    print(f"      extracting {len(ds)} test items...", flush=True)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    pre, labels, cams = [], [], []
    for imgs, lbls, cm in ldr:
        f = backbone(imgs.to(device, non_blocking=True))
        pre.append(f.cpu().numpy())
        labels.append(lbls.numpy())
        cams.append(cm.numpy())
    return np.vstack(pre), np.concatenate(labels), np.concatenate(cams)


def bn_forward_np(X_np, bn):
    X = torch.from_numpy(X_np).to(device)
    with torch.no_grad():
        Y = bn(X)
    return Y.cpu().numpy()


def sample_pool(y, cams, rng, max_ids=MAX_IDS, max_per_id=MAX_IMGS_PER_ID):
    id_to_idx = defaultdict(list)
    for i, lbl in enumerate(y):
        id_to_idx[int(lbl)].append(i)
    ids = [k for k, v in id_to_idx.items()
           if len(v) >= MIN_IMGS and len(set(cams[i] for i in v)) >= 2]
    if len(ids) > max_ids:
        ids = list(rng.choice(ids, size=max_ids, replace=False))
    chosen, chosen_ids, chosen_cams = [], [], []
    for k in ids:
        idx = id_to_idx[k]
        if len(idx) > max_per_id:
            idx = list(rng.choice(idx, size=max_per_id, replace=False))
        for i in idx:
            chosen.append(i)
            chosen_ids.append(k)
            chosen_cams.append(cams[i])
    return np.array(chosen), np.array(chosen_ids), np.array(chosen_cams)


def apply_tail_clamp(X, mu, r):
    delta = X - mu
    d = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, r / (d + 1e-12))
    return mu + delta * scale, np.linalg.norm(delta, axis=1), (d.squeeze(-1) > r)


def ablate_radial(X, mu, target_norm):
    """Hold direction (x - mu)/||x-mu|| fixed, set ||x-mu|| to target_norm."""
    delta = X - mu
    d = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = delta / (d + 1e-12)
    return mu + unit * target_norm


def pair_stats(X, ids, cams):
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1)
    dots = X @ X.T
    cos = dots / (norms[:, None] * norms[None, :] + 1e-12)
    d2 = (norms[:, None] ** 2) + (norms[None, :] ** 2) - 2 * dots
    d2 = np.clip(d2, 0, None)
    eucl = np.sqrt(d2)
    same_id = ids[:, None] == ids[None, :]
    same_cam = cams[:, None] == cams[None, :]
    iu = np.triu_indices(n, k=1)
    return cos[iu], eucl[iu], same_id[iu], same_cam[iu]


def dprime(same_vals, diff_vals):
    if len(same_vals) < 2 or len(diff_vals) < 2:
        return float('nan')
    ms, md = float(same_vals.mean()), float(diff_vals.mean())
    vs, vd = float(same_vals.var(ddof=1)), float(diff_vals.var(ddof=1))
    denom = (0.5 * (vs + vd)) ** 0.5
    if denom < 1e-12:
        return float('nan')
    return (md - ms) / denom


def pair_auc(same_vals, diff_vals, max_pairs=200000, seed=0):
    rng = np.random.default_rng(seed)
    n_same = len(same_vals)
    n_diff = len(diff_vals)
    if n_same == 0 or n_diff == 0:
        return float('nan')
    m = min(max_pairs, n_same * n_diff)
    si = rng.integers(0, n_same, size=m)
    di = rng.integers(0, n_diff, size=m)
    wins = (same_vals[si] < diff_vals[di]).sum()
    ties = (same_vals[si] == diff_vals[di]).sum()
    return float((wins + 0.5 * ties) / m)


def retrieval_metrics(X_post, pool_ids, pool_cams):
    """Compute post-BN cos d'/AUC for same-cam and cross-cam subsets
    at top-decile post-BN cos threshold."""
    cos, _, same_id, same_cam = pair_stats(X_post, pool_ids, pool_cams)
    cos_dist = 1.0 - cos
    thr = float(np.quantile(cos, 0.90))
    thr_mask = cos >= thr
    out = {'threshold_post_bn_cos': thr}
    for name, subset in [('same_cam', same_cam), ('cross_cam', ~same_cam)]:
        mask = subset & thr_mask
        d = cos_dist[mask]
        s = same_id[mask]
        sv = d[s]
        dv = d[~s]
        out[name] = {
            'n_pairs': int(mask.sum()),
            'n_same': int(s.sum()),
            'dprime': dprime(sv, dv),
            'auc': pair_auc(sv, dv),
        }
    return out


def main():
    print(f"device: {device}", flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    print("Loading backbones + BNNeck...", flush=True)
    m_mkt_bb, m_mkt_bn = load_backbone_and_bn(CKPT_MARKET)
    m_msmt_bb, m_msmt_bn = load_backbone_and_bn(CKPT_MSMT)
    models_map = {'market': (m_mkt_bb, m_mkt_bn), 'msmt': (m_msmt_bb, m_msmt_bn)}

    # Training reference per model
    ref = {}
    for key, (bb, bn), ds in [('market', (m_mkt_bb, m_mkt_bn), market),
                              ('msmt', (m_msmt_bb, m_msmt_bn), msmt)]:
        print(f"\n### training ref for {key} ###", flush=True)
        X_tr = extract_train(bb, ds.train)
        mu = X_tr.mean(axis=0)
        d_tr = np.linalg.norm(X_tr - mu, axis=1)
        qvals = {f"p{p}": float(np.quantile(d_tr, p / 100.0)) for p in PCTILES}
        median_norm_delta = float(np.median(d_tr))
        # Per-dim BN running stats
        mu_bn = bn.running_mean.detach().cpu().numpy()
        sigma2_bn = bn.running_var.detach().cpu().numpy()
        ref[key] = {
            'mu': mu, 'qvals': qvals, 'median_dist': median_norm_delta,
            'mu_bn': mu_bn, 'sigma2_bn': sigma2_bn,
            'n_train': len(X_tr),
            'mean_norm_train': float(np.linalg.norm(X_tr, axis=1).mean()),
            'mean_delta_norm_train': float(np.linalg.norm(X_tr - mu, axis=1).mean()),
        }
        print(f"  n_train={len(X_tr)}  median ||x-mu||={median_norm_delta:.3f}  "
              f"qvals={qvals}", flush=True)
        del X_tr

    # Per-arm experiments
    all_out = {}
    for arm, info in ARMS.items():
        print(f"\n### arm {arm} ###", flush=True)
        backbone, bn = models_map[info['ckpt']]
        test_ds = market if info['test_set'] == 'market' else msmt
        n_q = sum(1 for s in test_ds.query if not s.is_junk)
        gallery_samples = [s for s in (test_ds.query + test_ds.gallery)[n_q:]
                           if not s.is_junk]
        X_pre, y, cams = extract_test(backbone, gallery_samples)
        chosen, pool_ids, pool_cams = sample_pool(y, cams, rng)
        X_pool_pre = X_pre[chosen]
        print(f"  pool: {len(X_pool_pre)} feats, {len(set(pool_ids.tolist()))} multi-cam ids",
              flush=True)

        mu = ref[info['ckpt']]['mu']
        qvals = ref[info['ckpt']]['qvals']
        median_delta = ref[info['ckpt']]['median_dist']
        mu_bn = ref[info['ckpt']]['mu_bn']
        sigma2_bn = ref[info['ckpt']]['sigma2_bn']

        arm_out = {
            'ref_model': info['ckpt'],
            'quantiles_training_dist': qvals,
            'pool_size': int(len(X_pool_pre)),
        }

        # (4) Pre-BN norm vs BN running stats
        pre_norms = np.linalg.norm(X_pool_pre, axis=1)
        delta_norms = np.linalg.norm(X_pool_pre - mu, axis=1)
        # Standardised per-dim: (x - mu_BN) / sqrt(sigma2_BN + eps)
        std = (X_pool_pre - mu_bn) / np.sqrt(sigma2_bn + 1e-5)
        std_norm = np.linalg.norm(std, axis=1)
        arm_out['norm_stats'] = {
            'pre_bn_norm_median': float(np.median(pre_norms)),
            'pre_bn_norm_mean': float(pre_norms.mean()),
            'delta_from_mu_median': float(np.median(delta_norms)),
            'standardised_norm_median': float(np.median(std_norm)),
            'standardised_norm_mean': float(std_norm.mean()),
            'mu_bn_norm': float(np.linalg.norm(mu_bn)),
            'ref_training_mean_norm': float(ref[info['ckpt']]['mean_norm_train']),
            'ref_training_mean_delta_norm': float(ref[info['ckpt']]['mean_delta_norm_train']),
        }
        print(f"  norm stats: pre-BN median={np.median(pre_norms):.3f}  "
              f"delta-from-mu median={np.median(delta_norms):.3f}  "
              f"standardised median={np.median(std_norm):.3f}", flush=True)

        # Baseline: post-BN cos d' at top-decile (no clamp, no ablation)
        X_pool_post = bn_forward_np(X_pool_pre, bn)
        baseline = retrieval_metrics(X_pool_post, pool_ids, pool_cams)
        arm_out['baseline_post_bn'] = baseline
        print(f"  baseline post-BN cos: same-cam d'={baseline['same_cam']['dprime']:.3f} "
              f"AUC={baseline['same_cam']['auc']:.3f}  "
              f"cross-cam d'={baseline['cross_cam']['dprime']:.3f} "
              f"AUC={baseline['cross_cam']['auc']:.3f}", flush=True)

        # (1) Radial ablation: set ||x-mu|| = median training delta, keep direction
        X_ablated_pre = ablate_radial(X_pool_pre, mu, median_delta)
        X_ablated_post = bn_forward_np(X_ablated_pre, bn)
        ablated = retrieval_metrics(X_ablated_post, pool_ids, pool_cams)
        arm_out['radial_ablated_post_bn'] = ablated
        arm_out['radial_ablation_target_norm'] = float(median_delta)
        arm_out['radial_ablation_delta'] = {
            'same_cam_dprime': ablated['same_cam']['dprime'] - baseline['same_cam']['dprime'],
            'same_cam_auc': ablated['same_cam']['auc'] - baseline['same_cam']['auc'],
            'cross_cam_dprime': ablated['cross_cam']['dprime'] - baseline['cross_cam']['dprime'],
            'cross_cam_auc': ablated['cross_cam']['auc'] - baseline['cross_cam']['auc'],
        }
        a = arm_out['radial_ablation_delta']
        print(f"  RADIAL ABLATION (set ||x-mu||={median_delta:.3f}): "
              f"same-cam Δd'={a['same_cam_dprime']:+.3f} ΔAUC={a['same_cam_auc']:+.3f}  "
              f"cross-cam Δd'={a['cross_cam_dprime']:+.3f} ΔAUC={a['cross_cam_auc']:+.3f}",
              flush=True)

        # (2,3) Clamp × BN × cos + displacement
        arm_out['clamp_sweep'] = {}
        for p in PCTILES:
            r = qvals[f"p{p}"]
            X_clamped_pre, delta_norms_orig, outside = apply_tail_clamp(X_pool_pre, mu, r)
            frac_clamped = float(outside.mean())
            # Displacement on clamped-only features
            if outside.any():
                disp_rel = np.linalg.norm(X_clamped_pre[outside] - X_pool_pre[outside], axis=1) / \
                           (np.linalg.norm(X_pool_pre[outside], axis=1) + 1e-12)
                disp_rel_median = float(np.median(disp_rel))
                disp_rel_mean = float(disp_rel.mean())
            else:
                disp_rel_median = 0.0
                disp_rel_mean = 0.0
            # Induced Δcos on evaluated pairs (full pre-BN cos change)
            cos_orig = X_pool_pre @ X_pool_pre.T / \
                       (np.linalg.norm(X_pool_pre, axis=1)[:, None] *
                        np.linalg.norm(X_pool_pre, axis=1)[None, :] + 1e-12)
            cos_new = X_clamped_pre @ X_clamped_pre.T / \
                      (np.linalg.norm(X_clamped_pre, axis=1)[:, None] *
                       np.linalg.norm(X_clamped_pre, axis=1)[None, :] + 1e-12)
            iu = np.triu_indices(X_pool_pre.shape[0], k=1)
            d_cos_abs = np.abs(cos_new[iu] - cos_orig[iu])
            mean_abs_dcos = float(d_cos_abs.mean())
            median_abs_dcos = float(np.median(d_cos_abs))

            X_clamped_post = bn_forward_np(X_clamped_pre, bn)
            res = retrieval_metrics(X_clamped_post, pool_ids, pool_cams)
            res['radius_r'] = r
            res['fraction_clamped'] = frac_clamped
            res['displacement_rel_median'] = disp_rel_median
            res['displacement_rel_mean'] = disp_rel_mean
            res['induced_abs_dcos_mean'] = mean_abs_dcos
            res['induced_abs_dcos_median'] = median_abs_dcos
            res['same_cam_dprime_delta'] = res['same_cam']['dprime'] - baseline['same_cam']['dprime']
            res['same_cam_auc_delta'] = res['same_cam']['auc'] - baseline['same_cam']['auc']
            res['cross_cam_dprime_delta'] = res['cross_cam']['dprime'] - baseline['cross_cam']['dprime']
            res['cross_cam_auc_delta'] = res['cross_cam']['auc'] - baseline['cross_cam']['auc']
            arm_out['clamp_sweep'][f"p{p}"] = res
            print(f"  r=p{p:>2d}={r:.3f}  frac={frac_clamped:.3f}  "
                  f"disp_rel_med={disp_rel_median:.3f}  |Δcos|_mean={mean_abs_dcos:.4f}  "
                  f"sc Δd'={res['same_cam_dprime_delta']:+.3f} "
                  f"ΔAUC={res['same_cam_auc_delta']:+.3f}  "
                  f"cc Δd'={res['cross_cam_dprime_delta']:+.3f} "
                  f"ΔAUC={res['cross_cam_auc_delta']:+.3f}", flush=True)

        all_out[arm] = arm_out

    all_out['_reference'] = {
        'market': {'quantiles': ref['market']['qvals'],
                   'n_train': ref['market']['n_train'],
                   'median_dist': ref['market']['median_dist'],
                   'mean_norm_train': ref['market']['mean_norm_train']},
        'msmt': {'quantiles': ref['msmt']['qvals'],
                 'n_train': ref['msmt']['n_train'],
                 'median_dist': ref['msmt']['median_dist'],
                 'mean_norm_train': ref['msmt']['mean_norm_train']},
    }

    outp = RESULTS / 'e44_combined_i1_i5.json'
    with open(outp, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
