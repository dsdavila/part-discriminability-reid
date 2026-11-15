"""
E-snap-A, variant (a): snap-to-training-manifold at a single layer.

Spec: Channel 4 rev 626 (claude-chat, relaying Daniel).

For each layer L in {1, 2, 3, 4}:
  For each OOD query Q (and symmetrically each gallery G):
      g_L(Q) = GAP( layer_L_map(Q) )                    # one vector, dim C_L
      T* = argmax_T  cos( g_L(Q), g_L(T) )              # nearest training sample at L
      emb(Q) := terminal_embedding( T* )                # BN(GAP(layer4))·L2
  Run standard retrieval on the snapped (query, gallery) pair -> mAP, Rank-1.

Arms:
  - B'-sub (BoT-Market, MSMT query/gallery, 500Q × 5000G, saved indices)
  - full-set B' (contingent on GPU envelope — not kicked off here)
  - C       (BoT-MSMT,  Market query/gallery)

Writes /data/papers/low_dimensional_reid/results/e44_e_snap_a_<arm>.json
"""
import argparse
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
N_BOOT = 2000

CKPT_MARKET = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')
CKPT_MSMT   = Path('/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth')

INDICES_FILE = RESULTS / 'e44_i1_percell_indices.json'

TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# ----------------------- backbone returning all four layer maps -----------------------
class BoTBackbone4L(nn.Module):
    """Same BoT ResNet-50 (layer4 stride override) as compute_i1_percell_preview.py,
    exposing the four layer maps so E-snap-A can match at any depth."""
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
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return l1, l2, l3, l4


def load_backbone_and_bn(ckpt_path):
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = BoTBackbone4L().to(device).eval()
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
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), int(s.person_id), int(s.camera_id)


class DStrain(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        return TRANSFORM(self.samples[i].load_image())


@torch.no_grad()
def extract_gap_vectors_and_terminal(backbone, bn, samples, batch=32, mode='test', tag=''):
    """Returns:
        gap: dict layer_name -> (N, C_layer) array of GAP vectors (pre-norm)
        term: (N, 2048) terminal embedding (BN(GAP(l4)), L2-normalised), used as the snap output
        pids, cams (test mode only)
    """
    is_train = (mode == 'train')
    ds = DStrain(samples) if is_train else DStest(samples)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    gap_chunks = {k: [] for k in ('layer1', 'layer2', 'layer3', 'layer4')}
    term_chunks = []
    pid_chunks = []; cam_chunks = []
    n_done = 0
    for item in ldr:
        if is_train:
            imgs = item.to(device, non_blocking=True)
        else:
            imgs, p, c = item
            imgs = imgs.to(device, non_blocking=True)
            pid_chunks.append(p.numpy()); cam_chunks.append(c.numpy())
        l1, l2, l3, l4 = backbone(imgs)
        # GAP per layer
        gap_chunks['layer1'].append(l1.mean(dim=(2, 3)).cpu().numpy())
        gap_chunks['layer2'].append(l2.mean(dim=(2, 3)).cpu().numpy())
        gap_chunks['layer3'].append(l3.mean(dim=(2, 3)).cpu().numpy())
        gap_l4 = l4.mean(dim=(2, 3))
        gap_chunks['layer4'].append(gap_l4.cpu().numpy())
        # Terminal embedding = BN(gap_l4) L2-normalised
        term = bn(gap_l4)
        term_np = term.cpu().numpy()
        term_np /= (np.linalg.norm(term_np, axis=1, keepdims=True) + 1e-12)
        term_chunks.append(term_np)
        n_done += imgs.shape[0]
        if n_done % 512 == 0 or n_done == len(ds):
            print(f"      {tag} extract: {n_done}/{len(ds)}", flush=True)
    gap = {k: np.concatenate(v, axis=0) for k, v in gap_chunks.items()}
    term = np.concatenate(term_chunks, axis=0)
    if is_train:
        return gap, term
    return gap, term, np.concatenate(pid_chunks), np.concatenate(cam_chunks)


def l2(X, axis=1):
    n = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / (n + 1e-12)


def per_query_ap(q_emb, q_pids, q_cams, g_emb, g_pids, g_cams):
    """Returns (aps, valid_mask, r1). Market/MSMT-style eval: drop same-pid-same-cam
    gallery items as junk for each query."""
    sims = q_emb @ g_emb.T
    aps = np.zeros(len(q_pids), dtype=np.float64)
    valid = np.zeros(len(q_pids), dtype=bool)
    r1 = np.zeros(len(q_pids), dtype=bool)
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
        r1[i] = good[0]
    return aps, valid, r1


def paired_bootstrap_layers(aps_dict, valid_dict, pids, n_boot=2000, seed=0):
    """Paired identity-clustered bootstrap over the layer sweep (Channel 4 rev 628 §2).

    For each replicate: resample identities with replacement ONCE, pull all of each
    drawn identity's queries, score mAP on that SAME set for every entry in
    aps_dict (keys: 'baseline', 'layer1', ..., 'layer4'). Returns:
      marginal: {name: (mean, lo95, hi95)}
      differences: for each ordered pair (A, B), {pair: (mean(A-B), lo95, hi95)}
    """
    rng = np.random.default_rng(seed)
    names = list(aps_dict.keys())
    # Only identities whose BASELINE is valid; a query invalid in baseline has
    # no mate in gallery, making snap vs baseline comparison meaningless there.
    # Snap-invalid-but-baseline-valid rows contribute AP = 0 to the snapped mean,
    # which is the honest treatment (snap failed to retrieve any mate).
    base_valid = valid_dict['baseline']
    p = pids[base_valid]
    unique_ids = np.unique(p)
    id_to_idx = {int(pid): np.where(p == pid)[0] for pid in unique_ids}
    n_id = len(unique_ids)

    aps_filt = {k: v[base_valid] for k, v in aps_dict.items()}

    boot_maps = {k: [] for k in names}
    for _ in range(n_boot):
        sampled = rng.choice(unique_ids, size=n_id, replace=True)
        q_idx = np.concatenate([id_to_idx[int(pid)] for pid in sampled])
        for k in names:
            boot_maps[k].append(aps_filt[k][q_idx].mean())
    boot_arrs = {k: np.array(v) for k, v in boot_maps.items()}

    marginal = {k: (float(aps_filt[k].mean()),
                    float(np.quantile(boot_arrs[k], 0.025)),
                    float(np.quantile(boot_arrs[k], 0.975)))
                for k in names}

    # All ordered pairs; interesting ones are (layerN, layerN+1) and (layerN, baseline)
    differences = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a == b:
                continue
            d = boot_arrs[a] - boot_arrs[b]
            differences[f'{a}_minus_{b}'] = (
                float((aps_filt[a] - aps_filt[b]).mean()),
                float(np.quantile(d, 0.025)),
                float(np.quantile(d, 0.975)),
            )
    return marginal, differences, int(n_id)


def snap_indices(q_gap_L, train_gap_L):
    """Return nearest-training-sample index for each test sample at layer L.
    Both inputs are (N, C_L). Cosine similarity, argmax."""
    q = l2(q_gap_L)
    t = l2(train_gap_L)
    sims = q @ t.T
    return sims.argmax(axis=1)


def anchor_distribution_stats(nn_idx, n_train):
    """P3 ingredients: distinct-anchor count, top-k mass."""
    counts = np.bincount(nn_idx, minlength=n_train)
    nz = counts[counts > 0]
    nz_sorted = -np.sort(-nz)
    total = len(nn_idx)
    cum = np.cumsum(nz_sorted) / total
    return {
        'n_distinct_anchors': int(len(nz)),
        'n_train_pool': int(n_train),
        'anchors_used_fraction': float(len(nz) / n_train),
        'top1_fraction': float(nz_sorted[0] / total) if len(nz_sorted) else 0.0,
        'top10_fraction': float(cum[min(9, len(cum) - 1)]) if len(cum) else 0.0,
        'top100_fraction': float(cum[min(99, len(cum) - 1)]) if len(cum) else 0.0,
        'anchor_count_median': float(np.median(nz)) if len(nz) else 0.0,
    }


def load_saved_bprime_sub_indices():
    """B'-sub subsample indices from the posthoc_v3 replay (satisfies Ruling 43,
    matches baseline 9.12%). Returns (q_idx, g_idx) into MSMT non-junk lists."""
    idx = json.load(open(INDICES_FILE))
    arms = idx['scripts']['compute_i1_percell_posthoc_v3.py']['arms']
    return np.array(arms['B_prime']['q_idx']), np.array(arms['B_prime']['g_idx'])


def load_train_pool_indices(script_key='compute_i1_percell_posthoc_v3.py'):
    """The train-pool subsample in the saved replay used N_TRAIN_POOL=3000, same
    seed. Because the index file does not store the train-pool index list
    explicitly (it was drawn and discarded before the arm draws), we re-run
    rng.choice(len(market.train), 3000) with seed 42 to recover it."""
    rng = np.random.default_rng(RNG_SEED)
    # match the replay script's draw order: first draw is the train pool
    return rng.choice(12936, size=N_TRAIN_POOL, replace=False)


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32):
    print(f"\n=== arm {arm_name} (model={model_name}, train={train_dataset_name}, eval={eval_dataset_name}) ===", flush=True)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    tr_idx = rng.choice(len(train_all), size=N_TRAIN_POOL, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    print(f"  train pool: {len(train_samples)} / {len(train_all)}", flush=True)

    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
    if q_idx is None:
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
    if g_idx is None:
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
    query_samples = [q_all[int(i)] for i in q_idx]
    gallery_samples = [g_all[int(i)] for i in g_idx]
    print(f"  query: {len(query_samples)} / {len(q_all)} non-junk", flush=True)
    print(f"  gallery: {len(gallery_samples)} / {len(g_all)} non-junk", flush=True)

    backbone, bn = load_backbone_and_bn(model_ckpt)
    print(f"  backbone loaded: {model_ckpt.name}", flush=True)

    tr_gap, tr_term = extract_gap_vectors_and_terminal(
        backbone, bn, train_samples, batch=batch, mode='train', tag='train')
    q_gap, q_term, q_pids, q_cams = extract_gap_vectors_and_terminal(
        backbone, bn, query_samples, batch=batch, mode='test', tag='query')
    g_gap, g_term, g_pids, g_cams = extract_gap_vectors_and_terminal(
        backbone, bn, gallery_samples, batch=batch, mode='test', tag='gallery')

    # Baseline: raw terminal embeddings
    base_aps, base_valid, base_r1 = per_query_ap(
        q_term, q_pids, q_cams, g_term, g_pids, g_cams)
    base_map = float(base_aps[base_valid].mean()) if base_valid.any() else float('nan')
    base_r1_rate = float(base_r1[base_valid].mean()) if base_valid.any() else float('nan')
    print(f"\n  BASELINE (no snap): mAP={100*base_map:.3f}%  R1={100*base_r1_rate:.3f}%  "
          f"n_valid={int(base_valid.sum())}", flush=True)

    out = {
        'rev': 629,
        'arm': arm_name,
        'model_ckpt': str(model_ckpt),
        'train_dataset': train_dataset_name,
        'eval_dataset': eval_dataset_name,
        'variant': 'a (GAP-at-layer-L nearest-match, snap both q and g)',
        'params': {'N_QUERY': len(query_samples), 'N_GALLERY': len(gallery_samples),
                   'N_TRAIN_POOL': len(train_samples), 'RNG_SEED': RNG_SEED,
                   'n_boot': N_BOOT},
        'baseline': {
            'mAP': base_map, 'rank1': base_r1_rate,
            'n_valid': int(base_valid.sum()),
        },
        'per_layer': {},
    }

    aps_dict = {'baseline': base_aps}
    valid_dict = {'baseline': base_valid}
    r1_dict = {'baseline': base_r1}

    for Lname in ['layer1', 'layer2', 'layer3', 'layer4']:
        print(f"\n  --- snap at {Lname} ---", flush=True)
        q_nn = snap_indices(q_gap[Lname], tr_gap[Lname])
        g_nn = snap_indices(g_gap[Lname], tr_gap[Lname])
        q_snapped = tr_term[q_nn]
        g_snapped = tr_term[g_nn]
        aps, valid, r1 = per_query_ap(
            q_snapped, q_pids, q_cams, g_snapped, g_pids, g_cams)
        snap_map = float(aps[valid].mean()) if valid.any() else float('nan')
        snap_r1 = float(r1[valid].mean()) if valid.any() else float('nan')
        q_p3 = anchor_distribution_stats(q_nn, len(train_samples))
        g_p3 = anchor_distribution_stats(g_nn, len(train_samples))
        ratio = snap_map / base_map if base_map and base_map == base_map else float('nan')
        print(f"    mAP={100*snap_map:.3f}%  R1={100*snap_r1:.3f}%  "
              f"ratio_vs_baseline={ratio:.3f}  "
              f"q_distinct={q_p3['n_distinct_anchors']}/{q_p3['n_train_pool']}  "
              f"g_distinct={g_p3['n_distinct_anchors']}/{g_p3['n_train_pool']}", flush=True)
        out['per_layer'][Lname] = {
            'mAP': snap_map, 'rank1': snap_r1,
            'mAP_ratio_to_baseline': ratio,
            'n_valid': int(valid.sum()),
            'P3_query_anchor_stats': q_p3,
            'P3_gallery_anchor_stats': g_p3,
            'P3_anchor_pool_label': 'variant_a: pool = N_TRAIN_POOL images',
        }
        aps_dict[Lname] = aps
        valid_dict[Lname] = valid
        r1_dict[Lname] = r1

    # Paired identity-clustered bootstrap over layers (Channel 4 rev 628 §2).
    # Baseline-valid rows only; snap-invalid rows contribute AP=0 (honest "snap
    # failed to retrieve any mate"), which the filter in paired_bootstrap_layers
    # preserves since it only drops baseline-invalid.
    print(f"\n  --- paired identity-clustered bootstrap, B={N_BOOT} ---", flush=True)
    marginal, differences, n_id = paired_bootstrap_layers(
        aps_dict, valid_dict, q_pids, n_boot=N_BOOT, seed=RNG_SEED + 100)
    print(f"  n_identities (baseline-valid): {n_id}")
    print(f"  marginal mAP (name, mean, 95% CI):")
    for k, (m, lo, hi) in marginal.items():
        print(f"    {k:>10s}: {100*m:7.3f}%  [{100*lo:+7.3f}, {100*hi:+7.3f}]")
    print(f"  key paired differences (A - B, mean, 95% CI):")
    pairs_of_interest = [
        ('layer2', 'layer1'), ('layer3', 'layer2'), ('layer4', 'layer3'),
        ('layer1', 'baseline'), ('layer2', 'baseline'),
        ('layer3', 'baseline'), ('layer4', 'baseline'),
    ]
    paired_out = {}
    for a, b in pairs_of_interest:
        key = f'{a}_minus_{b}'
        m, lo, hi = differences[key]
        print(f"    {key:>22s}: {100*m:+7.3f} pp  [{100*lo:+7.3f}, {100*hi:+7.3f}]")
        paired_out[key] = {'mean_pp': m, 'lo95_pp': lo, 'hi95_pp': hi}
    out['paired_bootstrap'] = {
        'n_boot': N_BOOT, 'n_identities': n_id,
        'marginal_mAP': {k: {'mean': v[0], 'lo95': v[1], 'hi95': v[2]}
                         for k, v in marginal.items()},
        'key_differences': paired_out,
    }

    outp = RESULTS / f'e44_e_snap_a_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['B_prime_sub', 'C', 'both'], default='B_prime_sub')
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    if args.arm in ('B_prime_sub', 'both'):
        q_idx, g_idx = load_saved_bprime_sub_indices()
        run_arm('B_prime_sub', CKPT_MARKET, model_name='BoT-Market',
                train_dataset_name='market1501', eval_dataset_name='msmt17',
                q_idx=q_idx, g_idx=g_idx, batch=args.batch)

    if args.arm in ('C', 'both'):
        run_arm('C', CKPT_MSMT, model_name='BoT-MSMT',
                train_dataset_name='msmt17', eval_dataset_name='market1501',
                q_idx=None, g_idx=None, batch=args.batch)


if __name__ == '__main__':
    main()
