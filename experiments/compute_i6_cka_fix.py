"""
I-6 CI fix (rev 524 §1): bootstrap-with-replacement for linear CKA is
biased upward due to tied-row Gram-matrix diagonal inflation.
A' point estimate 0.1505 sat below its bootstrap CI [0.152, 0.158] — a
clean signature of this bias.

Fix: subsample WITHOUT replacement at 80% of n. No duplicates, no
diagonal inflation, point estimate recovered.

Also report more decimals on depth sweep (layer3 "exactly +0.0000" was
a formatting artifact) and fold in the rank-expansion framing per §5.
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
N_PROBE = 13115
N_BOOT = 100
SUBSAMPLE_FRAC = 0.80  # subsample WITHOUT replacement

STAGES = ['stem', 'layer1', 'layer2', 'layer3', 'layer4']

CKPT_MARKET = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_final.pth')
CKPT_MSMT   = Path('/data/papers/low_dimensional_reid/pretrained/bot_resnet50_msmt17_seed42_final.pth')

TRANSFORM = T.Compose([T.Resize((256, 128)), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class BoTSpatial(nn.Module):
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
        s = x
        l1 = self.layer1(x); l2 = self.layer2(l1)
        l3 = self.layer3(l2); l4 = self.layer4(l3)
        return {
            'stem':   s.mean(dim=(2, 3)),
            'layer1': l1.mean(dim=(2, 3)),
            'layer2': l2.mean(dim=(2, 3)),
            'layer3': l3.mean(dim=(2, 3)),
            'layer4': l4.mean(dim=(2, 3)),
        }


class BoTSpatialIN(BoTSpatial):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet.layer4[0].conv2.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.stem_conv = resnet.conv1; self.stem_bn = resnet.bn1
        self.stem_relu = resnet.relu; self.stem_pool = resnet.maxpool
        self.layer1 = resnet.layer1; self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3; self.layer4 = resnet.layer4


def load_into(model, ckpt_path):
    sd_raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {}
    MAP = {'0.0.': 'stem_conv.', '0.1.': 'stem_bn.',
           '0.4.': 'layer1.', '0.5.': 'layer2.',
           '0.6.': 'layer3.', '0.7.': 'layer4.'}
    for k, v in sd_raw.items():
        if 'num_batches_tracked' in k:
            continue
        for prefix, target in MAP.items():
            if k.startswith(prefix):
                sd[target + k[len(prefix):]] = v
                break
    model.load_state_dict(sd, strict=False)
    return model


class DS(Dataset):
    def __init__(self, samples):
        self.samples = [s for s in samples if not s.is_junk]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return TRANSFORM(s.load_image()), int(s.person_id)


@torch.no_grad()
def extract_stages(model, samples, batch=48):
    ds = DS(samples)
    print(f"      extracting {len(ds)} items...", flush=True)
    ldr = DataLoader(ds, batch_size=batch, num_workers=0, pin_memory=False)
    stages = {s: [] for s in STAGES}
    labels = []
    for imgs, lbls in ldr:
        out = model(imgs.to(device, non_blocking=True))
        for s in STAGES:
            stages[s].append(out[s].cpu().numpy())
        labels.append(lbls.numpy())
    return {s: np.vstack(v) for s, v in stages.items()}, np.concatenate(labels)


def linear_cka(A, B):
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    AB = A.T @ B
    AA = A.T @ A
    BB = B.T @ B
    num = (AB * AB).sum()
    den = np.sqrt((AA * AA).sum() * (BB * BB).sum())
    if den < 1e-20:
        return float('nan')
    return float(num / den)


def participation_ratio(X):
    X = X - X.mean(axis=0, keepdims=True)
    S = X.T @ X
    eigs = np.linalg.eigvalsh(S)
    eigs = np.clip(eigs, 0, None)
    s1 = eigs.sum()
    s2 = (eigs * eigs).sum()
    if s2 < 1e-20:
        return float('nan')
    return float((s1 * s1) / s2)


def effective_rank(X):
    X = X - X.mean(axis=0, keepdims=True)
    S = X.T @ X
    eigs = np.linalg.eigvalsh(S)
    eigs = np.clip(eigs, 0, None)
    s = eigs.sum()
    if s < 1e-20:
        return float('nan')
    p = eigs / s
    p = p[p > 1e-20]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def topk_project(X, k):
    X = X - X.mean(axis=0, keepdims=True)
    S = X.T @ X
    eigs, V = np.linalg.eigh(S)
    order = np.argsort(eigs)[::-1][:k]
    V_k = V[:, order]
    return X @ V_k


def subsample_ci(A, B, n_boot=N_BOOT, frac=SUBSAMPLE_FRAC, seed=0):
    """Subsample WITHOUT replacement at `frac` of n. No duplicate rows ->
    no Gram-matrix diagonal inflation, no upward bias in linear CKA."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    k = int(np.floor(frac * n))
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=k, replace=False)
        vals.append(linear_cka(A[idx], B[idx]))
    vals = np.array(vals)
    return (float(vals.mean()),
            float(np.quantile(vals, 0.025)),
            float(np.quantile(vals, 0.975)))


def main():
    print(f"device: {device}", flush=True)
    market = get_dataset('market1501', sample=False)
    msmt = get_dataset('msmt17', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    print("Loading models...", flush=True)
    m_mkt = BoTSpatial().to(device).eval()
    load_into(m_mkt, CKPT_MARKET)
    m_msmt = BoTSpatial().to(device).eval()
    load_into(m_msmt, CKPT_MSMT)
    m_in = BoTSpatialIN().to(device).eval()

    print("Preparing probe samples...", flush=True)
    def gallery(ds):
        n_q = sum(1 for s in ds.query if not s.is_junk)
        return [s for s in (ds.query + ds.gallery)[n_q:] if not s.is_junk]

    mkt_gallery = gallery(market)
    msmt_gallery = gallery(msmt)
    n_match = min(len(mkt_gallery), len(msmt_gallery), N_PROBE)
    print(f"  matched probe size: {n_match}", flush=True)

    mkt_idx = rng.choice(len(mkt_gallery), size=n_match, replace=False)
    msmt_idx = rng.choice(len(msmt_gallery), size=n_match, replace=False)
    mkt_probe = [mkt_gallery[i] for i in mkt_idx]
    msmt_probe = [msmt_gallery[i] for i in msmt_idx]

    probes = {'market': mkt_probe, 'msmt': msmt_probe}
    feats = {}
    for probe_name, probe in probes.items():
        print(f"  probe={probe_name}", flush=True)
        for model_name, model in [('market_trained', m_mkt),
                                  ('msmt_trained', m_msmt),
                                  ('imagenet_init', m_in)]:
            print(f"    model={model_name}", flush=True)
            fmap, _ = extract_stages(model, probe)
            feats.setdefault(probe_name, {})[model_name] = fmap
            torch.cuda.empty_cache()

    ARMS = {
        'A_prime': ('market_trained', 'market'),
        'B_prime': ('market_trained', 'msmt'),
        'C':       ('msmt_trained',   'market'),
        'D':       ('msmt_trained',   'msmt'),
    }

    all_out = {}
    for stage in STAGES:
        print(f"\n### stage {stage} ###", flush=True)
        stage_out = {}
        cell_cka = {}
        for i, (arm, (model_name, probe_name)) in enumerate(ARMS.items()):
            T = feats[probe_name][model_name][stage]
            IN = feats[probe_name]['imagenet_init'][stage]
            cka_point = linear_cka(T, IN)
            cka_mean, cka_lo, cka_hi = subsample_ci(T, IN, n_boot=N_BOOT,
                                                   frac=SUBSAMPLE_FRAC,
                                                   seed=RNG_SEED + i * 101)
            pr_t = participation_ratio(T)
            er_t = effective_rank(T)
            pr_i = participation_ratio(IN)
            er_i = effective_rank(IN)
            cell_cka[arm] = cka_point
            stage_out[arm] = {
                'cka_point': cka_point,
                'cka_subs_mean': cka_mean,
                'cka_subs_lo95': cka_lo,
                'cka_subs_hi95': cka_hi,
                'pr_trained': pr_t,
                'er_trained': er_t,
                'pr_init': pr_i,
                'er_init': er_i,
            }
            print(f"  {arm:<8s}: CKA={cka_point:.6f}  subs mean={cka_mean:.6f} "
                  f"[{cka_lo:.6f}, {cka_hi:.6f}]  "
                  f"PR_t={pr_t:.2f} ER_t={er_t:.2f}  "
                  f"PR_i={pr_i:.2f} ER_i={er_i:.2f}", flush=True)

        probe_main = 0.5 * ((cell_cka['A_prime'] + cell_cka['C']) -
                            (cell_cka['B_prime'] + cell_cka['D']))
        model_main = 0.5 * ((cell_cka['A_prime'] + cell_cka['B_prime']) -
                            (cell_cka['C'] + cell_cka['D']))
        own_to_foreign = ((cell_cka['B_prime'] - cell_cka['A_prime']) +
                          (cell_cka['C'] - cell_cka['D'])) * 0.5
        full_contrast = ((cell_cka['A_prime'] + cell_cka['D']) -
                         (cell_cka['B_prime'] + cell_cka['C']))
        stage_out['_main_probe_market_minus_msmt'] = probe_main
        stage_out['_main_model_market_minus_msmt'] = model_main
        stage_out['_mean_own_to_foreign_elevation'] = own_to_foreign
        stage_out['_full_2x2_contrast'] = full_contrast
        print(f"  main(probe M-MSMT)={probe_main:+.6f}  "
              f"main(model M-MSMT)={model_main:+.6f}", flush=True)
        print(f"  mean own->foreign elevation = {own_to_foreign:+.6f}", flush=True)
        print(f"  full 2x2 contrast (A'+D)-(B'+C) = {full_contrast:+.6f}", flush=True)

        if stage == 'layer4':
            min_pr = int(np.floor(min(stage_out[a]['pr_trained'] for a in ARMS)))
            min_pr = max(min_pr, 2)
            print(f"  rank-matched top-k k={min_pr}", flush=True)
            cell_cka_rm = {}
            for arm, (model_name, probe_name) in ARMS.items():
                T = feats[probe_name][model_name][stage]
                IN = feats[probe_name]['imagenet_init'][stage]
                T_rm = topk_project(T, min_pr)
                IN_rm = topk_project(IN, min_pr)
                cka_rm = linear_cka(T_rm, IN_rm)
                cell_cka_rm[arm] = cka_rm
                stage_out[arm]['cka_rank_matched'] = cka_rm
                print(f"    {arm}: CKA_rm={cka_rm:.6f}", flush=True)
            own_to_foreign_rm = ((cell_cka_rm['B_prime'] - cell_cka_rm['A_prime']) +
                                 (cell_cka_rm['C'] - cell_cka_rm['D'])) * 0.5
            stage_out['_mean_own_to_foreign_elevation_rank_matched'] = own_to_foreign_rm
            stage_out['_k_rank_matched'] = int(min_pr)
            print(f"  mean own->foreign elevation (rank-matched) = {own_to_foreign_rm:+.6f}",
                  flush=True)

        all_out[stage] = stage_out

    outp = RESULTS / 'e44_i6_cka_fix.json'
    with open(outp, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {outp}")


if __name__ == '__main__':
    main()
