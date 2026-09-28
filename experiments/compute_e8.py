"""E8 — Does F_k concentration predict the Market → MSMT17 transfer gap?

For each checkpoint compute:
  1. F_k per keypoint slot on Market-1501 test IDs (same Gaussian-pooling pipeline as R6/E2b)
  2. Concentration of F_k: entropy and Gini over the 6-slot distribution
  3. Market-1501 mAP (in-domain) and MSMT17 mAP (zero-shot, cross-domain)
  4. Transfer gap = Market mAP − MSMT17 mAP

Checkpoints (all ResNet-50, all Market-1501-trained):
  moco_market1501_final   — self-supervised, expected low concentration
  bot_market1501_epoch40  — supervised, early
  bot_market1501_epoch80  — supervised, mid
  bot_market1501_epoch120 — supervised, late
  bot_market1501_final    — supervised, converged

Outputs /tmp/e8_results.json.
"""

import json, pickle, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, '/workspace/reid-tk')

from reid_tk.datasets import get_dataset
from reid_tk.metrics import compute_retrieval_metrics

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR   = Path("/data/papers/low_dimensional_reid/pretrained")
KPT_CACHE  = Path("/tmp/pacm_cache_vitpose/keypoints")
CONF       = 0.3
IMG_H, IMG_W = 256, 128
FEAT_H, FEAT_W = 16, 8
SIGMA      = 1.0
SEED       = 42
BATCH_SIZE = 64
N_PAIRS    = 4

BODY_PAIRS = [(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
KPT_NAMES  = ['Sh','El','Wr','Hi','Kn','An']
K = len(BODY_PAIRS)

random.seed(SEED); np.random.seed(SEED)

CHECKPOINTS = [
    {'name': 'moco_final',  'path': 'moco_market1501_final.pth',    'type': 'moco'},
    {'name': 'bot_ep40',    'path': 'bot_market1501_epoch40.pth',   'type': 'bot'},
    {'name': 'bot_ep80',    'path': 'bot_market1501_epoch80.pth',   'type': 'bot'},
    {'name': 'bot_ep120',   'path': 'bot_market1501_epoch120.pth',  'type': 'bot'},
    {'name': 'bot_final',   'path': 'bot_market1501_final.pth',     'type': 'bot'},
]

TF = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model loading ─────────────────────────────────────────────────────────────

def build_resnet_base():
    resnet = models.resnet50(weights=None)
    resnet.layer4[0].conv2.stride = (1, 1)
    resnet.layer4[0].downsample[0].stride = (1, 1)
    return nn.Sequential(*list(resnet.children())[:-2])   # (B, 2048, 16, 8)


def load_spatial_and_extractor(ckpt_info):
    """Return (spatial_model, full_extractor) for F_k and retrieval respectively."""
    path = CKPT_DIR / ckpt_info['path']
    raw = torch.load(path, map_location=DEVICE)
    base = build_resnet_base()

    if ckpt_info['type'] == 'bot':
        bottleneck = nn.BatchNorm1d(2048)
        bottleneck.bias.requires_grad_(False)
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            # Epoch checkpoint: keys are base.*, bottleneck.*, classifier.*
            sd = raw['model_state_dict']
            base_sd = {k[len('base.'):]: v for k, v in sd.items() if k.startswith('base.')}
            bn_sd   = {k[len('bottleneck.'):]: v for k, v in sd.items() if k.startswith('bottleneck.')}
            base.load_state_dict(base_sd)
            bottleneck.load_state_dict(bn_sd)
        else:
            # Final checkpoint: flat Sequential keys (0.*, 1.*, 2.*, 3.*)
            extractor = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bottleneck)
            extractor.load_state_dict(raw)
        spatial = base
        extractor = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten(), bottleneck)
    else:
        # MoCo: flat base Sequential keys (0.weight, 1.weight, 4.*.*, ...)
        base.load_state_dict(raw)
        spatial = base
        extractor = nn.Sequential(base, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    spatial.to(DEVICE).eval()
    extractor.to(DEVICE).eval()
    return spatial, extractor


# ── Gaussian weighting ────────────────────────────────────────────────────────

def gauss_weights(nx, ny):
    pc = np.clip(nx, 0, 1) * (FEAT_W - 1)
    pr = np.clip(ny, 0, 1) * (FEAT_H - 1)
    rs = np.arange(FEAT_H, dtype=np.float32)
    cs = np.arange(FEAT_W, dtype=np.float32)
    R, C = np.meshgrid(rs, cs, indexing='ij')
    g = np.exp(-((R - pr)**2 + (C - pc)**2) / (2 * SIGMA**2))
    g /= g.sum()
    return g.reshape(-1)


# ── F_k pipeline ──────────────────────────────────────────────────────────────

def compute_fk(by_id):
    rng = random.Random(SEED)
    pids = list(by_id.keys())
    wi_sum  = np.zeros(K); wi_cnt  = np.zeros(K, dtype=int)
    btw_sum = np.zeros(K); btw_cnt = np.zeros(K, dtype=int)
    for pid in pids:
        feats = by_id[pid]
        if len(feats) < 2: continue
        pairs = [(a,b) for i,a in enumerate(feats) for b in feats[i+1:]]
        for fa, fb in rng.sample(pairs, min(N_PAIRS, len(pairs))):
            for s in range(K):
                if s in fa and s in fb:
                    d = float(np.linalg.norm(fa[s] - fb[s]))
                    wi_sum[s] += d; wi_cnt[s] += 1
        negs = rng.sample([p for p in pids if p != pid], min(N_PAIRS, len(pids)-1))
        for np_pid in negs:
            fa = rng.choice(feats); fb = rng.choice(by_id[np_pid])
            for s in range(K):
                if s in fa and s in fb:
                    d = float(np.linalg.norm(fa[s] - fb[s]))
                    btw_sum[s] += d; btw_cnt[s] += 1
    within  = np.where(wi_cnt  > 0, wi_sum  / wi_cnt,  np.nan)
    between = np.where(btw_cnt > 0, btw_sum / btw_cnt, np.nan)
    return between / (within + 1e-6), within, between


@torch.no_grad()
def extract_fk(samples, kpts_all, spatial, label=''):
    by_id = defaultdict(list)
    imgs, meta = [], []
    for i, (s, kpts) in enumerate(tqdm(
        zip(samples, kpts_all), total=len(samples), desc=f'fk {label}'
    )):
        if s.person_id <= 0: continue
        det = [(kidx, kpts[kidx]) for kidx in range(17) if kpts[kidx, 2] >= CONF]
        if not det: continue
        img = s.load_image(); W0, H0 = img.size
        imgs.append(TF(img)); meta.append((s.person_id, det, W0, H0))
        if len(imgs) < BATCH_SIZE and i < len(samples) - 1: continue
        batch = torch.stack(imgs).to(DEVICE)
        fm = spatial(batch).cpu().numpy()
        for b, (pid, det_b, W0, H0) in enumerate(meta):
            fm_flat = fm[b].reshape(2048, -1).T
            sf = {}
            for kidx, kpt in det_b:
                nx, ny = float(kpt[0])/W0, float(kpt[1])/H0
                gw = gauss_weights(nx, ny)
                fv = (fm_flat * gw[:, None]).sum(0)
                for sl, (l, r) in enumerate(BODY_PAIRS):
                    if kidx in (l, r):
                        sf[sl] = (sf[sl] + fv)/2 if sl in sf else fv
            if sf: by_id[pid].append(sf)
        imgs, meta = [], []
    return compute_fk(dict(by_id))


# ── Retrieval evaluation ──────────────────────────────────────────────────────

class _DS(Dataset):
    def __init__(self, samples): self.s = samples
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        return TF(s.load_image()), s.person_id, s.camera_id


@torch.no_grad()
def eval_map(extractor, samples, label=''):
    loader = DataLoader(_DS(samples), batch_size=BATCH_SIZE, num_workers=0)
    embs, pids, cams = [], [], []
    for imgs, ids, cs in tqdm(loader, desc=f'emb {label}'):
        e = extractor(imgs.to(DEVICE))
        embs.append(F.normalize(e, p=2, dim=1).cpu().numpy())
        pids.extend(ids.numpy()); cams.extend(cs.numpy())
    return np.vstack(embs), np.array(pids), np.array(cams)


def retrieval_map(q_emb, g_emb, q_ids, g_ids, q_cams, g_cams):
    dist = -q_emb @ g_emb.T   # cosine distance (negated similarity)
    aps, r1, valid = [], 0, 0
    for qi in range(len(q_ids)):
        qid, qcam = q_ids[qi], q_cams[qi]
        dists = dist[qi]
        order = np.argsort(dists)
        junk = (g_ids == -1) | ((g_ids == qid) & (g_cams == qcam))
        order = order[~junk[order]]
        matches = g_ids[order] == qid
        if not matches.any(): continue
        valid += 1
        if matches[0]: r1 += 1
        pos = np.where(matches)[0]
        aps.append(np.mean([(i+1)/(p+1) for i, p in enumerate(pos)]))
    return float(np.mean(aps)) if aps else 0.0, r1/len(q_ids)


# ── Concentration metrics ─────────────────────────────────────────────────────

def concentration(fk):
    fk = np.array(fk, dtype=float)
    fk = np.where(np.isfinite(fk), fk, 0.0)
    fk = np.clip(fk, 0, None)
    s = fk.sum()
    if s == 0: return float('nan'), float('nan'), float('nan')
    p = fk / s
    ent = -float(np.sum(p * np.log(p + 1e-12)))   # lower = more concentrated
    fk_range = float(fk.max() - fk.min())
    # Gini coefficient
    fk_sorted = np.sort(fk)
    n = len(fk_sorted)
    gini = float((2 * np.sum(np.arange(1, n+1) * fk_sorted) - (n+1)*fk_sorted.sum()) / (n * fk_sorted.sum()))
    return ent, gini, fk_range


# ── Main ──────────────────────────────────────────────────────────────────────

def load_kpts(tag, split):
    p = KPT_CACHE / f"{tag}_vitpose_conf{CONF}" / f"{split}.pkl"
    with open(p, 'rb') as f: return pickle.load(f)


def main():
    print(f"Device: {DEVICE}")

    # Load datasets once
    ds_mkt  = get_dataset('market1501', sample=False)
    ds_msmt = get_dataset('msmt17',     sample=False)
    kpts_mkt_q = load_kpts('market1501', 'query')
    kpts_mkt_g = load_kpts('market1501', 'gallery')
    kpts_msmt_q = load_kpts('msmt17', 'query')
    kpts_msmt_g = load_kpts('msmt17', 'gallery')

    mkt_test  = ds_mkt.query  + ds_mkt.gallery
    kpts_mkt  = kpts_mkt_q   + kpts_mkt_g
    msmt_test = ds_msmt.query + ds_msmt.gallery
    kpts_msmt = kpts_msmt_q  + kpts_msmt_g

    results = []

    for ckpt in CHECKPOINTS:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt['name']}  ({ckpt['type']})")
        spatial, extractor = load_spatial_and_extractor(ckpt)

        # F_k on Market test IDs
        fk, sw, sb = extract_fk(mkt_test, kpts_mkt, spatial, ckpt['name'])[:3]
        ent, gini, fk_range = concentration(fk)
        fk_rank = [KPT_NAMES[i] for i in np.argsort(fk)[::-1]]
        print(f"  F_k: {[f'{v:.3f}' for v in fk]}")
        print(f"  Rank: {' > '.join(fk_rank)}")
        print(f"  Entropy={ent:.4f}  Gini={gini:.4f}  Range={fk_range:.4f}")

        # Retrieval: Market (in-domain)
        q_emb, q_ids, q_cams = eval_map(extractor, ds_mkt.query,   f'{ckpt["name"]}-mkt-q')
        g_emb, g_ids, g_cams = eval_map(extractor, ds_mkt.gallery, f'{ckpt["name"]}-mkt-g')
        mkt_map, mkt_r1 = retrieval_map(q_emb, g_emb, q_ids, g_ids, q_cams, g_cams)
        print(f"  Market  mAP={mkt_map*100:.1f}%  R1={mkt_r1*100:.1f}%")

        # Retrieval: MSMT17 (cross-domain, zero-shot)
        qm_emb, qm_ids, qm_cams = eval_map(extractor, ds_msmt.query,   f'{ckpt["name"]}-msmt-q')
        gm_emb, gm_ids, gm_cams = eval_map(extractor, ds_msmt.gallery, f'{ckpt["name"]}-msmt-g')
        msmt_map, msmt_r1 = retrieval_map(qm_emb, gm_emb, qm_ids, gm_ids, qm_cams, gm_cams)
        print(f"  MSMT17  mAP={msmt_map*100:.1f}%  R1={msmt_r1*100:.1f}%")

        gap = mkt_map - msmt_map
        print(f"  Transfer gap = {gap*100:.1f}pp")

        results.append({
            'name': ckpt['name'], 'type': ckpt['type'],
            'fk': fk.tolist(), 'sw': sw.tolist(), 'sb': sb.tolist(),
            'entropy': ent, 'gini': gini, 'fk_range': fk_range,
            'mkt_map': mkt_map, 'mkt_r1': mkt_r1,
            'msmt_map': msmt_map, 'msmt_r1': msmt_r1,
            'transfer_gap': gap,
        })

    # Correlations
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  {'Name':<18}  {'Entropy':>8}  {'Gini':>6}  {'Range':>6}  {'MktmAP':>7}  {'MsmtmAP':>8}  {'Gap':>6}")
    for r in results:
        print(f"  {r['name']:<18}  {r['entropy']:>8.4f}  {r['gini']:>6.4f}  "
              f"{r['fk_range']:>6.4f}  {r['mkt_map']*100:>6.1f}%  "
              f"{r['msmt_map']*100:>7.1f}%  {r['transfer_gap']*100:>5.1f}pp")

    gaps = [r['transfer_gap'] for r in results]
    for metric, label in [('entropy', 'entropy (low=concentrated)'),
                           ('gini',    'Gini (high=concentrated)'),
                           ('fk_range','F_k range')]:
        vals = [r[metric] for r in results]
        # For entropy: lower = more concentrated, so flip sign for "concentration"
        rho, p = spearmanr(vals, gaps)
        direction = "(expect negative for entropy)" if metric == 'entropy' else "(expect positive)"
        print(f"\n  rho({label}, transfer_gap) = {rho:.3f}  p={p:.4f}  {direction}")

    out = Path('/tmp/e8_results.json')
    with open(out, 'w') as f:
        json.dump({'checkpoints': results, 'kpt_names': KPT_NAMES}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
