"""
E16: source-diversity training sweep.
8 arms: D1 (2 cams), D2 (4), D3 (6), D4 (9), D5 (12), D6 (15), C1 (6 diff cams), C2 (15, half IDs).
Fixed image budget = D1 total (7914 images). No REA. Seed 42. 120 epochs.
Evaluate all on Market-1501.
"""
import sys; sys.path.insert(0, '/workspace/reid-tk')
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
import time
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader

from scripts.train_bag_of_tricks import (
    BagOfTricksModel, TripletLoss, CrossEntropyLabelSmooth,
    ReIDDataset, setup_logging,
)
from reid_tk.datasets import get_dataset
from reid_tk.samplers import RandomIdentitySampler

OUT_DIR   = Path('/data/papers/low_dimensional_reid/pretrained/e16')
LOG_DIR   = Path('/data/papers/low_dimensional_reid/logs')
MKT_ROOT  = Path('/data/datasets/reid/Market-1501-v15.09.15')
MSMT_ROOT = Path('/data/datasets/reid/MSMT17_V1')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
SEED         = 42
EPOCHS       = 120
BATCH_SIZE   = 128
NUM_INSTANCES= 4
BASE_LR      = 0.00035
MARGIN       = 0.3
LABEL_SMOOTH = 0.1
WARMUP       = 10
NUM_WORKERS  = 0
IMAGE_BUDGET = 7914
device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

ARM_CAMERAS = {
    'D1': [1, 13],
    'D2': [1, 5, 11, 13],
    'D3': [1, 4,  7,  9, 13, 15],
    'D4': [1, 4,  5,  7,  9, 11, 13, 14, 15],
    'D5': [1, 4,  5,  6,  7,  8,  9, 11, 12, 13, 14, 15],
    'D6': list(range(1, 16)),
    'C1': [5, 6, 8, 10, 11, 14],
    'C2': list(range(1, 16)),
}

def seed_everything(s):
    import random; random.seed(s)
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def subsample_stratified(samples, budget, seed):
    if len(samples) <= budget:
        return list(samples)
    rng = np.random.default_rng(seed)
    cam_to_idx = defaultdict(list)
    for i, s in enumerate(samples):
        cam_to_idx[s.camera_id].append(i)
    total = len(samples)
    chosen = []
    for cam, idxs in cam_to_idx.items():
        n_take = max(1, round(budget * len(idxs) / total))
        take = rng.choice(idxs, min(n_take, len(idxs)), replace=False)
        chosen.extend(take.tolist())
    chosen = list(set(chosen))
    if len(chosen) > budget:
        chosen = rng.choice(chosen, budget, replace=False).tolist()
    elif len(chosen) < budget:
        rest = [i for i in range(total) if i not in set(chosen)]
        extra = rng.choice(rest, min(budget - len(chosen), len(rest)), replace=False)
        chosen.extend(extra.tolist())
    return [samples[i] for i in sorted(set(chosen))]

TRANSFORM_EVAL = T.Compose([
    T.Resize((256, 128)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class EvalDS(torch.utils.data.Dataset):
    def __init__(self, samps):
        self.samps = samps
    def __len__(self): return len(self.samps)
    def __getitem__(self, i):
        img = Image.open(self.samps[i].image_path).convert('RGB')
        return TRANSFORM_EVAL(img), self.samps[i].person_id, self.samps[i].camera_id

def extract_embeddings(model, samps, batch=128):
    dl = DataLoader(EvalDS(samps), batch_size=batch, num_workers=NUM_WORKERS, pin_memory=True)
    embs, pids, cids = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, ps, cs in dl:
            f = model(imgs.to(device))   # eval mode returns feat directly
            f = F.normalize(f, dim=1)
            embs.append(f.cpu().numpy())
            pids.extend(ps.numpy().tolist())
            cids.extend(cs.numpy().tolist())
    return np.vstack(embs), np.array(pids), np.array(cids)

def compute_map_r1(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    from torchreid.reid.metrics import evaluate_rank
    distmat = -(q_emb @ g_emb.T)
    cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_cids, g_cids, max_rank=10)
    return float(mAP * 100), float(cmc[0] * 100)

def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)

def eff_rank_fn(X):
    Xc = X - X.mean(0, keepdims=True)
    if Xc.shape[0] < 2: return 1.0
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    p = s**2 / (s**2).sum() + 1e-12
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))

def compute_geometry(embs, pids):
    embs = l2norm(embs.astype(np.float32))
    unique_p = np.unique(pids)
    cents = np.vstack([embs[pids==p].mean(0) for p in unique_p])
    cents = l2norm(cents)
    mu_hat = embs.mean(0); mu_hat /= np.linalg.norm(mu_hat) + 1e-12
    sw = np.vstack([embs[pids==p] - cents[i] for i,p in enumerate(unique_p)])
    grand = cents.mean(0, keepdims=True)
    sb = cents - grand
    tr_sb = float(np.sum(sb**2))
    tr_sw = float(np.sum(sw**2))
    _, s, Vt = np.linalg.svd(embs - embs.mean(0), full_matrices=False)
    total_var = float((s**2).sum())
    retained = {str(k): float((s[:min(k,len(s))]**2).sum() / total_var) for k in [32,64,128,256]}
    gram = np.clip(cents @ cents.T, -1, 1)
    rows, cols = np.tril_indices(len(cents), k=-1)
    return {
        'mu_norm':       float(np.linalg.norm(embs.mean(0))),
        'std_cone':      float(np.std(embs @ mu_hat)),
        'eff_rank_SB':   eff_rank_fn(cents),
        'eff_rank_SW':   eff_rank_fn(sw),
        'tr_SB':         tr_sb,
        'tr_SW':         tr_sw,
        'fisher_F':      float(tr_sb / (tr_sw + 1e-12)),
        'retained_var':  retained,
        'med_pair_angle': float(np.median(np.degrees(np.arccos(gram[rows, cols])))),
        'cap_radius_deg': float(np.mean([
            np.degrees(np.arccos(np.clip(embs[pids==p] @ cents[i], -1,1))).max()
            for i,p in enumerate(unique_p)
        ])),
    }

# ── Load MSMT17 ────────────────────────────────────────────────────────────
print("Loading MSMT17 train...", flush=True)
msmt_ds  = get_dataset('msmt17', MSMT_ROOT)
all_train = msmt_ds.train

print("Loading Market-1501 test...", flush=True)
mkt_ds = get_dataset('market1501', MKT_ROOT)
mkt_gallery = [s for s in mkt_ds.gallery if s.person_id != -1]
mkt_query   = mkt_ds.query

# ── Load existing results (resume) ────────────────────────────────────────
result_path = RESULTS / 'e16_training_results.json'
all_results = {}
if result_path.exists():
    with open(result_path) as f: all_results = json.load(f)
    print(f"Resuming: {list(all_results.keys())} already done.", flush=True)

# ── Train each arm ─────────────────────────────────────────────────────────
for arm in ['D1','D2','D3','D4','D5','D6','C1','C2']:
    if arm in all_results:
        print(f"Arm {arm}: done, skipping."); continue

    print(f"\n{'='*60}\nArm {arm}  cameras={ARM_CAMERAS[arm]}", flush=True)

    seed_everything(SEED)
    rng = np.random.default_rng(SEED + sum(ord(c) for c in arm))

    cam_set = set(ARM_CAMERAS[arm])
    arm_samples = [s for s in all_train if s.camera_id in cam_set]

    if arm == 'C2':
        uniq = sorted(set(s.person_id for s in arm_samples))
        half_ids = set(rng.choice(uniq, len(uniq)//2, replace=False).tolist())
        arm_samples = [s for s in arm_samples if s.person_id in half_ids]

    arm_samples = subsample_stratified(arm_samples, IMAGE_BUDGET,
                                       SEED + sum(ord(c) for c in arm))
    final_ids = sorted(set(s.person_id for s in arm_samples))
    n_ids = len(final_ids)
    print(f"  {len(arm_samples)} imgs, {n_ids} IDs, cams={sorted(set(s.camera_id for s in arm_samples))}", flush=True)

    ckpt_path = OUT_DIR / f'e16_{arm}_seed{SEED}_final.pth'
    model = BagOfTricksModel(num_classes=n_ids, last_stride=1, backbone='resnet50').to(device)

    if ckpt_path.exists():
        print(f"  Checkpoint found, skipping training.", flush=True)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        pid_to_label = {p: i for i, p in enumerate(final_ids)}
        train_tf = T.Compose([
            T.Resize((256, 128)), T.RandomHorizontalFlip(0.5), T.Pad(10),
            T.RandomCrop((256, 128)), T.ToTensor(),
            T.Normalize([0.485,.456,.406],[.229,.224,.225]),
            # no REA
        ])
        train_ds = ReIDDataset(arm_samples, train_tf, pid_to_label=pid_to_label)
        sampler  = RandomIdentitySampler(arm_samples, BATCH_SIZE, NUM_INSTANCES)
        loader   = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=False, drop_last=True,
                              persistent_workers=False)

        optim    = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=5e-4)
        sched    = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda ep: (ep+1)/WARMUP if ep < WARMUP else
                              0.1**len([m for m in [40,70] if m<=ep]))
        ce_fn    = CrossEntropyLabelSmooth(n_ids, LABEL_SMOOTH).to(device)
        tri_fn   = TripletLoss(MARGIN).to(device)

        logger, _ = setup_logging(LOG_DIR, f"e16_{arm}")
        t0 = time.time()

        for epoch in range(1, EPOCHS+1):
            model.train()
            total_ce = total_tri = n_batch = 0
            for imgs, labels, _ in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, feats = model(imgs)
                loss_ce  = ce_fn(logits, labels)
                loss_tri = tri_fn(feats, labels)
                loss = loss_ce + loss_tri
                optim.zero_grad(); loss.backward(); optim.step()
                total_ce  += loss_ce.item()
                total_tri += loss_tri.item()
                n_batch   += 1
            sched.step()
            if epoch % 10 == 0 or epoch == EPOCHS:
                elapsed = (time.time()-t0)/60
                logger.info(f"Ep {epoch} CE={total_ce/n_batch:.4f} Tri={total_tri/n_batch:.4f} "
                            f"LR={optim.param_groups[0]['lr']:.6f} t={elapsed:.1f}m")
                print(f"  ep{epoch} CE={total_ce/n_batch:.4f} t={elapsed:.1f}m", flush=True)

        torch.save(model.state_dict(), ckpt_path)

    print("  Evaluating on Market-1501...", flush=True)
    q_emb, q_pids, q_cids = extract_embeddings(model, mkt_query)
    g_emb, g_pids, g_cids = extract_embeddings(model, mkt_gallery)
    map_score, r1_score   = compute_map_r1(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
    geom = compute_geometry(g_emb, g_pids)
    print(f"  mAP={map_score:.2f}%  R1={r1_score:.2f}%  ||mu||={geom['mu_norm']:.4f}", flush=True)

    all_results[arm] = {
        'arm': arm, 'cameras': ARM_CAMERAS[arm],
        'n_train_images': len(arm_samples), 'n_ids': n_ids,
        'market_map': map_score, 'market_r1': r1_score,
        'geometry_market': geom, 'ckpt': str(ckpt_path),
    }
    with open(result_path, 'w') as f: json.dump(all_results, f, indent=2)
    print(f"  Saved → {result_path}", flush=True)

    del model; torch.cuda.empty_cache()

print("\n=== E16 Complete ===")
for arm in ['D1','D2','D3','D4','D5','D6','C1','C2']:
    r = all_results.get(arm)
    if not r: continue
    g = r['geometry_market']
    print(f"  {arm}: mAP={r['market_map']:.2f}%  ||mu||={g['mu_norm']:.4f}  "
          f"eff_SB={g['eff_rank_SB']:.1f}  eff_SW={g['eff_rank_SW']:.1f}")
