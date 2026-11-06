"""E16b — Replication seeds for D1, D3, D6, C1. Seeds 1337 and 2024."""
import sys; sys.path.insert(0, '/workspace/reid-tk')
import numpy as np, json, torch, torch.nn.functional as F, time
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader

from scripts.train_bag_of_tricks import (
    BagOfTricksModel, TripletLoss, CrossEntropyLabelSmooth,
    ReIDDataset, setup_logging,
)
from reid_tk.datasets import get_dataset
from reid_tk.samplers import RandomIdentitySampler

OUT_DIR   = Path('/data/papers/low_dimensional_reid/pretrained/e16b')
LOG_DIR   = Path('/data/papers/low_dimensional_reid/logs')
MKT_ROOT  = Path('/data/datasets/reid/Market-1501-v15.09.15')
MSMT_ROOT = Path('/data/datasets/reid/MSMT17_V1')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
EPOCHS, BATCH_SIZE, NUM_INSTANCES = 120, 128, 4
BASE_LR, MARGIN, LABEL_SMOOTH, WARMUP, NUM_WORKERS, IMAGE_BUDGET = 0.00035, 0.3, 0.1, 10, 0, 7914
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for d in (OUT_DIR, LOG_DIR, RESULTS): d.mkdir(parents=True, exist_ok=True)

ARMS = ['D1', 'D3', 'D6', 'C1']
SEEDS = [1337, 2024]
ARM_CAMERAS = {
    'D1': [1, 13],
    'D3': [1, 4, 7, 9, 13, 15],
    'D6': list(range(1, 16)),
    'C1': [5, 6, 8, 10, 11, 14],
}

def seed_everything(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def subsample_stratified(samples, budget, seed):
    if len(samples) <= budget: return list(samples)
    rng = np.random.default_rng(seed)
    cam_to_idx = defaultdict(list)
    for i, s in enumerate(samples): cam_to_idx[s.camera_id].append(i)
    total = len(samples); chosen = []
    for cam, idxs in cam_to_idx.items():
        n_take = max(1, round(budget * len(idxs) / total))
        chosen.extend(rng.choice(idxs, min(n_take, len(idxs)), replace=False).tolist())
    chosen = list(set(chosen))
    if len(chosen) > budget:
        chosen = rng.choice(chosen, budget, replace=False).tolist()
    elif len(chosen) < budget:
        rest = [i for i in range(total) if i not in set(chosen)]
        chosen.extend(rng.choice(rest, min(budget - len(chosen), len(rest)), replace=False).tolist())
    return [samples[i] for i in sorted(set(chosen))]

TRANSFORM_EVAL = T.Compose([T.Resize((256, 128)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class EvalDS(torch.utils.data.Dataset):
    def __init__(self, s): self.s = s
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        return TRANSFORM_EVAL(Image.open(self.s[i].image_path).convert('RGB')), \
               self.s[i].person_id, self.s[i].camera_id

def extract(model, samps):
    dl = DataLoader(EvalDS(samps), 128, num_workers=NUM_WORKERS, pin_memory=True)
    embs, pids, cids = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, ps, cs in dl:
            f = F.normalize(model(imgs.to(device)), dim=1)
            embs.append(f.cpu().numpy()); pids.extend(ps.tolist()); cids.extend(cs.tolist())
    return np.vstack(embs), np.array(pids), np.array(cids)

def map_r1(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids):
    from torchreid.reid.metrics import evaluate_rank
    cmc, mAP = evaluate_rank(-(q_emb @ g_emb.T), q_pids, g_pids, q_cids, g_cids, max_rank=10)
    return float(mAP * 100), float(cmc[0] * 100)

def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); return X / np.maximum(n, 1e-12)

def eff_rank(X):
    Xc = X - X.mean(0, keepdims=True)
    if Xc.shape[0] < 2: return 1.0
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    p = s**2 / (s**2).sum() + 1e-12; p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))

def geom(embs, pids):
    embs = l2norm(embs.astype(np.float32))
    uniq = np.unique(pids)
    cents = l2norm(np.vstack([embs[pids==p].mean(0) for p in uniq]))
    mu_hat = embs.mean(0); mu_hat /= np.linalg.norm(mu_hat) + 1e-12
    sw = np.vstack([embs[pids==p] - cents[i] for i,p in enumerate(uniq)])
    sb = cents - cents.mean(0, keepdims=True)
    return {'mu_norm': float(np.linalg.norm(embs.mean(0))),
            'eff_rank_SB': eff_rank(cents),
            'eff_rank_SW': eff_rank(sw),
            'tr_SB': float((sb**2).sum()),
            'tr_SW': float((sw**2).sum()),
            'fisher_F': float((sb**2).sum() / ((sw**2).sum() + 1e-12))}

print("Loading MSMT17 train...", flush=True); all_train = get_dataset('msmt17', MSMT_ROOT).train
print("Loading Market-1501 test...", flush=True)
mkt_ds = get_dataset('market1501', MKT_ROOT)
mkt_gal = [s for s in mkt_ds.gallery if s.person_id != -1]; mkt_q = mkt_ds.query

result_path = RESULTS / 'e16b_training_results.json'
all_results = {}
if result_path.exists():
    with open(result_path) as f: all_results = json.load(f)
    print(f"Resuming: {list(all_results.keys())} done.", flush=True)

for seed in SEEDS:
    for arm in ARMS:
        key = f'{arm}_seed{seed}'
        if key in all_results:
            print(f"{key}: done, skipping."); continue
        print(f"\n{'='*60}\n{key}  cameras={ARM_CAMERAS[arm]}", flush=True)
        seed_everything(seed)
        subsample_seed = seed + sum(ord(c) for c in arm)
        cam_set = set(ARM_CAMERAS[arm])
        samples = subsample_stratified([s for s in all_train if s.camera_id in cam_set],
                                        IMAGE_BUDGET, subsample_seed)
        final_ids = sorted(set(s.person_id for s in samples)); n_ids = len(final_ids)
        print(f"  {len(samples)} imgs, {n_ids} IDs", flush=True)

        ckpt = OUT_DIR / f'e16b_{arm}_seed{seed}_final.pth'
        model = BagOfTricksModel(num_classes=n_ids, last_stride=1, backbone='resnet50').to(device)
        if ckpt.exists():
            print(f"  ckpt found, skipping train.", flush=True)
            model.load_state_dict(torch.load(ckpt, map_location=device))
        else:
            pid_to_label = {p: i for i, p in enumerate(final_ids)}
            tf = T.Compose([T.Resize((256,128)), T.RandomHorizontalFlip(0.5), T.Pad(10),
                            T.RandomCrop((256,128)), T.ToTensor(),
                            T.Normalize([0.485,.456,.406],[.229,.224,.225])])
            ds = ReIDDataset(samples, tf, pid_to_label=pid_to_label)
            loader = DataLoader(ds, BATCH_SIZE,
                                sampler=RandomIdentitySampler(samples, BATCH_SIZE, NUM_INSTANCES),
                                num_workers=NUM_WORKERS, pin_memory=False, drop_last=True)
            optim = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=5e-4)
            sched = torch.optim.lr_scheduler.LambdaLR(
                optim, lambda ep: (ep+1)/WARMUP if ep < WARMUP
                                  else 0.1**len([m for m in [40,70] if m<=ep]))
            ce_fn = CrossEntropyLabelSmooth(n_ids, LABEL_SMOOTH).to(device)
            tri_fn = TripletLoss(MARGIN).to(device)
            logger, _ = setup_logging(LOG_DIR, f"e16b_{key}")
            t0 = time.time()
            for epoch in range(1, EPOCHS+1):
                model.train(); tc = tt = nb = 0
                for imgs, labels, _ in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits, feats = model(imgs)
                    loss_ce = ce_fn(logits, labels); loss_tri = tri_fn(feats, labels)
                    loss = loss_ce + loss_tri
                    optim.zero_grad(); loss.backward(); optim.step()
                    tc += loss_ce.item(); tt += loss_tri.item(); nb += 1
                sched.step()
                if epoch % 10 == 0 or epoch == EPOCHS:
                    print(f"  ep{epoch} t={(time.time()-t0)/60:.1f}m", flush=True)
            torch.save(model.state_dict(), ckpt)

        print("  Eval on Market-1501...", flush=True)
        q_emb, q_pids, q_cids = extract(model, mkt_q)
        g_emb, g_pids, g_cids = extract(model, mkt_gal)
        m, r = map_r1(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids)
        g_ = geom(g_emb, g_pids)
        print(f"  mAP={m:.2f}%  R1={r:.2f}%  fisher_F={g_['fisher_F']:.5f}", flush=True)
        all_results[key] = {'arm': arm, 'seed': seed, 'n_train_images': len(samples),
                            'n_ids': n_ids, 'market_map': m, 'market_r1': r,
                            'geometry_market': g_, 'ckpt': str(ckpt)}
        with open(result_path, 'w') as f: json.dump(all_results, f, indent=2)
        print(f"  Saved → {result_path}", flush=True)
        del model; torch.cuda.empty_cache()

print("\n=== E16b Complete ===")
