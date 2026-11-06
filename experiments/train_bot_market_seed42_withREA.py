"""
Retrain BoT-Market with REA enabled, seed=42.
Produces the seed-matched baseline for the REA ablation comparison.
Saves to /data/papers/low_dimensional_reid/pretrained/rea_ablation/bot_resnet50_market1501_seed42_withREA_final.pth
"""
import sys
sys.path.insert(0, '/workspace/reid-tk')

import os
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms as T
from torch.utils.data import DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from scripts.train_bag_of_tricks import (
    RandomErasing, BagOfTricksModel, TripletLoss,
    CrossEntropyLabelSmooth, RandomIdentitySampler,
    setup_logging, ReIDDataset
)
from reid_tk.datasets import get_dataset

DATASET_NAME  = 'market1501'
DATASET_ROOT  = Path('/data/datasets/reid/Market-1501-v15.09.15')
OUTPUT_DIR    = Path('/data/papers/low_dimensional_reid/pretrained/rea_ablation')
LOG_DIR       = Path('/data/papers/low_dimensional_reid/logs')
EPOCHS        = 120
BATCH_SIZE    = 128
NUM_INSTANCES = 4
NUM_WORKERS   = 0
BASE_LR       = 0.00035
MARGIN        = 0.3
LABEL_SMOOTH  = 0.1
RUN_NAME      = 'bot_resnet50_market1501_seed42_withREA'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger, log_file = setup_logging(LOG_DIR, RUN_NAME)
logger.info(f"Log file: {log_file}")
logger.info(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Seed: {SEED}, RandomErasing: ENABLED (p=0.5)")
device = 'cuda'

dataset = get_dataset(DATASET_NAME, DATASET_ROOT)
unique_pids = sorted(set(s.person_id for s in dataset.train))
num_classes = len(unique_pids)
pid_to_label = {pid: label for label, pid in enumerate(unique_pids)}
logger.info(f"\nLoading {DATASET_NAME} dataset...")
logger.info(f"Training on {len(dataset.train)} images, {num_classes} identities")

transform_train = T.Compose([
    T.Resize((256, 128)),
    T.RandomHorizontalFlip(p=0.5),
    T.Pad(10),
    T.RandomCrop((256, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    RandomErasing(probability=0.5),   # REA ENABLED
])

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        from PIL import Image
        s = self.samples[idx]
        img = Image.open(s.image_path).convert('RGB')
        return self.transform(img), s.person_id

sampler = RandomIdentitySampler(dataset.train, batch_size=BATCH_SIZE, num_instances=NUM_INSTANCES)
train_ds = ReIDDataset(dataset.train, transform_train, pid_to_label=pid_to_label)
loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                      num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

logger.info(f"\nBuilding Bag of Tricks model (backbone=resnet50)...")
model = BagOfTricksModel(num_classes=num_classes, backbone='resnet50').to(device)

optimizer  = torch.optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=5e-4)
scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)
ce_loss    = CrossEntropyLabelSmooth(num_classes, epsilon=LABEL_SMOOTH).to(device)
triplet    = TripletLoss(margin=MARGIN).to(device)

# Warmup
WARMUP = 10
base_lr = BASE_LR

logger.info(f"\nTraining for {EPOCHS} epochs (starting from 0)...")
logger.info("=" * 80)

for epoch in range(1, EPOCHS + 1):
    model.train()
    if epoch <= WARMUP:
        lr = base_lr * epoch / WARMUP
        for pg in optimizer.param_groups:
            pg['lr'] = lr
    total_ce = total_tri = 0.0
    n_batch = 0
    for imgs, labels, _ in tqdm(loader, leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)
        logits, feats = model(imgs)
        loss_ce  = ce_loss(logits, labels)
        loss_tri = triplet(feats, labels)
        loss     = loss_ce + loss_tri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_ce  += loss_ce.item()
        total_tri += loss_tri.item()
        n_batch   += 1
    if epoch > WARMUP:
        scheduler.step()
    cur_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {epoch}/{EPOCHS} - CE: {total_ce/n_batch:.4f}, "
                f"Triplet: {total_tri/n_batch:.4f}, LR: {cur_lr:.6f}")
    if epoch % 20 == 0:
        ckpt_path = OUTPUT_DIR / f'bot_resnet50_market1501_seed42_withREA_epoch{epoch}.pth'
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"  Saved checkpoint: {ckpt_path}")

final_path = OUTPUT_DIR / 'bot_resnet50_market1501_seed42_withREA_final.pth'
torch.save(model.state_dict(), final_path)
config = {'dataset': DATASET_NAME, 'seed': SEED, 'random_erasing': True,
          'num_classes': num_classes, 'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
          'base_lr': BASE_LR, 'margin': MARGIN, 'label_smooth': LABEL_SMOOTH}
json.dump(config, open(OUTPUT_DIR / 'bot_market1501_seed42_withREA_config.json', 'w'), indent=2)
logger.info(f"\nTraining complete! Final model saved to: {final_path}")
