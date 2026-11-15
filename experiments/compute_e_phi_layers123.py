"""
E-phi, layers 1-3: fit Phi where ground truth is NOT known in closed form.

Spec: Channel 4 rev 637 §4 item 4 / rev 645-646 order item: run after the
layer4 analytic validation passes. Unlike layer4 (where Phi = (1/128)*I is
a deterministic identity), at layers 1-3 the target (the terminal
pre-BNNeck GAP(layer4) vector) is reached through further nonlinear
convolutional layers, so a LINEAR position-shared Phi_L can only
approximate it. This is now a genuine statistical fit, scored on held-out
data (train/held-out split within the training pool), not a tautology.

Two variants per layer, both position-shared (same Phi applied at every
cell), per Daniel's spec:
  - full-rank dense: Phi_L: R^{C_L} -> R^{2048} (direct raw channels in,
    raw GAP(layer4) target out -- feasible at every layer per the
    constraints/param table in Channel 4 rev 637 §4).
  - reduced basis: patch PCs (of layer L's patch population) in, rank-205
    GAP(layer4) PCA basis out (corrected rank, Channel 4 rev 645-646).

Position-SPECIFIC dense (one Phi per spatial position, heavy shrinkage) is
flagged as a follow-up, not run here -- it requires per-position design
matrices and a ridge penalty grid; starting with the cheaper shared fit
first, per the assumption-testing logic in 637 §4 ("fit shared, and
per-position with heavy shrinkage, compare held-out fit. Report both.").

Writes /data/papers/low_dimensional_reid/results/e44_e_phi_layers123.json
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import RNG_SEED, load_backbone_and_bn, CKPT_MARKET
from compute_e_snap_a_cell_agreement import extract_layer_maps, TRAIN_POOL_SIZE_CELL_DIAGNOSTIC

RESULTS = Path('/data/papers/low_dimensional_reid/results')
N_RANK = 205
HOLDOUT_FRAC = 0.2
LAYERS = ['layer1', 'layer2', 'layer3']


def ols_fit(X, Y):
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def eval_fit(X, Y, W):
    Yhat = X @ W
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rel_err = float(np.linalg.norm(Y - Yhat) / (np.linalg.norm(Y) + 1e-12))
    return float(r2), rel_err


def pca_basis(X_centered, rank):
    cov = (X_centered.T @ X_centered) / X_centered.shape[0]
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(-eigval)
    return eigvec[:, order[:rank]], eigval[order]


def main():
    backbone, bn = load_backbone_and_bn(CKPT_MARKET)
    market = get_dataset('market1501', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    train_all = market.train
    pool_size = min(TRAIN_POOL_SIZE_CELL_DIAGNOSTIC, len(train_all))
    idx = rng.choice(len(train_all), size=pool_size, replace=False)
    all_samples = [train_all[int(i)] for i in idx]

    n_holdout = int(round(HOLDOUT_FRAC * pool_size))
    holdout_mask = np.zeros(pool_size, dtype=bool)
    holdout_idx = rng.choice(pool_size, size=n_holdout, replace=False)
    holdout_mask[holdout_idx] = True
    print(f"Pool: {pool_size} ({pool_size - n_holdout} fit / {n_holdout} held-out)", flush=True)

    print("\nExtracting layer4 GAP targets (shared across all layer-L fits)...", flush=True)
    l4_maps = extract_layer_maps(backbone, all_samples, 'layer4', batch=32, mode='train', tag='layer4_target')
    gap4 = l4_maps.mean(axis=(2, 3))  # (N, 2048) -- the fixed target for every layer's Phi
    del l4_maps

    gap4_fit = gap4[~holdout_mask]
    gap4_ho = gap4[holdout_mask]
    gap4_mean = gap4_fit.mean(axis=0)

    out = {'rev': 646, 'n_rank': N_RANK, 'n_pool': pool_size,
           'n_fit': int((~holdout_mask).sum()), 'n_holdout': int(holdout_mask.sum()),
           'per_layer': {}}

    for Lname in LAYERS:
        print(f"\n=== {Lname} ===", flush=True)
        maps = extract_layer_maps(backbone, all_samples, Lname, batch=32, mode='train', tag=Lname)
        N, C, H, W = maps.shape
        n_cells = H * W
        cell_sum = maps.reshape(N, C, n_cells).sum(axis=2)  # (N, C)
        cs_fit, cs_ho = cell_sum[~holdout_mask], cell_sum[holdout_mask]

        # --- full-rank dense (direct raw channels -> raw GAP4 target) ---
        W_full = ols_fit(cs_fit, gap4_fit)
        r2_fit_full, relerr_fit_full = eval_fit(cs_fit, gap4_fit, W_full)
        r2_ho_full, relerr_ho_full = eval_fit(cs_ho, gap4_ho, W_full)
        print(f"  full-rank dense: fit R2={r2_fit_full:.4f} rel_err={relerr_fit_full:.4f}  |  "
              f"held-out R2={r2_ho_full:.4f} rel_err={relerr_ho_full:.4f}", flush=True)

        # --- reduced basis (patch PCs in, rank-205 GAP4 PCs out) ---
        patches = maps[~holdout_mask].transpose(0, 2, 3, 1).reshape(-1, C)
        patch_mean = patches.mean(axis=0)
        patch_basis, _ = pca_basis(patches - patch_mean, min(N_RANK, C))
        gap4_basis, gap4_eigval = pca_basis(gap4_fit - gap4_mean, N_RANK)

        red_in_fit = (cs_fit - n_cells * patch_mean) @ patch_basis
        red_in_ho = (cs_ho - n_cells * patch_mean) @ patch_basis
        red_tgt_fit = (gap4_fit - gap4_mean) @ gap4_basis
        red_tgt_ho = (gap4_ho - gap4_mean) @ gap4_basis

        W_red = ols_fit(red_in_fit, red_tgt_fit)
        r2_fit_red, relerr_fit_red = eval_fit(red_in_fit, red_tgt_fit, W_red)
        r2_ho_red, relerr_ho_red = eval_fit(red_in_ho, red_tgt_ho, W_red)

        # full-GAP reconstruction error on held-out (the number that matters end to end)
        pred_red_ho = red_in_ho @ W_red
        pred_gap4_ho = pred_red_ho @ gap4_basis.T + gap4_mean
        full_relerr_ho = float(np.linalg.norm(gap4_ho - pred_gap4_ho) / (np.linalg.norm(gap4_ho) + 1e-12))
        # also report the trivial "predict the mean" baseline for context
        baseline_relerr_ho = float(np.linalg.norm(gap4_ho - gap4_mean) / (np.linalg.norm(gap4_ho) + 1e-12))

        print(f"  reduced basis (rank={min(N_RANK,C)}/{N_RANK}): "
              f"reduced-space fit R2={r2_fit_red:.4f}  held-out R2={r2_ho_red:.4f}  "
              f"full-GAP held-out rel_err={full_relerr_ho:.4f}  "
              f"(predict-mean baseline rel_err={baseline_relerr_ho:.4f})", flush=True)

        out['per_layer'][Lname] = {
            'n_cells': int(n_cells), 'n_channels': int(C),
            'full_rank_dense': {
                'fit_R2': r2_fit_full, 'fit_rel_err': relerr_fit_full,
                'holdout_R2': r2_ho_full, 'holdout_rel_err': relerr_ho_full,
            },
            'reduced_basis': {
                'input_rank': int(min(N_RANK, C)), 'output_rank': N_RANK,
                'fit_R2_reduced_space': r2_fit_red, 'holdout_R2_reduced_space': r2_ho_red,
                'holdout_full_GAP_rel_err': full_relerr_ho,
                'predict_mean_baseline_holdout_rel_err': baseline_relerr_ho,
            },
        }
        del maps

    outp = RESULTS / 'e44_e_phi_layers123.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp}", flush=True)


if __name__ == '__main__':
    main()
