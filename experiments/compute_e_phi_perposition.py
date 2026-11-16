"""
E-phi: per-position-shrunk variant, testing the position-sharing assumption.

Spec: Channel 4 rev 637 §4 ("fit shared, and per-position with heavy
shrinkage, compare held-out fit") / rev 649 §4a (run per-position after the
full-pool data-explanation check).

Decomposition: Phi_c = Phi_shared + Delta_c, so
    y_i = Sum_c Phi_shared(x_i,c) + Sum_c Delta_c(x_i,c)
        = Phi_shared(Sum_c x_i,c) + Sum_c Delta_c(x_i,c)
The first term is the already-fitted shared model's prediction (free).
The second term is fit as a per-position correction via RIDGE regression
with the position-shared prediction as the regularisation anchor (ridge
shrinks Delta_c toward ZERO, which is "no deviation from the shared
model" -- Daniel's "heavy shrinkage" literally).

Per-position params (n_cells * C_L * D_out) vastly exceed N even at the
full training pool, so this is solved in the RIDGE DUAL form: the Gram
matrix of the stacked per-cell features is

    K = Sum_c X_c @ X_c^T        (N x N, cheap: sum of n_cells (N,C)@(C,N) matmuls)

and the ridge solution for every output dimension shares one matrix
inverse (K + lambda*I)^-1, applied to the residual targets. Predictions
on held-out points use the analogous cross-Gram K* = Sum_c X*_c @ X_c^T.

Sweeps lambda on a log grid, reports held-out R^2 for shared-only vs
shared+per-position, at the reduced-rank-205 output basis (same as the
other E-phi checks).

Writes /data/papers/low_dimensional_reid/results/e44_e_phi_perposition.json
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import RNG_SEED, load_backbone_and_bn, CKPT_MARKET, device
from compute_e_snap_a_cell_agreement import extract_layer_maps, TRAIN_POOL_SIZE_CELL_DIAGNOSTIC

RESULTS = Path('/data/papers/low_dimensional_reid/results')
N_RANK = 205
HOLDOUT_FRAC = 0.2
LAYERS = ['layer1', 'layer2', 'layer3']
LAMBDA_GRID = [1e-1, 1, 10, 100, 1000, 10000, 1e5]


def ols_fit(X, Y):
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def eval_fit(X, Y, W):
    Yhat = X @ W
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(r2)


def pca_basis(X_centered, rank):
    cov = (X_centered.T @ X_centered) / X_centered.shape[0]
    eigval, eigvec = np.linalg.eigh(cov)
    order = np.argsort(-eigval)
    return eigvec[:, order[:rank]]


def gram_sum_over_cells(A_maps, B_maps, chunk_cells=16):
    """A_maps: (Na, C, H, W), B_maps: (Nb, C, H, W). Returns (Na, Nb) =
    sum over cells of A_cell @ B_cell^T, computed on GPU in cell chunks."""
    Na, C, H, W = A_maps.shape
    Nb = B_maps.shape[0]
    n_cells = H * W
    A_flat = A_maps.reshape(Na, C, n_cells).transpose(2, 0, 1)  # (n_cells, Na, C)
    B_flat = B_maps.reshape(Nb, C, n_cells).transpose(2, 0, 1)  # (n_cells, Nb, C)
    K = torch.zeros(Na, Nb, device=device, dtype=torch.float64)
    for start in range(0, n_cells, chunk_cells):
        end = min(start + chunk_cells, n_cells)
        a = torch.from_numpy(np.ascontiguousarray(A_flat[start:end])).to(device).double()
        b = torch.from_numpy(np.ascontiguousarray(B_flat[start:end])).to(device).double()
        K += torch.bmm(a, b.transpose(1, 2)).sum(dim=0)
        del a, b
    return K.cpu().numpy()


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

    print("\nExtracting layer4 GAP targets...", flush=True)
    l4_maps = extract_layer_maps(backbone, all_samples, 'layer4', batch=32, mode='train', tag='layer4_target')
    gap4 = l4_maps.mean(axis=(2, 3))
    del l4_maps

    gap4_fit, gap4_ho = gap4[~holdout_mask], gap4[holdout_mask]
    gap4_mean = gap4_fit.mean(axis=0)
    gap4_basis = pca_basis(gap4_fit - gap4_mean, N_RANK)
    tgt_fit_reduced = (gap4_fit - gap4_mean) @ gap4_basis  # (N_fit, 205)
    tgt_ho_reduced = (gap4_ho - gap4_mean) @ gap4_basis

    out = {'rev': 654, 'n_rank': N_RANK, 'n_pool': pool_size,
           'n_fit': int((~holdout_mask).sum()), 'n_holdout': int(holdout_mask.sum()),
           'lambda_grid': LAMBDA_GRID, 'per_layer': {}}

    for Lname in LAYERS:
        print(f"\n=== {Lname} ===", flush=True)
        maps = extract_layer_maps(backbone, all_samples, Lname, batch=32, mode='train', tag=Lname)
        N, C, H, W = maps.shape
        n_cells = H * W
        maps_fit, maps_ho = maps[~holdout_mask], maps[holdout_mask]
        cell_sum_fit = maps_fit.reshape(len(maps_fit), C, n_cells).sum(axis=2)
        cell_sum_ho = maps_ho.reshape(len(maps_ho), C, n_cells).sum(axis=2)

        # --- shared Phi (reduced basis), baseline for this comparison ---
        patches = maps_fit.transpose(0, 2, 3, 1).reshape(-1, C)
        patch_mean = patches.mean(axis=0)
        patch_basis = pca_basis(patches - patch_mean, min(N_RANK, C))
        red_in_fit = (cell_sum_fit - n_cells * patch_mean) @ patch_basis
        red_in_ho = (cell_sum_ho - n_cells * patch_mean) @ patch_basis
        W_shared = ols_fit(red_in_fit, tgt_fit_reduced)
        r2_shared_fit = eval_fit(red_in_fit, tgt_fit_reduced, W_shared)
        r2_shared_ho = eval_fit(red_in_ho, tgt_ho_reduced, W_shared)
        print(f"  shared Phi (reduced): fit R2={r2_shared_fit:.4f}  held-out R2={r2_shared_ho:.4f}", flush=True)

        # residual targets for the per-position delta
        resid_fit_reduced = tgt_fit_reduced - red_in_fit @ W_shared
        resid_ho_reduced = tgt_ho_reduced - red_in_ho @ W_shared

        # --- per-position delta, dual ridge ---
        print(f"  computing Gram matrices (sum over {n_cells} cells)...", flush=True)
        K_fit = gram_sum_over_cells(maps_fit, maps_fit)      # (N_fit, N_fit)
        K_ho_fit = gram_sum_over_cells(maps_ho, maps_fit)    # (N_ho, N_fit)
        n_fit = K_fit.shape[0]

        best_lambda, best_r2 = None, -1e9
        lambda_results = {}
        for lam in LAMBDA_GRID:
            alpha = np.linalg.solve(K_fit + lam * np.eye(n_fit), resid_fit_reduced)
            delta_pred_fit = K_fit @ alpha
            delta_pred_ho = K_ho_fit @ alpha
            combined_pred_ho = (red_in_ho @ W_shared) + delta_pred_ho
            r2_combined_ho = eval_fit_from_pred(tgt_ho_reduced, combined_pred_ho)
            combined_pred_fit = (red_in_fit @ W_shared) + delta_pred_fit
            r2_combined_fit = eval_fit_from_pred(tgt_fit_reduced, combined_pred_fit)
            lambda_results[str(lam)] = {'fit_R2': r2_combined_fit, 'holdout_R2': r2_combined_ho}
            print(f"    lambda={lam:<8g}  combined fit R2={r2_combined_fit:.4f}  "
                  f"held-out R2={r2_combined_ho:.4f}", flush=True)
            if r2_combined_ho > best_r2:
                best_r2, best_lambda = r2_combined_ho, lam

        improvement = best_r2 - r2_shared_ho
        print(f"  BEST: lambda={best_lambda}  combined held-out R2={best_r2:.4f}  "
              f"(shared-only={r2_shared_ho:.4f}, improvement={improvement:+.4f})", flush=True)
        verdict = ('position-sharing assumption REJECTED (per-position helps)' if improvement > 0.02
                   else 'position-sharing assumption NOT rejected (per-position does not help materially)')
        print(f"  verdict: {verdict}", flush=True)

        out['per_layer'][Lname] = {
            'n_cells': int(n_cells), 'n_channels': int(C),
            'shared_only': {'fit_R2': r2_shared_fit, 'holdout_R2': r2_shared_ho},
            'per_lambda': lambda_results,
            'best_lambda': best_lambda, 'best_combined_holdout_R2': best_r2,
            'improvement_over_shared': improvement,
            'verdict': verdict,
        }
        del maps, maps_fit, maps_ho, K_fit, K_ho_fit

    outp = RESULTS / 'e44_e_phi_perposition.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp}", flush=True)


def eval_fit_from_pred(Y, Yhat):
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


if __name__ == '__main__':
    main()
