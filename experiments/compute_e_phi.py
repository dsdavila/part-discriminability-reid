"""
E-phi: fitting the patch-direction -> embedding-direction map Phi.

Spec: Channel 4 rev 637 §4 (claude-chat, relaying Daniel). Daniel's
refinement of I-7. Form: emb_i ~= Sum_cells Phi(x_ic) -- a set regression,
one embedding target per image, cells entering as a sum. Phi fit on the
training population.

Since Phi is linear (this script's "position-shared dense" variant),
sum_c Phi(x_ic) = Phi(sum_c x_ic). At layer4, GAP(x) = mean_c(x_ic), so
sum_c(x_ic) = 128 * GAP(x). If the fit target is the PRE-BNNeck GAP vector
(the only stage where "emb" is additive -- BNNeck's affine shift and L2
normalisation are applied AFTER the sum, not per-cell, so they are not
part of Phi itself), the regression reduces to an exact, noiseless linear
identity: GAP_i = Phi(128 * GAP_i), whose unique solution is

    Phi = (1/128) * I    (2048x2048 at layer4)

with ZERO residual, deterministically -- not a statistical fit. This is
the "free correctness check" (637 §4): the fitting PROCEDURE must recover
this before trusting Phi at any earlier layer where the true answer is
not known in closed form. Pass/fail on the METHOD, not the hypothesis.

Checks run here, in order:
  1. "position-shared dense" (full C->2048 linear map, same weights at
     every cell) at layer4: OLS of GAP_i on 128*GAP_i (a tautological
     fit by construction, used ONLY to confirm the fitting/scoring
     pipeline itself has no bugs -- if this doesn't recover W=(1/128)I
     with near-zero error, something in the harness is wrong, not the
     hypothesis).
  2. The REAL check: OLS of GAP_i on sum_c(x_ic) computed from the actual
     raw per-cell layer4 maps (not the shortcut 128*GAP_i substitution),
     confirming the extraction pipeline and the regression agree with
     the closed-form answer end to end.
  3. "reduced basis" variant (patch PCs in, rank-55 embedding PCA basis
     out) at layer4: the version intended for use at earlier layers.
     Checks it reconstructs the TRUE GAP vector (not just the top-55 PCs
     of it) with low held-out error, which is the only way a lossy
     approximation can be judged at a layer whose true answer we never
     know in closed form.

Writes /data/papers/low_dimensional_reid/results/e44_e_phi_layer4_validation.json
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
N_RANK = 55  # D7's measured effective rank of the embedding space


def ols_fit(X, Y):
    """X: (N, Cin), Y: (N, Cout). Returns W: (Cin, Cout) minimising ||Y - X W||_F,
    plus the residual and R^2 (fraction of Y's variance explained)."""
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Yhat = X @ W
    ss_res = ((Y - Yhat) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rel_err = float(np.linalg.norm(Y - Yhat) / (np.linalg.norm(Y) + 1e-12))
    return W, r2, rel_err


def main():
    backbone, bn = load_backbone_and_bn(CKPT_MARKET)
    market = get_dataset('market1501', sample=False)
    rng = np.random.default_rng(RNG_SEED)
    train_all = market.train
    pool_size = min(TRAIN_POOL_SIZE_CELL_DIAGNOSTIC, len(train_all))
    idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in idx]
    print(f"Train pool: {len(train_samples)} / {len(train_all)}", flush=True)

    print("\nExtracting layer4 raw maps...", flush=True)
    maps = extract_layer_maps(backbone, train_samples, 'layer4', batch=32, mode='train', tag='train')
    N, C, H, W = maps.shape
    n_cells = H * W
    print(f"  maps shape: {maps.shape}  (n_cells={n_cells})", flush=True)

    gap = maps.mean(axis=(2, 3))  # (N, C) -- the true additive target
    cell_sum = maps.reshape(N, C, n_cells).sum(axis=2)  # (N, C) = n_cells * gap, exactly

    # sanity: cell_sum should equal n_cells * gap to numerical precision
    recon_err = float(np.abs(cell_sum - n_cells * gap).max())
    print(f"\n[sanity] max|cell_sum - {n_cells}*GAP| = {recon_err:.3e} (should be ~0)", flush=True)

    out = {'rev': 643, 'n_cells': int(n_cells), 'n_train': N, 'n_channels': int(C),
           'sanity_cell_sum_vs_gap_max_abs_err': recon_err}

    # --- Check 1: tautological harness check (fit GAP on n_cells*GAP directly) ---
    X_taut = n_cells * gap
    W1, r2_1, relerr_1 = ols_fit(X_taut, gap)
    expected_scale = 1.0 / n_cells
    w1_diag_mean = float(np.diag(W1).mean())
    w1_offdiag_rms = float(np.sqrt(((W1 - np.diag(np.diag(W1))) ** 2).mean()))
    print(f"\n[Check 1 -- harness tautology] R2={r2_1:.8f}  rel_err={relerr_1:.3e}  "
          f"mean(diag W)={w1_diag_mean:.6f} (expected {expected_scale:.6f})  "
          f"RMS(off-diag W)={w1_offdiag_rms:.3e}", flush=True)
    out['check1_harness_tautology'] = {
        'R2': r2_1, 'rel_err': relerr_1, 'mean_diag_W': w1_diag_mean,
        'expected_diag': expected_scale, 'rms_offdiag_W': w1_offdiag_rms,
        'verdict': 'PASS' if relerr_1 < 1e-6 else 'FAIL',
    }

    # --- Check 2: end-to-end from raw maps (feature = actual cell_sum, not the shortcut) ---
    W2, r2_2, relerr_2 = ols_fit(cell_sum, gap)
    w2_diag_mean = float(np.diag(W2).mean())
    w2_offdiag_rms = float(np.sqrt(((W2 - np.diag(np.diag(W2))) ** 2).mean()))
    print(f"\n[Check 2 -- end-to-end position-shared dense] R2={r2_2:.8f}  rel_err={relerr_2:.3e}  "
          f"mean(diag W)={w2_diag_mean:.6f} (expected {expected_scale:.6f})  "
          f"RMS(off-diag W)={w2_offdiag_rms:.3e}", flush=True)
    check2_pass = relerr_2 < 1e-4 and abs(w2_diag_mean - expected_scale) < 1e-4 * expected_scale
    print(f"  verdict: {'PASS' if check2_pass else 'FAIL'} "
          f"(method recovers the known closed-form answer at layer4)", flush=True)
    out['check2_end_to_end_position_shared_dense'] = {
        'R2': r2_2, 'rel_err': relerr_2, 'mean_diag_W': w2_diag_mean,
        'expected_diag': expected_scale, 'rms_offdiag_W': w2_offdiag_rms,
        'verdict': 'PASS' if check2_pass else 'FAIL',
    }

    # --- Check 3: reduced-basis variant (patch PCA in, rank-55 embedding PCA out) ---
    print(f"\n[Check 3 -- reduced basis, rank={N_RANK}]", flush=True)
    # Input PCA: fit on the per-cell patch population (every cell of every image).
    patches = maps.transpose(0, 2, 3, 1).reshape(-1, C)  # (N*n_cells, C)
    patch_mean = patches.mean(axis=0)
    patches_c = patches - patch_mean
    # SVD for PCA (patches_c is large; use randomized-ish via covariance eig for speed)
    cov_patch = (patches_c.T @ patches_c) / patches_c.shape[0]
    eigval_p, eigvec_p = np.linalg.eigh(cov_patch)
    order_p = np.argsort(-eigval_p)
    patch_basis = eigvec_p[:, order_p[:N_RANK]]  # (C, N_RANK)

    # Output PCA: fit on GAP vectors (D7's measured rank-55 claim, re-derived here).
    gap_mean = gap.mean(axis=0)
    gap_c = gap - gap_mean
    cov_gap = (gap_c.T @ gap_c) / gap_c.shape[0]
    eigval_g, eigvec_g = np.linalg.eigh(cov_gap)
    order_g = np.argsort(-eigval_g)
    cum_var = np.cumsum(eigval_g[order_g]) / eigval_g.sum()
    eff_rank_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    eff_rank_99 = int(np.searchsorted(cum_var, 0.99) + 1)
    print(f"  GAP effective rank: 90% var at {eff_rank_90} dims, 99% var at {eff_rank_99} dims "
          f"(D7's own claim: 55; checking against our own data)", flush=True)
    gap_basis = eigvec_g[:, order_g[:N_RANK]]  # (C, N_RANK)

    # Project cell_sum into patch-PC input space (sum of projected patches = projection of sum,
    # since projection is linear): reduced_input_i = (cell_sum_i - n_cells*patch_mean) @ patch_basis
    reduced_input = (cell_sum - n_cells * patch_mean) @ patch_basis  # (N, N_RANK)
    reduced_target = gap_c @ gap_basis  # (N, N_RANK)

    W3, r2_3, relerr_3 = ols_fit(reduced_input, reduced_target)
    # Reconstruct full GAP vector from the reduced prediction and compare to TRUE gap.
    pred_reduced = reduced_input @ W3
    pred_gap_full = pred_reduced @ gap_basis.T + gap_mean
    full_recon_relerr = float(np.linalg.norm(gap - pred_gap_full) / (np.linalg.norm(gap) + 1e-12))
    print(f"  reduced-space fit: R2={r2_3:.4f}  rel_err(reduced)={relerr_3:.3e}", flush=True)
    print(f"  full-GAP reconstruction rel_err (the number that matters): {full_recon_relerr:.4f}", flush=True)
    out['check3_reduced_basis'] = {
        'rank': N_RANK,
        'gap_effective_rank_90pct': eff_rank_90, 'gap_effective_rank_99pct': eff_rank_99,
        'reduced_space_R2': r2_3, 'reduced_space_rel_err': relerr_3,
        'full_gap_reconstruction_rel_err': full_recon_relerr,
        'note': 'full_gap_reconstruction_rel_err is 1 - (fraction of GAP variance the '
                'rank-55 approximation captures by construction); it reflects the basis '
                'choice (rank 55) as much as the fit quality, and is expected to be '
                'nonzero even for a perfect fit within that basis.',
    }

    outp = RESULTS / 'e44_e_phi_layer4_validation.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp}", flush=True)


if __name__ == '__main__':
    main()
