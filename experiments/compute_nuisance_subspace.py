"""
Nuisance-subspace alignment + magnitude (Daniel via claude-chat, Channel 4
rev 676 section 1 / rev 677 item 2).

Daniel's correction (676 section 0): identity-metric training does not
destroy viewpoint/pose variation, it compresses it to just below the
triplet margin and leaves it alone (zero gradient once d(a,p)+m < d(a,n)).
So OOD within-identity variation might be the SAME kind of nuisance
training learned to tolerate, merely too large (a bounding problem,
fixable) -- or a DIFFERENT kind the model never bounded at all (far more
serious). This measures which.

Per layer, per OOD arm (B'-sub, C):
  1. Training nuisance subspace: pool within-identity deviations (sample
     minus that identity's mean) across all training identities with >=2
     images; PCA (SVD); keep top-r right singular vectors as an orthonormal
     basis, r swept.
  2. OOD within-identity deviations, same construction, over the combined
     query+gallery pool (identities with >=2 images there).
  3. Alignment: fraction of OOD deviation ENERGY captured by the top-r
     training nuisance basis, vs chance level r/d (d = ambient dim at that
     layer).
  4. Magnitude: ratio of mean OOD deviation energy to mean training
     deviation energy.

Writes /data/papers/low_dimensional_reid/results/e44_nuisance_subspace.json
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, '/workspace/reid-tk')
from reid_tk.datasets import get_dataset

sys.path.insert(0, str(Path(__file__).parent))
from compute_e_snap_a import (
    RNG_SEED, N_QUERY, N_GALLERY, load_backbone_and_bn,
    CKPT_MARKET, CKPT_MSMT, load_saved_bprime_sub_indices,
    extract_gap_vectors_and_terminal,
)

RESULTS = Path('/data/papers/low_dimensional_reid/results')
LAYER_NAMES = ['layer1', 'layer2', 'layer3', 'layer4']
LAYER_DIMS = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
R_SWEEP = [5, 10, 20, 50, 100, 200]
MIN_IMGS = 2


def within_identity_deviations(X, pids):
    """X: (N, C). pids: (N,). Returns (M, C) pooled deviation vectors over
    all identities with >= MIN_IMGS images (sample minus that identity's own
    mean), and the per-deviation squared norms."""
    id_to_idx = defaultdict(list)
    for i, p in enumerate(pids):
        id_to_idx[int(p)].append(i)
    devs = []
    for pid, idxs in id_to_idx.items():
        if len(idxs) < MIN_IMGS:
            continue
        Xk = X[idxs]
        mu = Xk.mean(axis=0)
        devs.append(Xk - mu)
    if not devs:
        return np.zeros((0, X.shape[1]), dtype=X.dtype)
    return np.concatenate(devs, axis=0)


def run_arm(arm_name, model_ckpt, model_name, train_dataset_name, eval_dataset_name,
            q_idx=None, g_idx=None, batch=32, train_pool_size=3000):
    print(f"\n=== nuisance subspace, arm {arm_name} (model={model_name}) ===", flush=True)
    backbone, bn = load_backbone_and_bn(model_ckpt)
    train_ds_obj = get_dataset(train_dataset_name, sample=False)
    eval_ds_obj = get_dataset(eval_dataset_name, sample=False)

    rng = np.random.default_rng(RNG_SEED)
    train_all = train_ds_obj.train
    pool_size = min(train_pool_size, len(train_all))
    tr_idx = rng.choice(len(train_all), size=pool_size, replace=False)
    train_samples = [train_all[int(i)] for i in tr_idx]
    train_pids = np.array([int(s.person_id) for s in train_samples])

    q_all = [s for s in eval_ds_obj.query if not s.is_junk]
    g_all = [s for s in eval_ds_obj.gallery if not s.is_junk]
    if q_idx is None:
        q_idx = rng.choice(len(q_all), size=min(N_QUERY, len(q_all)), replace=False)
        g_idx = rng.choice(len(g_all), size=min(N_GALLERY, len(g_all)), replace=False)
    query_samples = [q_all[int(i)] for i in q_idx]
    gallery_samples = [g_all[int(i)] for i in g_idx]
    print(f"  train pool: {len(train_samples)}  query: {len(query_samples)}  gallery: {len(gallery_samples)}", flush=True)

    tr_gap, _ = extract_gap_vectors_and_terminal(backbone, bn, train_samples, batch=batch,
                                                   mode='train', tag='train')
    q_gap, _, q_pids, q_cams = extract_gap_vectors_and_terminal(backbone, bn, query_samples,
                                                                  batch=batch, mode='test', tag='query')
    g_gap, _, g_pids, g_cams = extract_gap_vectors_and_terminal(backbone, bn, gallery_samples,
                                                                  batch=batch, mode='test', tag='gallery')

    out = {'rev': 676, 'arm': arm_name, 'model': model_name, 'r_sweep': R_SWEEP, 'per_layer': {}}

    for Lname in LAYER_NAMES:
        d = LAYER_DIMS[Lname]
        tr_dev = within_identity_deviations(tr_gap[Lname], train_pids)
        # OOD within-identity deviations: pool query+gallery by pid (same OOD identity population)
        ood_X = np.concatenate([q_gap[Lname], g_gap[Lname]], axis=0)
        ood_pids = np.concatenate([q_pids, g_pids], axis=0)
        ood_dev = within_identity_deviations(ood_X, ood_pids)

        n_tr_dev = tr_dev.shape[0]
        n_ood_dev = ood_dev.shape[0]
        if n_tr_dev < 2 or n_ood_dev < 2:
            print(f"  {Lname}: insufficient deviations (train={n_tr_dev}, ood={n_ood_dev}), skipping", flush=True)
            continue

        # SVD of training deviations -> right singular vectors = PCA basis
        # (pooled deviations are already identity-mean-centered, so no extra centering needed)
        U, S, Vt = np.linalg.svd(tr_dev, full_matrices=False)  # Vt: (min(N,C), C)
        scatter_train = float((tr_dev ** 2).sum(axis=1).mean())
        scatter_ood = float((ood_dev ** 2).sum(axis=1).mean())
        magnitude_ratio = scatter_ood / scatter_train if scatter_train > 0 else float('nan')

        ood_energy_total = float((ood_dev ** 2).sum())
        per_r = {}
        for r in R_SWEEP:
            r_eff = min(r, Vt.shape[0])
            basis = Vt[:r_eff]  # (r_eff, C)
            proj = ood_dev @ basis.T  # (N_ood_dev, r_eff)
            energy_in = float((proj ** 2).sum())
            alignment = energy_in / ood_energy_total if ood_energy_total > 0 else float('nan')
            chance = r_eff / d
            per_r[str(r)] = {'r_effective': r_eff, 'alignment': alignment, 'chance': chance,
                              'ratio_to_chance': alignment / chance if chance > 0 else float('nan')}
            print(f"    {Lname} r={r:<4d}: alignment={100*alignment:.2f}%  chance={100*chance:.2f}%  "
                  f"ratio={alignment/chance if chance>0 else float('nan'):.2f}", flush=True)

        print(f"    {Lname} magnitude: scatter_ood/scatter_train = {magnitude_ratio:.3f}  "
              f"(n_train_dev={n_tr_dev}, n_ood_dev={n_ood_dev})", flush=True)

        out['per_layer'][Lname] = {
            'ambient_dim': d, 'n_train_dev': n_tr_dev, 'n_ood_dev': n_ood_dev,
            'scatter_train': scatter_train, 'scatter_ood': scatter_ood,
            'magnitude_ratio': magnitude_ratio,
            'alignment_by_r': per_r,
        }

    outp = RESULTS / f'e44_nuisance_subspace_{arm_name}.json'
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n  saved {outp}", flush=True)
    return out


def main():
    q_idx, g_idx = load_saved_bprime_sub_indices()
    run_arm('B_prime_sub', CKPT_MARKET, model_name='BoT-Market',
             train_dataset_name='market1501', eval_dataset_name='msmt17',
             q_idx=q_idx, g_idx=g_idx, batch=32)
    run_arm('C', CKPT_MSMT, model_name='BoT-MSMT',
            train_dataset_name='msmt17', eval_dataset_name='market1501',
            q_idx=None, g_idx=None, batch=32)


if __name__ == '__main__':
    main()
