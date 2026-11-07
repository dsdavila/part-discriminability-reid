"""
Run 1: hub-pruning sweep (Ch2 rev 158 spec, as refined by rev 171/179).

Prune ood-hub images from the gallery progressively, most-hubby first,
plus the three controls:
  - random-prune (secondary control, 3 draws)
  - id-level hub-prune (whole identity pruned by number of hub images;
    strand count reported)
  - in-domain arm's mAP as a harm monitor per ruling 29

Levels: 0, 0.5, 1, 2, 5, 10% of gallery items (or identities for id-level).
Both datasets. Both id and ood arms.

Report: per arm per prune level per scheme:
  - mAP over queries with >=1 remaining positive
  - count of queries stranded (no remaining positives)
  - absolute (id - ood) gap and gap / id (ruling 26)

Rulings 29 reminder: in-domain is a harm monitor, not a convergence
target. Evaluate the ood arm's own trajectory; id arm flags harm.
"""
import json, time
import numpy as np
from pathlib import Path
import torch

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
HUBNESS   = RESULTS / 'hubness_koccurrence.json'
PRUNE_LEVELS_PCT = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
N_RANDOM_DRAWS = 3
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'MSMT': {
        'id':  ('bot_msmt_seed42_msmt.npz',  11659),
        'ood': ('msmt17_test_noREA.npz',     11659),
    },
    'Market': {
        'id':  ('market_test_noREA.npz',      3368),
        'ood': ('bot_msmt_seed42_market.npz', 3368),
    },
}


def load_qg(fname, n_q):
    d = np.load(CACHE_DIR / fname)
    emb = d['emb'].astype(np.float32)
    return {
        'Q': emb[:n_q], 'Qp': d['pids'][:n_q], 'Qc': d['cids'][:n_q],
        'G': emb[n_q:], 'Gp': d['pids'][n_q:], 'Gc': d['cids'][n_q:],
    }


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def mAP_pruned(sims, q_pids, q_cids, g_pids, g_cids, pruned_mask):
    """mAP under pruning: pruned gallery items removed from every query's
    ranked list. Returns (mAP, n_used, n_stranded) where n_stranded is
    queries with no remaining positives after same-cam-same-pid exclusion
    AND pruning.
    """
    Nq = sims.shape[0]
    ap_sum = 0.0; n_used = 0; n_stranded = 0
    for q in range(Nq):
        qp, qc = q_pids[q], q_cids[q]
        exc_same_cam = (g_pids == qp) & (g_cids == qc)
        keep = ~exc_same_cam & ~pruned_mask
        if keep.sum() == 0:
            n_stranded += 1
            continue
        sv = sims[q][keep]
        gp = g_pids[keep]
        pos = (gp == qp)
        n_pos = int(pos.sum())
        if n_pos == 0:
            n_stranded += 1
            continue
        order = np.argsort(-sv, kind='stable')
        pos_at = pos[order]
        cum = np.cumsum(pos_at)
        ranks = np.arange(1, len(order) + 1)
        prec = cum / ranks
        ap = prec[pos_at].sum() / n_pos
        ap_sum += ap
        n_used += 1
    return (ap_sum / n_used if n_used > 0 else 0.0, n_used, n_stranded)


def run_dataset(ds_name, cfg):
    print(f"\n########## {ds_name} ##########", flush=True)
    hub = json.load(open(HUBNESS))
    ood_kocc = np.array(hub[ds_name]['per_item_occ']['ood'])
    Ng = len(ood_kocc)
    hub_order = np.argsort(-ood_kocc, kind='stable')
    print(f"  Ng={Ng}, ood kocc range {ood_kocc.min()}-{ood_kocc.max()}", flush=True)

    # Load both arms, compute sims once per arm
    arms = {}
    for arm_name, (fname, n_q) in cfg.items():
        data = load_qg(fname, n_q)
        Q = l2(data['Q']); G = l2(data['G'])
        print(f"  [{arm_name}] Nq={len(Q)} Ng={len(G)}", flush=True)
        G_cuda = torch.from_numpy(G).to(_DEVICE)
        Q_cuda = torch.from_numpy(Q).to(_DEVICE)
        sims = (Q_cuda @ G_cuda.T).cpu().numpy()
        del G_cuda, Q_cuda
        torch.cuda.empty_cache()
        arms[arm_name] = {
            'Qp': data['Qp'], 'Qc': data['Qc'],
            'Gp': data['Gp'], 'Gc': data['Gc'],
            'sims': sims,
        }

    # Identity-level: rank identities by # hub images among top X% of gallery
    # at each level (we compute these on the fly inside the loop).
    g_pids = arms['id']['Gp']  # same across arms
    g_cids = arms['id']['Gc']
    unique_pids = np.unique(g_pids)
    hub_count_per_pid = {}
    # Baseline: full-gallery hub count per identity = number of items from this
    # id whose ood_kocc rank is in the top-X% (precomputed per level inside loop).
    # Hub images top-X% defined once per level.

    out = {'Ng': int(Ng), 'per_arm': {}}
    for arm_name in arms:
        out['per_arm'][arm_name] = {'hub_image': [], 'random_image': [],
                                      'id_level_hub': []}

    for level_pct in PRUNE_LEVELS_PCT:
        n_mask = int(Ng * level_pct / 100.0)
        # Hub-image prune set
        hub_pruned = np.zeros(Ng, dtype=bool)
        if n_mask > 0:
            hub_pruned[hub_order[:n_mask]] = True
        # Random-image prune: N_RANDOM_DRAWS draws
        rng = np.random.default_rng(42 + int(level_pct * 10))
        random_pruned_draws = []
        for d in range(N_RANDOM_DRAWS):
            r = np.zeros(Ng, dtype=bool)
            if n_mask > 0:
                sel = rng.choice(Ng, size=n_mask, replace=False)
                r[sel] = True
            random_pruned_draws.append(r)
        # ID-level hub prune: rank identities by their count of hub images at
        # this level (what fraction of each identity's items are hubs). Use
        # the top-X% hub image set as the "hub pool" and prune whole identities
        # that have the highest count of their own items in the pool.
        id_hub_count = {int(p): 0 for p in unique_pids}
        for i_hub in hub_order[:n_mask] if n_mask > 0 else []:
            id_hub_count[int(g_pids[i_hub])] += 1
        # Prune identities with the most hub images, until pruning them removes
        # approximately n_mask items (match level by items, not by identity count).
        pid_sorted_by_hub = sorted(unique_pids, key=lambda p: -id_hub_count[int(p)])
        id_level_pruned = np.zeros(Ng, dtype=bool)
        items_pruned = 0
        id_pruned_count = 0
        for pid in pid_sorted_by_hub:
            if items_pruned >= n_mask: break
            pid_mask = (g_pids == pid)
            id_level_pruned |= pid_mask
            items_pruned += int(pid_mask.sum())
            id_pruned_count += 1

        for arm_name in arms:
            a = arms[arm_name]
            t0 = time.time()
            # Hub-image prune
            mAP_hub, n_used_hub, n_strand_hub = mAP_pruned(
                a['sims'], a['Qp'], a['Qc'], a['Gp'], a['Gc'], hub_pruned)
            # Random-image prune (mean over draws)
            rand_mAPs, rand_used, rand_strand = [], [], []
            for r in random_pruned_draws:
                m, u, s = mAP_pruned(a['sims'], a['Qp'], a['Qc'],
                                       a['Gp'], a['Gc'], r)
                rand_mAPs.append(m); rand_used.append(u); rand_strand.append(s)
            # ID-level prune
            mAP_id, n_used_id, n_strand_id = mAP_pruned(
                a['sims'], a['Qp'], a['Qc'], a['Gp'], a['Gc'], id_level_pruned)

            print(f"  [{arm_name}] level {level_pct:>5.1f}% (n_items={n_mask}): "
                  f"hub mAP={mAP_hub*100:.3f}  "
                  f"rand mAP={np.mean(rand_mAPs)*100:.3f}±{np.std(rand_mAPs)*100:.3f}  "
                  f"id-level mAP={mAP_id*100:.3f} (id_pruned={id_pruned_count}, items={items_pruned}, strand={n_strand_id})  "
                  f"t={time.time()-t0:.1f}s", flush=True)

            out['per_arm'][arm_name]['hub_image'].append({
                'level_pct': level_pct,
                'n_items_pruned': int(n_mask),
                'mAP_pct': float(mAP_hub * 100.0),
                'n_queries_used': int(n_used_hub),
                'n_queries_stranded': int(n_strand_hub),
            })
            out['per_arm'][arm_name]['random_image'].append({
                'level_pct': level_pct,
                'n_items_pruned': int(n_mask),
                'mAP_pct_draws': [float(m * 100.0) for m in rand_mAPs],
                'mAP_pct_mean': float(np.mean(rand_mAPs) * 100.0),
                'mAP_pct_std': float(np.std(rand_mAPs) * 100.0),
                'n_queries_stranded_mean': float(np.mean(rand_strand)),
            })
            out['per_arm'][arm_name]['id_level_hub'].append({
                'level_pct': level_pct,
                'id_pruned_count': int(id_pruned_count),
                'items_pruned': int(items_pruned),
                'mAP_pct': float(mAP_id * 100.0),
                'n_queries_stranded': int(n_strand_id),
            })

    # id - ood gap per scheme per level: both absolute and gap/id
    def _gap_series(scheme):
        rows = []
        for i, lvl in enumerate(PRUNE_LEVELS_PCT):
            if scheme == 'random_image':
                id_m = out['per_arm']['id'][scheme][i]['mAP_pct_mean']
                ood_m = out['per_arm']['ood'][scheme][i]['mAP_pct_mean']
            else:
                id_m = out['per_arm']['id'][scheme][i]['mAP_pct']
                ood_m = out['per_arm']['ood'][scheme][i]['mAP_pct']
            gap = id_m - ood_m
            frac = gap / id_m if id_m > 0 else 0.0
            rows.append({
                'level_pct': lvl, 'id_mAP': id_m, 'ood_mAP': ood_m,
                'abs_gap': gap, 'gap_over_id': frac,
            })
        return rows

    out['gap_series'] = {s: _gap_series(s) for s in ['hub_image', 'random_image', 'id_level_hub']}
    for s in ['hub_image', 'random_image', 'id_level_hub']:
        print(f"\n  [{ds_name}] gap series, scheme {s}:")
        for row in out['gap_series'][s]:
            print(f"    level {row['level_pct']:>5.1f}%: id={row['id_mAP']:.3f} "
                  f"ood={row['ood_mAP']:.3f}  abs_gap={row['abs_gap']:.3f}  "
                  f"gap/id={row['gap_over_id']:.4f}")

    return out


def main():
    print(f"Device: {_DEVICE}", flush=True)
    all_out = {}
    for ds_name, cfg in DATASETS.items():
        all_out[ds_name] = run_dataset(ds_name, cfg)
    out_path = RESULTS / 'e43_run1_hub_pruning.json'
    with open(out_path, 'w') as f:
        json.dump(all_out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
