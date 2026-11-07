"""
Asymmetric hubness audit (Ch2 rev 145). Stranger-hubness per identity:
of the items retrieving this identity's items in their top-10, how
many are NOT own-class. Isolates "hub for strangers" from "dense own
cluster" that the raw k-occurrence conflates.

Pre-registered sign (rev 145): stranger-hubness predicts LOWER
own-class rate.

Flow:
  1. Compute top-10 neighbour lists per item per arm.
  2. For each item j, count retrievers (q with j in top-10 of q)
     split by whether pid(q) == pid(j).
  3. Per identity: mean stranger-retriever count across its items.
  4. Tercile by stranger-kocc on id arm, run 3x3 (difficulty x stranger-kocc)
     bootstrap on own-class rate.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_BOOT = 500
K_OCC = 10
K_MAX = 20
DEPTH_RANKS = [1, 5, 10, 20]

CELLS = {
    'D':  ('bot_msmt_seed42_msmt.npz',  11659),
    "B'": ('msmt17_test_noREA.npz',     11659),
}
SIGMA_D = 0.0625
K_RANDPROJ = 17


def load_cell(name):
    fn, n_q = CELLS[name]
    d = np.load(CACHE_DIR / fn); emb = d['emb'].astype(np.float32)
    return emb[n_q:], d['pids'][n_q:], d['cids'][n_q:]


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def noise_degrade(G, sigma, seed=42):
    rng = np.random.default_rng(seed)
    g_eps = rng.normal(0, sigma, G.shape).astype(np.float32)
    return l2(G + g_eps)


def randproj_degrade(G, k_target, seed=42):
    D = G.shape[1]
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, k_target)).astype(np.float32)
    Qmat, _ = np.linalg.qr(R)
    return l2(G @ Qmat)


def per_id_rank1_err(G, g_pids, g_cids, chunk=256):
    N = len(G); wrong = np.zeros(N, dtype=bool)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local, gi in enumerate(range(start, end)):
            exc = (g_pids == g_pids[gi]) & (g_cids == g_cids[gi])
            sims[local, exc] = -2.0
            wrong[gi] = g_pids[int(np.argmax(sims[local]))] != g_pids[gi]
    out = {}
    for pid in np.unique(g_pids):
        out[int(pid)] = float(wrong[g_pids == pid].mean())
    return out


def tercile_assign(values_by_pid):
    pids = np.array(sorted(values_by_pid.keys()))
    vals = np.array([values_by_pid[int(p)] for p in pids])
    order = np.argsort(vals, kind='stable')
    ranks = np.empty_like(order); ranks[order] = np.arange(len(vals))
    edges = np.linspace(0, len(vals), 4).astype(int)
    bins = np.zeros(len(vals), dtype=int)
    for b in range(3):
        bins[(ranks >= edges[b]) & (ranks < edges[b + 1])] = b
    bins[ranks == len(vals) - 1] = 2
    return {int(p): int(b) for p, b in zip(pids, bins)}


def compute_neighbours_and_stats(G, g_pids, g_cids, k_max=K_MAX, chunk=128):
    """For each item i, top-k_max neighbour indices (same-cam-same-pid excluded).
    Returns:
      nb_idx: (N, k_max) int32 — neighbour indices
      same_mask: (N, k_max) bool — True if neighbour is same-id
    """
    N = len(G)
    nb_idx = np.zeros((N, k_max), dtype=np.int32)
    same_mask = np.zeros((N, k_max), dtype=bool)
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        sims = G[start:end] @ G.T
        for local in range(end - start):
            gi = start + local
            qpid, qcid = g_pids[gi], g_cids[gi]
            exc = (g_pids == qpid) & (g_cids == qcid)
            sims[local, exc] = -2.0
            idx = np.argpartition(-sims[local], k_max)[:k_max]
            order = idx[np.argsort(-sims[local, idx])]
            nb_idx[gi] = order
            same_mask[gi] = (g_pids[order] == qpid)
    return nb_idx, same_mask


def main():
    print("Loading MSMT...", flush=True)
    G_id, g_pids, g_cids = load_cell('D')
    G_ood, _, _ = load_cell("B'")
    N = len(G_id)

    G_noise = noise_degrade(G_id, SIGMA_D)
    G_rp = randproj_degrade(G_id, K_RANDPROJ)
    arms = {'id': G_id, 'ood': G_ood, 'noise': G_noise, 'randproj': G_rp}

    # Compute neighbours per arm
    arm_nb = {}
    arm_same = {}
    for arm_name, G in arms.items():
        t0 = time.time()
        print(f"  [{arm_name}] top-{K_MAX} neighbours...", flush=True)
        nb, sm = compute_neighbours_and_stats(G, g_pids, g_cids)
        arm_nb[arm_name] = nb
        arm_same[arm_name] = sm
        print(f"    t={(time.time()-t0)/60:.1f}m", flush=True)

    # Stranger-retriever count per item per arm (at top-K_OCC=10):
    #   stranger_occ[j] = |{q : j in nb[q, :K_OCC] and pid(q) != pid(j)}|
    # We compute this per arm from nb[:, :K_OCC] and same_mask[:, :K_OCC].
    per_item_stranger = {}
    per_item_own = {}
    for arm_name, nb in arm_nb.items():
        sm = arm_same[arm_name][:, :K_OCC]
        nb10 = nb[:, :K_OCC]
        stranger = np.zeros(N, dtype=np.int64)
        own = np.zeros(N, dtype=np.int64)
        for q in range(N):
            for r in range(K_OCC):
                j = nb10[q, r]
                if sm[q, r]:
                    own[j] += 1
                else:
                    stranger[j] += 1
        per_item_stranger[arm_name] = stranger
        per_item_own[arm_name] = own
        print(f"  [{arm_name}] stranger-occ mean={stranger.mean():.2f} max={stranger.max()} "
              f"own-occ mean={own.mean():.2f}", flush=True)

    # Per identity: mean stranger-retriever count across its items (id arm basis)
    stranger_mean_id = {}
    for pid in np.unique(g_pids):
        mask = g_pids == pid
        stranger_mean_id[int(pid)] = float(per_item_stranger['id'][mask].mean())
    stranger_bin = tercile_assign(stranger_mean_id)

    # Difficulty tercile from id arm
    diff = per_id_rank1_err(G_id, g_pids, g_cids)
    diff_bin = tercile_assign(diff)

    pid_to_cell = {int(p): diff_bin[int(p)] * 3 + stranger_bin[int(p)]
                    for p in diff.keys()}
    cell_pids = {c: [] for c in range(9)}
    for pid, c in pid_to_cell.items():
        cell_pids[c].append(int(pid))
    print(f"  cell id counts (row=diff, col=stranger-kocc tercile):", flush=True)
    for s in range(3):
        print(f"    s={s}: {[len(cell_pids[s*3+d]) for d in range(3)]}", flush=True)

    # Per-id same-id counts cumulative by rank per arm (reuse arm_same)
    pid_to_qi = {}
    for qi, pid in enumerate(g_pids):
        pid_to_qi.setdefault(int(pid), []).append(qi)

    per_id = {int(p): {'n_items': len(qi_list)} for p, qi_list in pid_to_qi.items()}
    for arm_name in arms:
        sm = arm_same[arm_name]
        for pid, qi_list in pid_to_qi.items():
            qi_arr = np.array(qi_list)
            subs = sm[qi_arr]  # (n_items, k_max)
            per_id[int(pid)][arm_name] = {r: int(subs[:, :r].sum()) for r in DEPTH_RANKS}

    # Bootstrap
    rng = np.random.default_rng(42)
    row_eff = {arm: {r: {s: [] for s in range(3)} for r in DEPTH_RANKS}
               for arm in arms}
    col_eff = {arm: {r: {d: [] for d in range(3)} for r in DEPTH_RANKS}
               for arm in arms}
    cell_rates_boot = {arm: {r: {c: [] for c in range(9)} for r in DEPTH_RANKS}
                        for arm in arms}

    t0 = time.time()
    for b in range(N_BOOT):
        resampled = {c: rng.choice(cell_pids[c], size=len(cell_pids[c]), replace=True)
                     for c in range(9)}
        for arm in arms:
            for r_pos in DEPTH_RANKS:
                cell_rate = {}
                for c in range(9):
                    same_sum = 0; total = 0
                    for pid in resampled[c]:
                        pid = int(pid)
                        same_sum += per_id[pid][arm][r_pos]
                        total += per_id[pid]['n_items'] * r_pos
                    cell_rate[c] = same_sum / total if total > 0 else 0.0
                for s in range(3):
                    row_eff[arm][r_pos][s].append(cell_rate[s*3+2] - cell_rate[s*3+0])
                for d in range(3):
                    col_eff[arm][r_pos][d].append(cell_rate[2*3+d] - cell_rate[0*3+d])
                for c in range(9):
                    cell_rates_boot[arm][r_pos][c].append(cell_rate[c])
        if (b + 1) % 100 == 0:
            print(f"    boot {b+1}/{N_BOOT} t={(time.time()-t0)/60:.1f}m", flush=True)

    def ci(xs):
        a = np.array(xs)
        return {'mean': float(a.mean()), 'median': float(np.median(a)),
                'lo': float(np.quantile(a, 0.025)), 'hi': float(np.quantile(a, 0.975))}

    out = {'metric': 'own_class_rate_vs_stranger_kocc',
           'row_axis': 'difficulty_rank1err_tercile',
           'col_axis': 'stranger_kocc_mean_id_arm_tercile',
           'per_arm_stranger_summary': {
               arm: {'mean': float(per_item_stranger[arm].mean()),
                      'max': int(per_item_stranger[arm].max()),
                      'skew_items': float(_skew(per_item_stranger[arm]))}
               for arm in arms
           },
           'cell_id_counts': {c: len(cell_pids[c]) for c in range(9)},
           'per_arm': {}}
    for arm in arms:
        out['per_arm'][arm] = {}
        for r_pos in DEPTH_RANKS:
            out['per_arm'][arm][str(r_pos)] = {
                'cell_rates': {str(c): {**ci(cell_rates_boot[arm][r_pos][c]),
                                         'raw': cell_rates_boot[arm][r_pos][c]}
                                for c in range(9)},
                'row_effects': {str(s): {**ci(row_eff[arm][r_pos][s]),
                                          'raw': row_eff[arm][r_pos][s]}
                                 for s in range(3)},
                'col_effects': {str(d): {**ci(col_eff[arm][r_pos][d]),
                                          'raw': col_eff[arm][r_pos][d]}
                                 for d in range(3)},
            }

    print("\n========== Own-class rate vs STRANGER-kocc 3x3 ==========")
    print("Pre-reg: row effect (hi stranger - lo stranger) should be NEGATIVE on id r=1")
    for arm in ('id', 'ood'):
        print(f"\n[{arm}]")
        for r_pos in DEPTH_RANKS:
            print(f"  r={r_pos}:")
            for s in range(3):
                c = ci(row_eff[arm][r_pos][s])
                sig = "*" if c['lo'] * c['hi'] > 0 else " "
                print(f"    row s={s} (hi-lo stranger){sig}: {c['median']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
            for d in range(3):
                c = ci(col_eff[arm][r_pos][d])
                sig = "*" if c['lo'] * c['hi'] > 0 else " "
                print(f"    col d={d} (hard-easy){sig}: {c['median']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")

    out_path = RESULTS / 'e42_stranger_kocc_grid.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


def _skew(x):
    from scipy import stats as _s
    return float(_s.skew(x))


if __name__ == '__main__':
    main()
