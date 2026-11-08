"""
E17e mean-only removal rebased onto the P13 canonical cache
(Ch2 rev 209). Identity-clustered paired bootstrap; comparable to
the P15 and P13 intervals.

Protocol: project out gallery mean direction u = mu/|mu|, L2-renorm.
Cache: msmt17_test_noREA.npz (NOT msmt17_test.npz).
Bootstrap: identity-clustered 500 draws on query identities; paired
against the matched baseline draw.
"""
import json, time
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
N_QUERY = 11659  # MSMT: msmt17_test_noREA.npz matches Run 1b's split (Nq=11659)
N_BOOT = 500


def compute_per_query_ap_canonical(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, batch=512):
    n_q = len(q_pids)
    aps = np.zeros(n_q, dtype=np.float32)
    for start in range(0, n_q, batch):
        end = min(start + batch, n_q)
        sim = q_emb[start:end].astype(np.float32) @ g_emb.T
        order = np.argsort(-sim, axis=1)
        for qi in range(end - start):
            qpid, qcid = q_pids[start + qi], q_cids[start + qi]
            idx = order[qi]
            gp = g_pids[idx]; gc = g_cids[idx]
            mask_valid = ~((gp == qpid) & (gc == qcid))
            mask_pos = (gp == qpid) & mask_valid
            if mask_pos.sum() == 0:
                aps[start + qi] = 0.0
                continue
            valid_idx = np.where(mask_valid)[0]
            pos_in_valid = np.where(mask_pos[valid_idx])[0]
            prec = np.arange(1, len(pos_in_valid) + 1) / (pos_in_valid + 1)
            aps[start + qi] = float(prec.mean())
    return aps


def project_out_mean(q_emb, g_emb):
    mu = g_emb.astype(np.float64).mean(0)
    u = mu / np.linalg.norm(mu)
    def rm(X):
        X64 = X.astype(np.float64)
        residual = X64 - np.outer(X64 @ u, u)
        norms = np.linalg.norm(residual, axis=1, keepdims=True)
        return (residual / np.maximum(norms, 1e-10)).astype(np.float32)
    return rm(q_emb), rm(g_emb), u


def main():
    print("Loading msmt17_test_noREA.npz (P13 canonical cache)...", flush=True)
    d = np.load(CACHE_DIR / 'msmt17_test_noREA.npz')
    emb = d['emb'].astype(np.float32)
    pids = d['pids']; cids = d['cids']
    q_emb = emb[:N_QUERY]; q_pids = pids[:N_QUERY]; q_cids = cids[:N_QUERY]
    g_emb = emb[N_QUERY:]; g_pids = pids[N_QUERY:]; g_cids = cids[N_QUERY:]
    print(f"  queries: {len(q_pids)}  gallery: {len(g_pids)}", flush=True)

    # Baseline
    t0 = time.time()
    aps_base = compute_per_query_ap_canonical(q_emb, q_pids, q_cids,
                                                 g_emb, g_pids, g_cids)
    baseline = float(aps_base.mean() * 100)
    print(f"  baseline mAP = {baseline:.4f}%  (expect ~5.396)  t={time.time()-t0:.1f}s", flush=True)

    # Mean-only projection
    print("  projecting out gallery mean direction...", flush=True)
    q_proj, g_proj, u = project_out_mean(q_emb, g_emb)

    t0 = time.time()
    aps_proj = compute_per_query_ap_canonical(q_proj, q_pids, q_cids,
                                                 g_proj, g_pids, g_cids)
    post = float(aps_proj.mean() * 100)
    delta_point = post - baseline
    print(f"  post mAP = {post:.4f}%  delta = {delta_point:+.4f} pp  t={time.time()-t0:.1f}s", flush=True)

    # Identity-clustered paired bootstrap on queries
    print(f"  id-clustered paired bootstrap n={N_BOOT}...", flush=True)
    unique = np.unique(q_pids)
    pid_to_idx = {int(p): np.where(q_pids == p)[0] for p in unique}
    rng = np.random.default_rng(42)
    base_boots = []; post_boots = []; diff_boots = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([pid_to_idx[int(p)] for p in sampled])
        b = float(aps_base[idx].mean() * 100)
        p = float(aps_proj[idx].mean() * 100)
        base_boots.append(b); post_boots.append(p); diff_boots.append(p - b)

    def _ci(arr):
        a = np.array(arr)
        return {'median': float(np.median(a)),
                'lo': float(np.quantile(a, 0.025)),
                'hi': float(np.quantile(a, 0.975))}

    out = {
        'cache': 'msmt17_test_noREA.npz (P13 canonical)',
        'protocol': 'canonical junk; mean-only removal via u=mu/|mu|; L2-renorm',
        'baseline_pct_point': baseline,
        'post_pct_point': post,
        'delta_pp_point': delta_point,
        'baseline_marginal_ci': _ci(base_boots),
        'post_marginal_ci': _ci(post_boots),
        'delta_paired_ci': _ci(diff_boots),
        'n_boot': N_BOOT,
        'bootstrap': 'identity-clustered on query pids, paired',
    }

    print(f"\n  baseline pct: {baseline:.4f}  marg CI=[{out['baseline_marginal_ci']['lo']:.4f}, {out['baseline_marginal_ci']['hi']:.4f}]")
    print(f"  post pct:     {post:.4f}  marg CI=[{out['post_marginal_ci']['lo']:.4f}, {out['post_marginal_ci']['hi']:.4f}]")
    print(f"  PAIRED delta: {delta_point:+.4f} pp  CI=[{out['delta_paired_ci']['lo']:+.4f}, {out['delta_paired_ci']['hi']:+.4f}]")

    out_path = RESULTS / 'e43_e17e_rebase_p13_cache.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
