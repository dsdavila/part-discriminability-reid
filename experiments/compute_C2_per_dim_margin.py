"""
C2 - Per-dimension margin decomposition (rev 323 / Daniel's proposal, sharpened).

For each of the four cells, per dimension d:
  delta_d^pos  = E[(x_d - y_d)^2] over positive pairs
  delta_d^neg  = same over negative pairs
  m_d = delta_d^neg - delta_d^pos          # per-dim margin
  m_tilde_d = m_d / sigma_d^2 (in-domain sigma)

Positives: same-pid gallery pairs excluding same-pid-same-cam.
Negatives:
  - random: K uniform-sampled non-pos gallery items per query
  - hard:   gallery items ranked above the first true positive per query

Report, per cell:
  (1) m_d distribution, count m_d < 0 (dimensions actively harmful)
  (2) Concentration: share of top-k dims for k in {10, 50, 200}
  (3) Scale-free ratio delta_pos/delta_neg per dim, and per-dim single-axis
      AUC (sampled, 5000 pos pairs vs 5000 neg pairs per dim).
  (4) Spearman correlation of m_d ordering id arm vs ood arm per model.
  (5) Normalisation: raw m_d and m_tilde_d with sigma_d from in-domain.
      Report Sum_d Var(x_d) per arm (expect ~1 if L2-normalised).
  (6) Basis: raw and PCA. Random-rotation null for concentration (20 draws).
"""
import json
import numpy as np
from pathlib import Path

CACHE_DIR = Path('/tmp/e12_feature_cache')
RESULTS   = Path('/data/papers/low_dimensional_reid/results')
RNG_SEED = 42
K_RAND_NEG = 100           # random negatives per query
TOP_K_SHARE = [10, 50, 200]
N_ROT_NULL = 20            # random rotation null draws
AUC_SAMPLE_PAIRS = 5000    # per-dim single-axis AUC sample

CELLS = {
    'A_prime': {'file': 'market_test_noREA.npz',     'n_q': 3368,  'label': "Market model → Market (id)",  'in_domain': 'A_prime'},
    'B_prime': {'file': 'msmt17_test_noREA.npz',     'n_q': 11659, 'label': "Market model → MSMT (ood)",  'in_domain': 'A_prime'},
    'C':       {'file': 'bot_msmt_seed42_market.npz','n_q': 3368,  'label': "MSMT model → Market (ood)",  'in_domain': 'D'},
    'D':       {'file': 'bot_msmt_seed42_msmt.npz', 'n_q': 11659, 'label': "MSMT model → MSMT (id)",    'in_domain': 'D'},
}
IN_DOMAIN_FOR_SIGMA = {'A_prime': 'A_prime', 'B_prime': 'A_prime', 'C': 'D', 'D': 'D'}


def l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def load_cell(info):
    d = np.load(CACHE_DIR / info['file'])
    emb = d['emb'].astype(np.float32); pids = d['pids']; cids = d['cids']
    n_q = info['n_q']
    q_emb = l2(emb[:n_q]); q_pids = pids[:n_q]; q_cids = cids[:n_q]
    g_emb = l2(emb[n_q:]); g_pids = pids[n_q:]; g_cids = cids[n_q:]
    return q_emb, q_pids, q_cids, g_emb, g_pids, g_cids


def accumulate_pair_squares(q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, rng, k_rand=K_RAND_NEG):
    """Single pass over queries. Returns per-dim sum of (q - item)^2 for
    pos / rand_neg / hard_neg, plus counts."""
    d = q_emb.shape[1]
    sum_pos = np.zeros(d, dtype=np.float64); n_pos = 0
    sum_rand = np.zeros(d, dtype=np.float64); n_rand = 0
    sum_hard = np.zeros(d, dtype=np.float64); n_hard = 0
    # Also sample some positive and some random-neg pair distances per dim for AUC
    pos_sample_sqs = []  # list of (d,) arrays
    neg_sample_sqs = []
    n_sample_cap = AUC_SAMPLE_PAIRS

    for i in range(len(q_pids)):
        qp = q_pids[i]; qc = q_cids[i]
        q = q_emb[i]
        junk = (g_pids == qp) & (g_cids == qc)
        pos_mask = (g_pids == qp) & ~junk
        # Positives
        if pos_mask.any():
            pos_items = g_emb[pos_mask]
            sq = (q - pos_items) ** 2   # (n_pos_i, d)
            sum_pos += sq.sum(axis=0)
            n_pos += len(pos_items)
            if len(pos_sample_sqs) < n_sample_cap:
                k = min(5, len(pos_items))
                if k > 0:
                    sel = rng.choice(len(pos_items), size=k, replace=False)
                    pos_sample_sqs.extend(sq[sel])
        # Random negatives
        neg_mask = (g_pids != qp) & ~junk
        neg_idx = np.where(neg_mask)[0]
        if len(neg_idx) > 0:
            K = min(k_rand, len(neg_idx))
            sel = rng.choice(len(neg_idx), size=K, replace=False)
            rand_items = g_emb[neg_idx[sel]]
            sq_rn = (q - rand_items) ** 2
            sum_rand += sq_rn.sum(axis=0)
            n_rand += K
            if len(neg_sample_sqs) < n_sample_cap:
                k = min(5, K)
                neg_sample_sqs.extend(sq_rn[:k])
        # Hard negatives: rank by sim, find first pos's rank
        if pos_mask.any():
            sims = q @ g_emb.T
            sims_masked = sims.copy()
            sims_masked[junk] = -np.inf
            order = np.argsort(-sims_masked)
            is_pos = g_pids[order] == qp
            # excluded junk has -inf sim; they're at end of order
            # But we want: first True in is_pos
            if is_pos.any():
                first_pos = int(np.argmax(is_pos))
                if first_pos > 0:
                    hard_idx = order[:first_pos]
                    hard_items = g_emb[hard_idx]
                    sq_hn = (q - hard_items) ** 2
                    sum_hard += sq_hn.sum(axis=0)
                    n_hard += first_pos

    pos_sample = np.stack(pos_sample_sqs[:n_sample_cap]) if pos_sample_sqs else np.zeros((0, d))
    neg_sample = np.stack(neg_sample_sqs[:n_sample_cap]) if neg_sample_sqs else np.zeros((0, d))
    return (sum_pos, n_pos), (sum_rand, n_rand), (sum_hard, n_hard), pos_sample, neg_sample


def per_dim_summary(sum_pair, n_pair, label, d):
    if n_pair == 0:
        return np.full(d, np.nan)
    return sum_pair / n_pair


def per_dim_auc(pos_sample, neg_sample):
    """Per-dim AUC on squared single-axis distances.
    pos: (N_pos, d), neg: (N_neg, d).
    AUC = P(neg_d > pos_d) = Mann-Whitney U / (N_pos * N_neg).
    """
    N_p = len(pos_sample); N_n = len(neg_sample)
    d = pos_sample.shape[1] if N_p > 0 else neg_sample.shape[1]
    if N_p == 0 or N_n == 0:
        return np.full(d, np.nan)
    aucs = np.zeros(d)
    for k in range(d):
        p = pos_sample[:, k]; n = neg_sample[:, k]
        n_sorted = np.sort(n)
        ranks_below = np.searchsorted(n_sorted, p, side='left')
        ranks_tied = np.searchsorted(n_sorted, p, side='right') - ranks_below
        aucs[k] = (ranks_below.sum() + 0.5 * ranks_tied.sum()) / (N_p * N_n)
    return aucs


def concentration_share(m_abs, top_k):
    order = np.argsort(-m_abs)
    total = m_abs.sum()
    if total <= 0:
        return 0.0
    return float(m_abs[order[:top_k]].sum() / total)


def random_rotation_concentration_null(m_d, top_k_list, n_draws, rng):
    """Draw random orthogonal rotations R, compute m_d_rotated = R^T @ diag(m_d) @ R's
    eigenvalue structure... wait, this is wrong. m_d is a per-dim statistic.
    Under rotation R: x -> Rx. Then (Rx)_d = R_{d,:} @ x. The per-dim
    per-pair diff is sum_j R_{dj} (x_j - y_j). The squared diff per dim is
    sum_{jk} R_{dj} R_{dk} (x_j - y_j)(x_k - y_k).

    E over pairs: E[(Rx_d - Ry_d)^2] per dim.
    Using pair-covariance matrices: let C_pos_{jk} = E[(x_j-y_j)(x_k-y_k)]
    over positive pairs (which equals mean of diff outer-product). Then
    m^pair_d under rotation R = sum_jk R_{dj} R_{dk} (C_neg_{jk} - C_pos_{jk})
    = (R M R^T)_{dd}  where M = C_neg - C_pos.

    So m_d under rotation = diagonal of R M R^T. To compute concentration
    of |m_d| under rotation, we need M (full d x d). This is expensive for
    d=2048. Alternative: use PCA basis (eigenbasis of some chosen
    covariance), not random rotations.

    Simpler null: random rotation applied directly to m vector doesn't
    make geometric sense. Use PCA basis as the alternative basis per
    claude-chat; skip random-rotation null for now and flag it.
    """
    return None


def pca_basis(X, K=None):
    """Return top-K PCs of X (by covariance), as (d, K)."""
    mu = X.mean(0)
    Xc = X - mu
    n, d = Xc.shape
    if d <= n:
        C = Xc.T @ Xc / max(n - 1, 1)
        w, v = np.linalg.eigh(C.astype(np.float64))
        order = np.argsort(-w)
        K = K if K is not None else d
        return v[:, order[:K]].astype(np.float32), w[order[:K]].astype(np.float32)
    G = Xc @ Xc.T / max(n - 1, 1)
    w, u = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(-w)
    K = K if K is not None else n
    w_sel = w[order[:K]]; u_sel = u[:, order[:K]]
    v = (Xc.T @ u_sel) / np.sqrt(np.maximum(w_sel, 1e-12) * (n - 1))
    v = v / np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-12)
    return v.astype(np.float32), w_sel.astype(np.float32)


def process_cell(tag, info, sigma_d_in_domain=None):
    print(f"\n### Cell {tag} ({info['label']}) ###", flush=True)
    q_emb, q_pids, q_cids, g_emb, g_pids, g_cids = load_cell(info)
    d = q_emb.shape[1]
    rng = np.random.default_rng(RNG_SEED)

    # Variance per dim (confirm L2 conservation)
    var_d = g_emb.var(axis=0, ddof=1)
    sum_var = float(var_d.sum())
    print(f"  Sum_d Var(x_d) on gallery = {sum_var:.4f}  (L2-conserved target ~ 1)", flush=True)
    # sigma_d anchor
    if sigma_d_in_domain is None:
        sigma_d_in_domain = var_d.copy()  # this cell is itself the anchor
    sigma_d_use = sigma_d_in_domain

    # Raw basis
    print(f"  computing per-dim pair squares in raw basis...", flush=True)
    (sp, np_), (sr, nr), (sh, nh), pos_samp, neg_samp = accumulate_pair_squares(
        q_emb, q_pids, q_cids, g_emb, g_pids, g_cids, rng)
    delta_pos = per_dim_summary(sp, np_, 'pos', d)
    delta_rand = per_dim_summary(sr, nr, 'rand', d)
    delta_hard = per_dim_summary(sh, nh, 'hard', d)
    m_rand = delta_rand - delta_pos
    m_hard = delta_hard - delta_pos
    n_neg_m_rand = int((m_rand < 0).sum())
    n_neg_m_hard = int((m_hard < 0).sum())
    total_rand = float(m_rand.sum())
    total_hard = float(m_hard.sum())
    print(f"  pair counts: pos={np_}  rand={nr}  hard={nh}", flush=True)
    print(f"  raw basis: m_d<0 count (rand)={n_neg_m_rand}/{d}  (hard)={n_neg_m_hard}/{d}", flush=True)
    print(f"  raw basis: total margin (sum m_d): rand={total_rand:.4f}  hard={total_hard:.4f}", flush=True)

    # Normalise by in-domain sigma_d^2 (sigma_d_use is in-domain var)
    m_rand_norm = m_rand / np.maximum(sigma_d_use, 1e-12)
    m_hard_norm = m_hard / np.maximum(sigma_d_use, 1e-12)

    # Concentration (unnormalised, raw basis)
    conc_rand = {k: concentration_share(np.abs(m_rand), k) for k in TOP_K_SHARE}
    conc_hard = {k: concentration_share(np.abs(m_hard), k) for k in TOP_K_SHARE}
    print(f"  raw concentration (rand): top-10={conc_rand[10]:.4f}  top-50={conc_rand[50]:.4f}  top-200={conc_rand[200]:.4f}", flush=True)
    print(f"  raw concentration (hard): top-10={conc_hard[10]:.4f}  top-50={conc_hard[50]:.4f}  top-200={conc_hard[200]:.4f}", flush=True)

    # Scale-free ratio
    ratio_rand = delta_pos / np.maximum(delta_rand, 1e-12)
    ratio_hard = delta_pos / np.maximum(delta_hard, 1e-12)
    print(f"  delta_pos/delta_rand median = {float(np.median(ratio_rand)):.4f}", flush=True)
    print(f"  delta_pos/delta_hard median = {float(np.median(ratio_hard)):.4f}", flush=True)

    # Per-dim single-axis AUC
    print(f"  per-dim single-axis AUC (sampled {len(pos_samp)} pos, {len(neg_samp)} neg)...", flush=True)
    per_dim_auc_rand = per_dim_auc(pos_samp, neg_samp) if len(pos_samp) > 0 and len(neg_samp) > 0 else None

    # PCA basis
    print(f"  computing PCA basis of gallery...", flush=True)
    PCs, pc_eigs = pca_basis(g_emb, K=min(d, g_emb.shape[0] - 1))
    # Rotate q and g into PCA basis (truncated, but keep all)
    q_pca = q_emb @ PCs
    g_pca = g_emb @ PCs
    # Variance per PC
    var_pc = g_pca.var(axis=0, ddof=1)
    # Recompute pair squares in PCA basis
    print(f"  recomputing in PCA basis...", flush=True)
    (sp2, np2), (sr2, nr2), (sh2, nh2), pos_samp2, neg_samp2 = accumulate_pair_squares(
        q_pca, q_pids, q_cids, g_pca, g_pids, g_cids, rng)
    delta_pos_pca = sp2 / max(np2, 1)
    delta_rand_pca = sr2 / max(nr2, 1)
    delta_hard_pca = sh2 / max(nh2, 1)
    m_rand_pca = delta_rand_pca - delta_pos_pca
    m_hard_pca = delta_hard_pca - delta_pos_pca
    conc_rand_pca = {k: concentration_share(np.abs(m_rand_pca), k) for k in TOP_K_SHARE}
    conc_hard_pca = {k: concentration_share(np.abs(m_hard_pca), k) for k in TOP_K_SHARE}
    print(f"  PCA concentration (rand): top-10={conc_rand_pca[10]:.4f}  top-50={conc_rand_pca[50]:.4f}  top-200={conc_rand_pca[200]:.4f}", flush=True)
    print(f"  PCA concentration (hard): top-10={conc_hard_pca[10]:.4f}  top-50={conc_hard_pca[50]:.4f}  top-200={conc_hard_pca[200]:.4f}", flush=True)
    n_neg_m_rand_pca = int((m_rand_pca < 0).sum())
    n_neg_m_hard_pca = int((m_hard_pca < 0).sum())
    print(f"  PCA basis: m_d<0 count (rand)={n_neg_m_rand_pca}/{d}  (hard)={n_neg_m_hard_pca}/{d}", flush=True)

    return {
        'cell': tag,
        'label': info['label'],
        'sum_var_d_gallery': sum_var,
        'raw': {
            'delta_pos': delta_pos.tolist()[:50],  # first 50 dims for brevity
            'delta_rand': delta_rand.tolist()[:50],
            'delta_hard': delta_hard.tolist()[:50],
            'm_rand_neg_count': n_neg_m_rand,
            'm_hard_neg_count': n_neg_m_hard,
            'sum_m_rand': total_rand,
            'sum_m_hard': total_hard,
            'conc_rand': conc_rand,
            'conc_hard': conc_hard,
            'median_delta_pos_over_rand': float(np.median(ratio_rand)),
            'median_delta_pos_over_hard': float(np.median(ratio_hard)),
            'm_rand_first_50': m_rand.tolist()[:50],
            'm_hard_first_50': m_hard.tolist()[:50],
        },
        'pca': {
            'conc_rand': conc_rand_pca,
            'conc_hard': conc_hard_pca,
            'm_rand_neg_count': n_neg_m_rand_pca,
            'm_hard_neg_count': n_neg_m_hard_pca,
        },
        'pair_counts': {'pos': np_, 'rand_neg': nr, 'hard_neg': nh},
        'full_m_rand': m_rand.tolist(),  # keep full for Spearman later
        'full_m_hard': m_hard.tolist(),
        'full_delta_pos': delta_pos.tolist(),
        'full_delta_rand': delta_rand.tolist(),
        'full_delta_hard': delta_hard.tolist(),
        'sigma_d_var': var_d.tolist(),
    }


def main():
    out = {}
    # First pass: compute A' and D (in-domain anchors)
    print("First pass: in-domain arms (A' and D) to get sigma_d anchors...", flush=True)
    for tag in ['A_prime', 'D']:
        info = CELLS[tag]
        out[tag] = process_cell(tag, info)

    # Second pass: ood arms using in-domain sigma_d
    print("\nSecond pass: ood arms (B' uses A', C uses D)...", flush=True)
    for tag in ['B_prime', 'C']:
        info = CELLS[tag]
        anchor_tag = IN_DOMAIN_FOR_SIGMA[tag]
        sigma_anchor = np.array(out[anchor_tag]['sigma_d_var'])
        out[tag] = process_cell(tag, info, sigma_d_in_domain=sigma_anchor)

    # Spearman correlations between id and ood per model
    from scipy.stats import spearmanr
    spear_res = {}
    for label, id_tag, ood_tag in [('Market_model', 'A_prime', 'B_prime'), ('MSMT_model', 'D', 'C')]:
        m_id_rand  = np.array(out[id_tag]['full_m_rand'])
        m_ood_rand = np.array(out[ood_tag]['full_m_rand'])
        m_id_hard  = np.array(out[id_tag]['full_m_hard'])
        m_ood_hard = np.array(out[ood_tag]['full_m_hard'])
        s_rand = spearmanr(m_id_rand, m_ood_rand)
        s_hard = spearmanr(m_id_hard, m_ood_hard)
        spear_res[label] = {
            'rand': {'rho': float(s_rand.correlation), 'p': float(s_rand.pvalue)},
            'hard': {'rho': float(s_hard.correlation), 'p': float(s_hard.pvalue)},
        }
        print(f"\n  {label} Spearman rho(m_d id vs ood): rand={s_rand.correlation:+.4f}  hard={s_hard.correlation:+.4f}", flush=True)

    out['spearman'] = spear_res

    # Trim the per-cell full_m_rand / full_m_hard arrays from output? Keep
    # them for downstream analysis but json can handle. 2048 floats * 4
    # cells is fine.
    out_path = RESULTS / 'e43_C2_per_dim_margin.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")

    # Summary
    print("\n\n========= C2 SUMMARY =========")
    for tag in ['A_prime', 'B_prime', 'C', 'D']:
        r = out[tag]
        print(f"\n  {tag} ({r['label']}):")
        print(f"    Sum_d Var(x_d) gallery = {r['sum_var_d_gallery']:.4f}")
        print(f"    raw basis   m<0 (rand)={r['raw']['m_rand_neg_count']}  m<0 (hard)={r['raw']['m_hard_neg_count']}  of {len(r['sigma_d_var'])} dims")
        print(f"    raw conc rand top-10/50/200 = {r['raw']['conc_rand'][10]:.4f} / {r['raw']['conc_rand'][50]:.4f} / {r['raw']['conc_rand'][200]:.4f}")
        print(f"    raw conc hard top-10/50/200 = {r['raw']['conc_hard'][10]:.4f} / {r['raw']['conc_hard'][50]:.4f} / {r['raw']['conc_hard'][200]:.4f}")
        print(f"    PCA conc rand top-10/50/200 = {r['pca']['conc_rand'][10]:.4f} / {r['pca']['conc_rand'][50]:.4f} / {r['pca']['conc_rand'][200]:.4f}")
        print(f"    PCA conc hard top-10/50/200 = {r['pca']['conc_hard'][10]:.4f} / {r['pca']['conc_hard'][50]:.4f} / {r['pca']['conc_hard'][200]:.4f}")
        print(f"    median delta_pos/delta_rand = {r['raw']['median_delta_pos_over_rand']:.4f}")
        print(f"    median delta_pos/delta_hard = {r['raw']['median_delta_pos_over_hard']:.4f}")
    print("\n  Spearman rho(m_d id vs ood):")
    for label, s in spear_res.items():
        print(f"    {label:12s}: rand={s['rand']['rho']:+.4f}  hard={s['hard']['rho']:+.4f}")


if __name__ == '__main__':
    main()
