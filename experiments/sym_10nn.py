"""
Symmetric-10NN fraction. Shared implementation for all compute scripts
and the unit test (Ch2 rev 186).

Prior to this module there were four copies of this logic across the
codebase (test, Run 2, k-rec gallery-shape, k-rec blended bootstrap),
two of which were wrong at different times. Single implementation,
imported, tested.
"""
import numpy as np


def symmetric_knn_fraction(top_nb, k=10):
    """Fraction of ordered pairs (i, j = top_nb[i, r]) for r in [0, k)
    that satisfy i in top_nb[j, :k] (reciprocal).

    Returns a float in [0, 1]. Normaliser is top_nb.shape[0] * k.
    """
    N = top_nb.shape[0]
    total = N * k
    topk_sets = [set(int(x) for x in top_nb[j, :k]) for j in range(N)]
    sym = 0
    for i in range(N):
        for r in range(k):
            j = int(top_nb[i, r])
            if i in topk_sets[j]:
                sym += 1
    return sym / total


def per_item_reciprocity(top_nb, k=10):
    """Per-item indicator: fraction of i's top-k neighbours j for which
    i is also in j's top-k, i.e. sym_i = |{r in [0,k): i in top_nb[j=top_nb[i,r]]}| / k.

    Returns a 1-D float array of length N. Averaging over any subset of
    items gives the sym-10NN fraction on that subset conditional on the
    full-gallery neighbour graph. Used by the bootstrap per Ch2 rev 183
    (no neighbour filtering).
    """
    N = top_nb.shape[0]
    topk_sets = [set(int(x) for x in top_nb[j, :k]) for j in range(N)]
    out = np.zeros(N, dtype=np.float64)
    for i in range(N):
        count = 0
        for r in range(k):
            j = int(top_nb[i, r])
            if i in topk_sets[j]:
                count += 1
        out[i] = count / k
    return out
