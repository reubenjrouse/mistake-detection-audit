"""Timing-matched pairs: each error step paired with a correct step of the same step type from another
recording, with similar values of every annotation-only (blind) feature: step duration, position, how
usual the step order is, and the recording's step count and length. On these pairs the blind features
cannot tell errors from correct steps, so a detector can only score above chance from what it sees.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_pairs(key, features, rec, is_error, is_correct, tol):
    """(n, 2) row indices of (error, correct) pairs; one-to-one, optimal per step type.

    features: (rows, d) values to match; tol: (d,) largest allowed absolute difference per feature.
    """
    features, tol = np.asarray(features, float), np.asarray(tol, float)
    pairs = []
    for k in np.unique(key[is_error]):
        e, c = np.flatnonzero(is_error & (key == k)), np.flatnonzero(is_correct & (key == k))
        if not len(c):
            continue
        diff = np.abs(features[e][:, None] - features[c][None]) / tol  # (e, c, d), 1 = at the tolerance
        cost = diff.sum(-1)
        cost[(diff > 1).any(-1) | (rec[e][:, None] == rec[c][None])] = 1e9  # too far, or same recording
        rows, cols = linear_sum_assignment(cost)
        ok = cost[rows, cols] < 1e9
        pairs += list(zip(e[rows[ok]], c[cols[ok]]))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def pair_metrics(score_error, score_correct, n_boot=1000, seed=0):
    """AUROC over all paired steps and pair accuracy (error scored above its partner; ties count half),
    each with a 95% CI from resampling pairs."""
    from sklearn.metrics import roc_auc_score

    def both(i):
        y = np.r_[np.ones(len(i)), np.zeros(len(i))]
        s = np.r_[score_error[i], score_correct[i]]
        auc = roc_auc_score(y, s) if s.std() else 0.5
        diff = score_error[i] - score_correct[i]
        return auc, (diff > 0).mean() + 0.5 * (diff == 0).mean()

    rng = np.random.default_rng(seed)
    n = len(score_error)
    point = both(np.arange(n))
    boots = np.array([both(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi
