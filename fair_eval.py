"""Fixed evaluation: score steps only, threshold-free, and measure what a method adds over blind baselines.

For each method (scores from evaluate.py --scores_out, averaged over seeds) and each blind score
(shortcuts.py --scores_out), on the same evaluation steps:
  AUROC        standard step-level ROC AUC (no background frames, no thresholds), 95% CI by recording
  blind+       AUROC of a logistic regression on [the five blind scores, their sum, this method's score]
  gain         blind+ minus the same regression without the method's score, with a paired 95% CI.
               Only information the blind scores do not already carry can make this positive.
All regression inputs are ranks in [0, 1].
The regressions are fit with 5-fold cross-validation grouped by recording, so every prediction is
for a recording the regression did not see.

Usage: python fair_eval.py <steps table> <method scores npz> <blind scores npz> [--group execution]
"""
import argparse

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

EXECUTION = ["Technique Error", "Preparation Error", "Measurement Error", "Error_Modification", "Error_Slip"]  # CC4D, EgoPER
PROCEDURAL = ["Order Error", "Error_Addition"]

parser = argparse.ArgumentParser()
parser.add_argument("table")
parser.add_argument("method_scores")
parser.add_argument("blind_scores")
parser.add_argument("--group", default="all", choices=["all", "execution", "procedural"])
parser.add_argument("--boot", type=int, default=1000)
args = parser.parse_args()

t = np.load(args.table)
categories = t["categories"]
has = lambda names: t["errors"][:, np.isin(categories, names)].any(1)
correct = ~t["errors"].any(1)
target = {"all": ~correct, "execution": has(EXECUTION) & ~has(PROCEDURAL),
          "procedural": has(PROCEDURAL) & ~has(EXECUTION)}[args.group]

ms = np.load(args.method_scores)
idx = ms["eval_idx"]
keep = correct[idx] | target[idx]  # this group's error steps vs all correct steps
idx = idx[keep]
y, rec = target[idx], t["recording_id"][idx]

bs = np.load(args.blind_scores)
# Blind inputs: the five scores plus their sum, as ranks in [0, 1] (the raw scores have long tails).
rank = lambda v: rankdata(v) / len(v)
blind = np.column_stack([rank(v) for v in [*bs["features"][idx].T, bs["sum"][idx]]])
candidates = {"constant": np.zeros(len(idx)), "blind sum": bs["sum"][idx]}
for name in sorted({k.rsplit("_seed", 1)[0] for k in ms.files if k != "eval_idx"}):
    seeds = [ms[k][keep] for k in ms.files if k.rsplit("_seed", 1)[0] == name]
    candidates[name] = np.mean(seeds, 0)


def cross_val_scores(X):
    out = np.empty(len(y))
    for fit, held in GroupKFold(5).split(X, y, rec):
        out[held] = LogisticRegression(max_iter=1000).fit(X[fit], y[fit]).decision_function(X[held])
    return out


def boot_ci(*scores):
    """95% CI of AUROC(scores[0]) - AUROC(scores[1]) (or of AUROC(scores[0])), resampling recordings."""
    rng = np.random.default_rng(0)
    uniq = np.unique(rec)
    idx_of = {u: np.flatnonzero(rec == u) for u in uniq}
    vals = []
    for _ in range(args.boot):
        b = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))])
        if 0 < y[b].sum() < len(b):
            a = roc_auc_score(y[b], scores[0][b])
            vals.append(a - roc_auc_score(y[b], scores[1][b]) if len(scores) > 1 else a)
    return np.percentile(vals, [2.5, 97.5])


base = cross_val_scores(blind)
print(f"{args.group} errors vs correct steps: {y.sum()} vs {(~y).sum()}, {len(np.unique(rec))} recordings")
print(f"blind scores alone (cross-validated regression): AUROC {roc_auc_score(y, base):.3f}\n")
print(f"{'score':<14} {'AUROC [95% CI]':>22} {'blind+':>8} {'gain [95% CI]':>24}")
for name, s in candidates.items():
    auc = roc_auc_score(y, s) if s.std() else 0.5
    lo, hi = boot_ci(s) if s.std() else (0.5, 0.5)
    plus = cross_val_scores(np.column_stack([blind, rank(s)]))
    g_lo, g_hi = boot_ci(plus, base)
    gain = roc_auc_score(y, plus) - roc_auc_score(y, base)
    print(f"{name:<14} {auc:6.3f} [{lo:.3f}, {hi:.3f}] {roc_auc_score(y, plus):8.3f} "
          f"{gain:+8.3f} [{g_lo:+.3f}, {g_hi:+.3f}]")
