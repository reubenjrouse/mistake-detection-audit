"""Fixed evaluation of per-step mistake scores: step AUROC, gain over blind scores, and timing-matched pairs.

1. Gain over blind (all, execution and procedural errors vs correct steps). AUROC of a logistic regression on
   the blind scores (ranks in [0, 1], 5-fold cross-validation grouped by recording) with and without the
   method's score; the difference is what the method adds. 95% CIs resample recordings.
2. Timing-matched pairs (all and execution errors). Each error step is paired with a correct step of the same
   recipe from another recording whose five blind scores are all within --tol standard deviations (see
   matching.py). Blind scores are near chance here, so only what a method sees can put it above 0.5.
   Pair accuracy = how often the error step scores higher than its partner.

Method scores: npz files from evaluate.py --scores_out, amnar_scores.py or published_protocol.py --export
(keys eval_idx and <name>_seed<k>; seeds are averaged). Blind scores: blind_scores.py --scores_out or
published_protocol.py --export.

Usage: python audit/audit.py <steps table> <blind scores npz> <method scores npz> [...] [--pairs_out pairs.csv]
"""
import argparse
import csv

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from matching import match_pairs, pair_metrics

EXECUTION = ["Technique Error", "Preparation Error", "Measurement Error", "Error_Modification", "Error_Slip"]  # CC4D, EgoPER
PROCEDURAL = ["Order Error", "Error_Addition"]
BLIND = ["duration", "position", "transition", "rec_length", "rec_duration"]

parser = argparse.ArgumentParser()
parser.add_argument("table")
parser.add_argument("blind_scores")
parser.add_argument("method_scores", nargs="+")
parser.add_argument("--tol", default="0.75,0.75,2,2,2",
                    help="allowed difference per blind score in SDs: duration, position, order, rec. steps, rec. length")
parser.add_argument("--boot", type=int, default=1000)
parser.add_argument("--pairs_out", default="", help="save the matched pairs (csv)")
args = parser.parse_args()

t = np.load(args.table)
has = lambda names: t["errors"][:, np.isin(t["categories"], names)].any(1)
correct = ~t["errors"].any(1)
rec, step = t["recording_id"], t["step_id"]
groups = {"all": ~correct, "execution": has(EXECUTION) & ~has(PROCEDURAL),
          "procedural": has(PROCEDURAL) & ~has(EXECUTION)}

# Every score as a full-length array (NaN outside the evaluation steps); seeds of one method are averaged.
bs = np.load(args.blind_scores)
eval_idx, methods = None, {}
for path in args.method_scores:
    ms = np.load(path)
    assert eval_idx is None or np.array_equal(eval_idx, ms["eval_idx"]), f"{path}: different evaluation steps"
    eval_idx = ms["eval_idx"]
    for name in sorted({k.rsplit("_seed", 1)[0] for k in ms.files if k != "eval_idx"}):
        s = np.full(len(step), np.nan)
        s[eval_idx] = np.mean([ms[k] for k in ms.files if k.rsplit("_seed", 1)[0] == name], 0)
        methods[name] = s


def cv_scores(X, y, r):
    out = np.empty(len(y))
    for fit, held in GroupKFold(5).split(X, y, r):
        out[held] = LogisticRegression(max_iter=1000).fit(X[fit], y[fit]).decision_function(X[held])
    return out


def boot_ci(y, r, a, b=None):
    """95% CI of AUROC(a) - AUROC(b), or of AUROC(a), resampling recordings."""
    rng = np.random.default_rng(0)
    uniq = np.unique(r)
    idx_of = {u: np.flatnonzero(r == u) for u in uniq}
    vals = []
    for _ in range(args.boot):
        i = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))])
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], a[i]) - (roc_auc_score(y[i], b[i]) if b is not None else 0))
    return np.percentile(vals, [2.5, 97.5])


print("1. GAIN OVER BLIND SCORES")
rank = lambda v: rankdata(v) / len(v)
for gname, target in groups.items():
    idx = eval_idx[correct[eval_idx] | target[eval_idx]]
    y, r = target[idx], rec[idx]
    blind = np.column_stack([rank(v) for v in [*bs["features"][idx].T, bs["sum"][idx]]])
    base = cv_scores(blind, y, r)
    print(f"\n{gname} errors vs correct steps: {y.sum()} vs {(~y).sum()}, {len(np.unique(r))} recordings; "
          f"blind scores alone: AUROC {roc_auc_score(y, base):.3f}")
    print(f"  {'score':<18} {'AUROC [95% CI]':>22} {'blind+':>8} {'gain [95% CI]':>24}")
    for name, s in {"constant": np.zeros(len(step)), "blind sum": bs["sum"], **methods}.items():
        s = s[idx]
        auc = roc_auc_score(y, s) if s.std() else 0.5
        lo, hi = boot_ci(y, r, s) if s.std() else (0.5, 0.5)
        plus = cv_scores(np.column_stack([blind, rank(s)]), y, r)
        g_lo, g_hi = boot_ci(y, r, plus, base)
        a_plus = roc_auc_score(y, plus)
        print(f"  {name:<18} {auc:6.3f} [{lo:.3f}, {hi:.3f}] {a_plus:8.3f} "
              f"{a_plus - roc_auc_score(y, base):+8.3f} [{g_lo:+.3f}, {g_hi:+.3f}]")

print("\n2. TIMING-MATCHED PAIRS")
pool = t["recording_error"]
tol = [float(v) for v in args.tol.split(",")]
rows = []
for gname in ["all", "execution"]:
    pairs = match_pairs(t["activity_id"], bs["features"], rec, pool & groups[gname], pool & correct, tol)
    e, c = pairs[:, 0], pairs[:, 1]
    rows += [(gname, rec[i], step[i], f"{t['start'][i]:.2f}", f"{t['end'][i]:.2f}",
              rec[j], step[j], f"{t['start'][j]:.2f}", f"{t['end'][j]:.2f}") for i, j in pairs]
    print(f"\n{gname} errors: {len(pairs)} pairs, {len(set(rec[e]) | set(rec[c]))} recordings, "
          f"{len(set(step[e]))} step types (same recipe, blind scores within {args.tol} SD)")
    print(f"  {'score':<18} {'AUROC [95% CI]':>22} {'pair accuracy [95% CI]':>26}")
    for name, s in {"constant": np.zeros(len(step)), **{k: bs[k] for k in BLIND + ["sum"]}, **methods}.items():
        (auc, acc), lo, hi = pair_metrics(s[e], s[c])
        print(f"  {name:<18} {auc:6.3f} [{lo[0]:.3f}, {hi[0]:.3f}]   {acc:6.3f} [{lo[1]:.3f}, {hi[1]:.3f}]")

if args.pairs_out:
    with open(args.pairs_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "error_recording", "error_step", "error_start", "error_end",
                    "correct_recording", "correct_step", "correct_start", "correct_end"])
        w.writerows(rows)
    print(f"\nwrote {args.pairs_out}")
