"""Blind baselines: how well can CaptainCook4D mistakes be detected without looking at the video?

Every score uses only annotations (which step, when, how long, in what order), fitted on correct steps
of the training recordings, like the one-class methods. AUROC / AP per error group, as in evaluate.py.
  duration       |z| of log step duration vs the same step in training
  position       |z| of relative position in the recording vs the same step in training
  transition     -log P(step | previous performed step) in training recordings
  rec_length     |z| of the recording's number of performed steps vs its recipe (same score for every step)
  rec_duration   |z| of the recording's total duration vs its recipe (same score for every step)
  sum            sum of the five
A last line fits a logistic regression on the five scores with error labels (person split train -> val),
to show how much signal the annotations alone carry.

Usage: python shortcuts.py C:/Users/Reuben/data/cc4d/steps_dinov2_s_w2.npz [--protocol normal_vs_error]
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EXECUTION = ["Technique Error", "Preparation Error", "Measurement Error"]
PROCEDURAL = ["Order Error"]
GROUPS = ["execution", "procedural", "mixed", "other", "all"]

parser = argparse.ArgumentParser()
parser.add_argument("table")
parser.add_argument("--protocol", default="normal_vs_error", choices=["person", "normal_vs_error"])
parser.add_argument("--boot", type=int, default=1000)
parser.add_argument("--scores_out", default="", help="save blind scores for every step (npz)")
args = parser.parse_args()

t = dict(np.load(args.table))
categories = t.pop("categories")
has = lambda names: t["errors"][:, np.isin(categories, names)].any(1)
execution, procedural = has(EXECUTION), has(PROCEDURAL)
group = np.full(len(execution), "other", dtype=object)
group[~t["errors"].any(1)] = "correct"
group[execution & ~procedural] = "execution"
group[procedural & ~execution] = "procedural"
group[execution & procedural] = "mixed"

rec, step = t["recording_id"], t["step_id"]
duration = np.log(np.maximum(t["end"] - t["start"], 0.5))
n_steps = {r: (rec == r).sum() for r in np.unique(rec)}
rel_pos = np.array([p / max(n_steps[r] - 1, 1) for p, r in zip(t["position"], rec)])
prev = np.full(len(step), -1)
for r in np.unique(rec):
    idx = np.flatnonzero(rec == r)
    idx = idx[np.argsort(t["position"][idx])]
    prev[idx[1:]] = step[idx[:-1]]
rec_len = np.array([n_steps[r] for r in rec], float)
rec_dur = np.log(np.array([t["end"][rec == r].max() for r in rec]))

if args.protocol == "person":
    fit_rec = t["split"] == "train"
    evalset = t["split"] == "val"
else:
    fit_rec = ~t["recording_error"]
    evalset = t["recording_error"]
fit = fit_rec & (group == "correct")


def abs_z(values, keys, fit_mask):
    """|z| of each value vs fitted rows with the same key (global stats if fewer than 3)."""
    out = np.empty(len(values))
    g_mu, g_sd = values[fit_mask].mean(), values[fit_mask].std() + 1e-6
    for k in np.unique(keys):
        m = keys == k
        ref = values[fit_mask & m]
        mu, sd = (ref.mean(), ref.std() + 1e-6) if len(ref) >= 3 else (g_mu, g_sd)
        out[m] = np.abs(values[m] - mu) / sd
    return out


counts = {}
for p, s in zip(prev[fit_rec], step[fit_rec]):  # all steps of training recordings: their real order
    counts[p] = counts.get(p, {})
    counts[p][s] = counts[p].get(s, 0) + 1
transition = np.array([-np.log((counts.get(p, {}).get(s, 0) + 0.1) / (sum(counts.get(p, {}).values()) + 1))
                       for p, s in zip(prev, step)])
recipe_of = t["activity_id"]
one_per_rec = np.zeros(len(rec), bool)
one_per_rec[np.unique(rec, return_index=True)[1]] = True
scores = {
    "duration": abs_z(duration, step, fit),
    "position": abs_z(rel_pos, step, fit),
    "transition": transition,
    "rec_length": abs_z(rec_len, recipe_of, fit_rec & ~t["recording_error"] & one_per_rec),
    "rec_duration": abs_z(rec_dur, recipe_of, fit_rec & ~t["recording_error"] & one_per_rec),
}
z = lambda s: (s - s[fit].mean()) / (s[fit].std() + 1e-6)
scores["sum"] = sum(z(s) for s in scores.values())


def report(name, s, mask):
    rng = np.random.default_rng(0)
    line = f"{name:<14}"
    for g in GROUPS:
        m = mask & (np.ones(len(s), bool) if g == "all" else np.isin(group, ["correct", g]))
        y, sc, r = group[m] != "correct", s[m], rec[m]
        uniq = np.unique(r)
        idx_of = {u: np.flatnonzero(r == u) for u in uniq}
        boots = []
        for _ in range(args.boot):
            idx = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq))])
            if 0 < y[idx].sum() < len(idx):
                boots.append(roc_auc_score(y[idx], sc[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        line += f"  {roc_auc_score(y, sc):.3f} [{lo:.2f},{hi:.2f}]"
    print(line)


print(f"protocol {args.protocol}: eval " + ", ".join(f"{g}={(group[evalset] == g).sum()}"
                                                     for g in ["correct", "execution", "procedural", "mixed", "other"]))
print(f"{'AUROC':<14}" + "".join(f"  {g:<17}" for g in GROUPS))
for name, s in scores.items():
    report(name, s, evalset)

X = np.stack([z(s) for s in scores.values() if s is not scores["sum"]], 1)
if args.scores_out:
    np.savez(args.scores_out, features=X, **scores)
    print(f"wrote {args.scores_out}")
tr, va = t["split"] == "train", t["split"] == "val"
clf = LogisticRegression(max_iter=1000).fit(X[tr], group[tr] != "correct")
print("\nsupervised on the five blind scores (person split, train -> val)")
report("logreg", clf.decision_function(X), va)
