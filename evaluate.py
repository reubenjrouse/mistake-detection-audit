"""E1: score steps with each method, report AUROC / AP (95% CI) per error group.

Each group is evaluated as (that group's error steps) vs (all correct steps).
Develop on val (default); run --test only for final numbers.

Usage: python evaluate.py C:/Users/Reuben/data/cc4d/steps_dinov2_s_w2.npz [--test] [--raw]
       [--protocol normal_vs_error] [--seeds 0,1,2,3,4] [--out results/e1_test.csv]
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from methods import METHODS

# Fixed before looking at any results.
EXECUTION = ["Technique Error", "Preparation Error", "Measurement Error"]
PROCEDURAL = ["Order Error"]
GROUPS = ["execution", "procedural", "mixed", "other"]
MIN_CALIB = 3  # steps with fewer out-of-fold correct scores use global stats


def assign_groups(errors, categories):
    """correct | execution (no order) | procedural (order only) | mixed | other."""
    has = lambda names: errors[:, np.isin(categories, names)].any(1)
    execution, procedural = has(EXECUTION), has(PROCEDURAL)
    group = np.full(len(errors), "other", dtype=object)
    group[~errors.any(1)] = "correct"
    group[execution & ~procedural] = "execution"
    group[procedural & ~execution] = "procedural"
    group[execution & procedural] = "mixed"
    return group


def select(rows, mask):
    return {k: v[mask] for k, v in rows.items()}


def calibrate(make, train):
    """Per-step mean/std of scores on correct steps the method did not see.

    Leave-one-person-out over training people, so no val/test data is used.
    """
    step_ids, scores = [], []
    for person in np.unique(train["person_id"]):
        fit_rows = select(train, train["person_id"] != person)
        held = select(train, (train["person_id"] == person)
                      & np.isin(train["step_id"], fit_rows["step_id"]))
        step_ids.append(held["step_id"])
        scores.append(make().fit(fit_rows).score(held))
    step_ids, scores = np.concatenate(step_ids), np.concatenate(scores)
    stats = {"global": (scores.mean(), scores.std())}
    for sid in np.unique(step_ids):
        s = scores[step_ids == sid]
        if len(s) >= MIN_CALIB:
            stats[sid] = (s.mean(), s.std() + 1e-6)
    return stats


def normalize(scores, step_ids, stats):
    mu_sd = np.array([stats.get(sid, stats["global"]) for sid in step_ids])
    return (scores - mu_sd[:, 0]) / mu_sd[:, 1]


def metrics_with_ci(y, s, recordings, n_boot, rng):
    """AUROC and AP, with 95% CI from resampling whole recordings."""
    point = roc_auc_score(y, s), average_precision_score(y, s)
    uniq = np.unique(recordings)
    idx_of = {r: np.flatnonzero(recordings == r) for r in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[r] for r in rng.choice(uniq, len(uniq))])
        if 0 < y[idx].sum() < len(idx):
            boots.append((roc_auc_score(y[idx], s[idx]), average_precision_score(y[idx], s[idx])))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return point, lo, hi


parser = argparse.ArgumentParser()
parser.add_argument("table")
parser.add_argument("--test", action="store_true", help="evaluate on test (final runs only)")
parser.add_argument("--protocol", default="person", choices=["person", "normal_vs_error"],
                    help="person: our person split. normal_vs_error: train on correct steps of normal "
                         "recordings, test on all error recordings (the setup of EgoPED/AMNAR/AEM)")
parser.add_argument("--raw", action="store_true", help="skip per-step score normalization")
parser.add_argument("--boot", type=int, default=1000)
parser.add_argument("--seeds", default="0", help="comma-separated seeds for learned methods")
parser.add_argument("--methods", default="", help="comma-separated subset of methods")
parser.add_argument("--out", default="", help="write one CSV row per method/seed/group")
args = parser.parse_args()

table = dict(np.load(args.table))
categories = table.pop("categories")
table["group"] = assign_groups(table["errors"], categories)

if args.protocol == "person":
    eval_name = "test" if args.test else "val"
    train = select(table, (table["split"] == "train") & (table["group"] == "correct"))
    evalset = select(table, table["split"] == eval_name)
else:
    eval_name = "error recordings"
    train = select(table, ~table["recording_error"] & (table["group"] == "correct"))
    evalset = select(table, table["recording_error"])

print(f"protocol {args.protocol}   train: {len(train['step_id'])} correct steps   "
      f"normalization: {'none' if args.raw else 'per-step (leave-one-person-out)'}")
print(f"{eval_name}: " + ", ".join(f"{g}={(evalset['group'] == g).sum()}"
                                   for g in ["correct"] + GROUPS))

rows = []
for name, Method in METHODS.items():
    if args.methods and name not in args.methods.split(","):
        continue
    for seed in (args.seeds.split(",") if Method.stochastic else ["0"]):
        make = lambda: Method(seed=int(seed))
        method = make().fit(train)
        scores = method.score(evalset)
        pred_err = scores[evalset["group"] == "correct"].mean() if hasattr(method, "expected") else np.nan
        print(f"\n{name} seed {seed}" + ("" if np.isnan(pred_err) else
                                         f"   prediction error on correct steps: {pred_err:.4f}"))
        if not args.raw:
            scores = normalize(scores, evalset["step_id"], calibrate(make, train))
        rng = np.random.default_rng(0)
        for g in GROUPS + ["all"]:
            m = np.ones(len(scores), bool) if g == "all" else np.isin(evalset["group"], ["correct", g])
            y = evalset["group"][m] != "correct"
            (auc, ap), lo, hi = metrics_with_ci(y, scores[m], evalset["recording_id"][m], args.boot, rng)
            print(f"  {g:<11} AUROC {auc:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]   "
                  f"AP {ap:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]   (n={y.sum()})")
            rows.append(dict(protocol=args.protocol, eval=eval_name, method=name, seed=seed, group=g,
                             n_pos=int(y.sum()), n_neg=int((~y).sum()), auroc=auc, auroc_lo=lo[0],
                             auroc_hi=hi[0], ap=ap, ap_lo=lo[1], ap_hi=hi[1], pred_err=pred_err))

if args.out:
    import csv
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")
