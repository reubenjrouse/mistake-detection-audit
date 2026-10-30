"""Evaluate on timing-matched pairs of CaptainCook4D steps (see matching.py).

Pairs come from the error recordings (the published test set): each error step is paired with a correct
step of the same recipe (or, with --key step, the same step type) from another error recording, matched
on the five blind scores of shortcuts.py (step duration, position, order, recording step count and
length), each within its --tol in standard deviations. Matching all five within one step type leaves
too few pairs on CaptainCook4D, so the default matches within a recipe. Scores are the
saved per-step scores of evaluate.py --scores_out (averaged over seeds) and shortcuts.py --scores_out.
A blind score near 0.5 here means the matching removed that shortcut.

Usage: python matched_eval.py <steps table> <method scores npz> <blind scores npz> [--key recipe] [--tol 0.75,0.75,2,2,2]
"""
import argparse

import numpy as np

from matching import match_pairs, pair_metrics

EXECUTION = ["Technique Error", "Preparation Error", "Measurement Error", "Error_Modification", "Error_Slip"]  # CC4D, EgoPER
PROCEDURAL = ["Order Error", "Error_Addition"]

parser = argparse.ArgumentParser()
parser.add_argument("table")
parser.add_argument("method_scores")
parser.add_argument("blind_scores")
parser.add_argument("--key", default="recipe", choices=["recipe", "step"], help="pair only within this")
parser.add_argument("--tol", default="0.75,0.75,2,2,2",
                    help="allowed difference per blind score in SDs: duration, position, order, rec. steps, rec. length")
parser.add_argument("--out", default="", help="save the pairs (npz)")
args = parser.parse_args()

t = np.load(args.table)
categories = t["categories"]
has = lambda names: t["errors"][:, np.isin(categories, names)].any(1)
correct = ~t["errors"].any(1)
rec, step = t["recording_id"], t["step_id"]

# Every score as a full-length array (NaN outside the evaluation set).
ms, bs = np.load(args.method_scores), np.load(args.blind_scores)
scores = {"constant": np.zeros(len(step))}
scores.update({k: bs[k] for k in bs.files if k != "features"})
for name in sorted({k.rsplit("_seed", 1)[0] for k in ms.files if k != "eval_idx"}):
    s = np.full(len(step), np.nan)
    s[ms["eval_idx"]] = np.mean([ms[k] for k in ms.files if k.rsplit("_seed", 1)[0] == name], 0)
    scores[name] = s

pool = t["recording_error"]
blind = bs["features"]  # the five blind scores, standardized: tolerance is in standard deviations
tol = [float(v) for v in args.tol.split(",")]
key = step if args.key == "step" else t["activity_id"]
groups = {"all": ~correct, "execution": has(EXECUTION) & ~has(PROCEDURAL)}
saved = {}
for gname, g in groups.items():
    pairs = match_pairs(key, blind, rec, pool & g, pool & correct, tol)
    saved[gname] = pairs
    e, c = pairs[:, 0], pairs[:, 1]
    print(f"\n{gname} errors: {len(pairs)} pairs, {len(set(rec[e]) | set(rec[c]))} recordings, "
          f"{len(set(step[e]))} step types (same {args.key}, blind scores within {args.tol} SD)")
    print(f"  {'score':<14} {'AUROC [95% CI]':>22} {'pair accuracy [95% CI]':>26}")
    for name, s in scores.items():
        (auc, acc), lo, hi = pair_metrics(s[e], s[c])
        print(f"  {name:<14} {auc:6.3f} [{lo[0]:.3f}, {hi[0]:.3f}]   {acc:6.3f} [{lo[1]:.3f}, {hi[1]:.3f}]")

if args.out:
    np.savez(args.out, **saved)
    print(f"\nwrote {args.out}")
