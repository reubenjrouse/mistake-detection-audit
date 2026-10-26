"""E1: score test steps with each method, report AUROC / AP per error group.

Each group is evaluated as (that group's error steps) vs (all correct test steps).

Usage: python evaluate.py C:/Users/Reuben/data/cc4d/steps_omnivore_w2.npz
"""
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from methods import METHODS

# Fixed before looking at any results.
EXECUTION = ["Technique Error", "Preparation Error", "Measurement Error"]
PROCEDURAL = ["Order Error"]
GROUPS = ["execution", "procedural", "mixed", "other"]


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


table = dict(np.load(sys.argv[1]))
categories = table.pop("categories")
table["group"] = assign_groups(table["errors"], categories)
select = lambda mask: {k: v[mask] for k, v in table.items()}

train = select((table["split"] == "train") & (table["group"] == "correct"))
test = select(table["split"] == "test")

print(f"train: {len(train['step_id'])} correct steps")
print("test:  " + ", ".join(f"{g}={(test['group'] == g).sum()}" for g in ["correct"] + GROUPS))
print(f"\n{'method':<12}" + "".join(f"{g:>22}" for g in GROUPS))
print(f"{'':<12}" + "".join(f"{'AUROC    AP':>22}" for _ in GROUPS))

for name, Method in METHODS.items():
    scores = Method().fit(train).score(test)
    line = f"{name:<12}"
    for g in GROUPS:
        m = np.isin(test["group"], ["correct", g])
        y = test["group"][m] == g
        line += f"{roc_auc_score(y, scores[m]):>16.3f}{average_precision_score(y, scores[m]):>6.3f}"
    print(line)
