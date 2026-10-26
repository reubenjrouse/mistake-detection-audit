"""Day-1 check: is H1 testable? Count step instances by error category and split.

Usage: python inspect_annotations.py C:/Users/Reuben/data/cc4d/annotations [split_name]
split_name: person (default) | environment | recipes | recordings
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
split_name = sys.argv[2] if len(sys.argv) > 2 else "person"

recordings = json.load(open(root / "annotation_json/error_annotations.json"))
splits = json.load(open(root / f"data_splits/{split_name}_data_split_combined.json"))
split_of = {rid: s for s, rids in splits.items() for rid in rids}

counts = defaultdict(Counter)        # split -> category -> n steps
n_correct_train = Counter()          # step_id -> n correct training instances
n_cat_per_step = Counter()           # number of error categories on an error step

for rec in recordings:
    split = split_of.get(rec["recording_id"], "unsplit")
    for step in rec["step_annotations"]:
        tags = {e["tag"] for e in step.get("errors", [])}
        counts[split]["all steps"] += 1
        if not tags:
            counts[split]["correct"] += 1
            if split == "train":
                n_correct_train[step["step_id"]] += 1
        else:
            counts[split]["any error"] += 1
            n_cat_per_step[len(tags)] += 1
            for t in tags:
                counts[split][t] += 1

cats = ["all steps", "correct", "any error", "Preparation Error", "Measurement Error",
        "Technique Error", "Timing Error", "Temperature Error", "Order Error",
        "Missing Step", "Other"]
split_names = sorted(counts)
print(f"Split: {split_name}  recordings per split: "
      + ", ".join(f"{s}={len(splits.get(s, []))}" for s in split_names))
print(f"{'category':<20}" + "".join(f"{s:>9}" for s in split_names))
for c in cats:
    print(f"{c:<20}" + "".join(f"{counts[s][c]:>9}" for s in split_names))

print("\nError categories per error step:", dict(sorted(n_cat_per_step.items())))

v = sorted(n_correct_train.values())
print(f"\nCorrect training instances per step type ({len(v)} step types):")
print(f"  min={v[0]}  median={v[len(v) // 2]}  max={v[-1]}  "
      f"<5: {sum(x < 5 for x in v)}  <10: {sum(x < 10 for x in v)}")
