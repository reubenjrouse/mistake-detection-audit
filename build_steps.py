"""Build the steps table: one row per performed step, with s_before / s_after features.

Every method reads this one file, so all methods see exactly the same data.

Usage: python build_steps.py C:/Users/Reuben/data/cc4d [--backbone omnivore] [--window 2]
Writes <data_root>/steps_<backbone>_w<window>.npz
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

CATEGORIES = ["Preparation Error", "Measurement Error", "Order Error", "Timing Error",
              "Technique Error", "Temperature Error", "Missing Step", "Other"]

parser = argparse.ArgumentParser()
parser.add_argument("data_root", type=Path)
parser.add_argument("--backbone", default="omnivore")
parser.add_argument("--window", type=int, default=2, help="seconds averaged for each state")
parser.add_argument("--split", default="person")
args = parser.parse_args()

ann = args.data_root / "annotations"
feat_dir = args.data_root / "features" / args.backbone
recordings = json.load(open(ann / "annotation_json/error_annotations.json"))
meta = json.load(open(ann / "annotation_json/complete_step_annotations.json"))
step_text = json.load(open(ann / "annotation_json/step_idx_description.json"))
splits = json.load(open(ann / f"data_splits/{args.split}_data_split_combined.json"))
split_of = {rid: s for s, rids in splits.items() for rid in rids}

rows = {k: [] for k in ["recording_id", "activity_id", "person_id", "split", "step_id",
                        "position", "start", "end", "description", "errors",
                        "s_before", "s_after"]}
n_missing, n_short = 0, 0

for rec in recordings:
    rid = rec["recording_id"]
    feats = np.load(feat_dir / f"{rid}_360p.mp4_1s_1s.npz")["arr_0"]  # (seconds, dim)
    T = len(feats)
    position = 0
    for step in rec["step_annotations"]:
        if step["start_time"] < 0:  # missing step: never performed, nothing to observe
            n_missing += 1
            continue
        start = min(math.floor(step["start_time"]), T - 1)
        end = min(math.ceil(step["end_time"]), T)
        if end - start < args.window:
            n_short += 1
        # Before: the `window` seconds just before the step starts (at least 1 s).
        b0, b1 = max(0, start - args.window), max(start, 1)
        # After: the last `window` seconds of the step, never earlier than its start.
        a0, a1 = max(start, end - args.window), max(end, start + 1)

        tags = {e["tag"] for e in step.get("errors", [])}
        rows["recording_id"].append(rid)
        rows["activity_id"].append(rec["activity_id"])
        rows["person_id"].append(meta[rid]["person_id"])
        rows["split"].append(split_of[rid])
        rows["step_id"].append(step["step_id"])
        rows["position"].append(position)
        rows["start"].append(step["start_time"])
        rows["end"].append(step["end_time"])
        rows["description"].append(step_text[str(step["step_id"])])
        rows["errors"].append([c in tags for c in CATEGORIES])
        rows["s_before"].append(feats[b0:b1].mean(0))
        rows["s_after"].append(feats[a0:a1].mean(0))
        position += 1

table = {k: np.array(v) for k, v in rows.items()}
table["categories"] = np.array(CATEGORIES)
out = args.data_root / f"steps_{args.backbone}_w{args.window}.npz"
np.savez(out, **table)

print(f"wrote {out}")
print(f"{len(table['step_id'])} performed steps, {n_missing} missing steps skipped, "
      f"{n_short} steps shorter than the {args.window}s window")
print(f"s_before {table['s_before'].shape}, NaNs: {np.isnan(table['s_before']).sum()}, "
      f"s_after NaNs: {np.isnan(table['s_after']).sum()}")
for s in ["train", "val", "test"]:
    m = table["split"] == s
    print(f"  {s}: {m.sum()} steps, {table['errors'][m].any(1).sum()} with errors")
