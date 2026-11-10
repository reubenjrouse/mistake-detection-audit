"""Turn AMNAR's test output into one error score per annotated step, in evaluate.py --scores_out format.

AMNAR (github.com/iSEE-Laboratory/AMNAR) flags a predicted segment when its distance to the nearest
normal action representation exceeds mean + threshold * std of that action. test_ed.py saves, for each
threshold, (distance, mean + threshold * std, flag) per predicted segment; from the files for thresholds
0 and 1 we recover the continuous score z = (distance - mean) / std. Each annotated step gets the largest
z among predicted segments that overlap it (steps no segment covers get the lowest score).

Usage: python audit/amnar_scores.py <AMNAR ckpt dir with pred_seg_results_*.pkl> <steps table> <out.npz>
       [--fps 1] [--quantile 0.85] [--name amnar]
"""
import argparse
from pathlib import Path

import numpy as np

from safe_pickle import load

parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir", type=Path)
parser.add_argument("table")
parser.add_argument("out")
parser.add_argument("--fps", type=float, default=1, help="frames per second of AMNAR's segments")
parser.add_argument("--quantile", type=float, default=0.85)
parser.add_argument("--name", default="amnar")
args = parser.parse_args()

res0 = load(args.ckpt_dir / "pred_seg_results_0.00.pkl")
res1 = load(args.ckpt_dir / "pred_seg_results_1.00.pkl")

segments = {}  # recording id -> (n, 2) seconds, (n,) z
for vid, r0 in res0.items():
    d0 = r0["diff_and_threshold"][args.quantile]
    d1 = res1[vid]["diff_and_threshold"][args.quantile]
    dist, mean = np.array([float(x[0]) for x in d0]), np.array([float(x[1]) for x in d0])
    std = np.array([float(x[1]) for x in d1]) - mean
    valid = std > 0  # background segments carry (0, 0, False)
    z = np.where(valid, (dist - mean) / np.where(valid, std, 1), np.nan)
    segs = np.asarray(r0["segments"], float).reshape(-1, 2)[:len(z)] / args.fps
    rid = vid.split("_", 1)[1] if vid.count("_") == 2 else vid  # task '0' prefixes the recipe index
    segments[rid] = (segs, z)

t = np.load(args.table)
eval_idx = np.flatnonzero(t["recording_error"])
all_z = np.concatenate([z[np.isfinite(z)] for _, z in segments.values()])
lowest = all_z.min() - 1
scores, missing = np.full(len(eval_idx), lowest), 0
for j, i in enumerate(eval_idx):
    rid = t["recording_id"][i]
    if rid not in segments:
        missing += 1
        continue
    segs, z = segments[rid]
    overlap = (segs[:, 0] < t["end"][i]) & (segs[:, 1] > t["start"][i]) & np.isfinite(z)
    if overlap.any():
        scores[j] = z[overlap].max()

np.savez(args.out, eval_idx=eval_idx, **{f"{args.name}_seed0": scores})
print(f"wrote {args.out}: {len(eval_idx)} steps, {(scores > lowest).sum()} covered by a scored segment, "
      f"{missing} steps in recordings without AMNAR output")
