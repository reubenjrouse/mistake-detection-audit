"""Average per-frame features (~30 fps) into one vector per second, like the segment features.

The CaptainCook4D frame features are ~35 GB in total; per second they are ~1 GB.
Writes <out_dir>/<recording_id>_360p.mp4_1s_1s.npz (float16) so build_steps.py can read it.

Usage: python shrink_frames.py C:/Users/Reuben/Downloads C:/Users/Reuben/data/cc4d/features/omnivore_frames [--delete]
"""
import argparse
import math
from pathlib import Path

import numpy as np

FPS = 29.97

parser = argparse.ArgumentParser()
parser.add_argument("in_dir", type=Path)
parser.add_argument("out_dir", type=Path)
parser.add_argument("--delete", action="store_true", help="delete each original after it is written")
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

for path in sorted(args.in_dir.glob("*_360p.npz")):
    rid = path.name.removesuffix("_360p.npz")
    frames = np.load(path)["arr_0"]
    n_sec = math.ceil(len(frames) / FPS)
    second_of = np.minimum((np.arange(len(frames)) / FPS).astype(int), n_sec - 1)
    sums = np.zeros((n_sec, frames.shape[1]))
    np.add.at(sums, second_of, frames)
    per_sec = sums / np.bincount(second_of, minlength=n_sec)[:, None]

    out = args.out_dir / f"{rid}_360p.mp4_1s_1s.npz"
    np.savez(out, per_sec.astype(np.float16))
    print(f"{rid}: {frames.shape} -> {per_sec.shape}")
    if args.delete:
        path.unlink()
