"""Lay out CaptainCook4D for AMNAR: annotations converted from 10 fps to 1 fps, features as <rid>_360p.npy.

Annotations come from AMNAR's attachments/CC4D_process; features are 1 fps npz files (our DINOv2-S, or the
released Omnivore / SlowFast features). Output root is the one set in the patched captaincook4d.py.

Usage: python prepare_cc4d.py <AMNAR attachments dir> <features root> <out root> [--features_names dinov2_s,omnivore,slowfast]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("attachments", type=Path)
parser.add_argument("features", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--features_names", default="dinov2_s,omnivore,slowfast")
args = parser.parse_args()

src, dst = args.attachments / "CC4D_process", args.out / "CC4D_process"
dst.mkdir(parents=True, exist_ok=True)
for name in ["error_samples_processed.json", "non_error_samples_processed.json"]:
    d = json.load(open(src / name))
    for task in d.values():
        for v in task.values():
            v["segments"] = [[a // 10, b // 10] for a, b in v["segments"]]
    json.dump(d, open(dst / name, "w"))
for name in ["step_id_mapping.json", "task_graph.json"]:
    shutil.copy(src / name, dst / name)

for feat in args.features_names.split(","):
    out = args.out / "feats" / feat
    out.mkdir(parents=True, exist_ok=True)
    files = list((args.features / feat).glob("*_360p.mp4_1s_1s.npz"))
    for f in files:
        np.save(out / (f.name.split("_360p")[0] + "_360p.npy"), np.load(f)["arr_0"].astype(np.float32))
    print(feat, len(files), "recordings")
