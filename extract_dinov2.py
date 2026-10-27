"""Extract DINOv2 features at 1 frame per second, in the same format as the segment features.

Row t = frame at t + 0.5 s. Each row is [CLS token, mean of patch tokens].
Writes <out_dir>/<recording_id>_360p.mp4_1s_1s.npz (float16); existing outputs are skipped.

Usage: python extract_dinov2.py D:/cc4d_videos C:/Users/Reuben/data/cc4d/features/dinov2_s [--model dinov2_vits14]
       [--crop 0.25,0.75,0.2,0.85]   # x0,x1,y0,y1 as fractions of the frame: keep only the hands/food region
"""
import argparse
import time
from pathlib import Path

import av
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("video_dir", type=Path)
parser.add_argument("out_dir", type=Path)
parser.add_argument("--model", default="dinov2_vits14")
parser.add_argument("--only", default="", help="comma-separated recording ids")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--crop", default="", help="x0,x1,y0,y1 fractions of the frame to keep")
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

H, W = 224, 392  # 360p is 16:9; keep the whole frame, both sides multiples of the 14 px patch
CROP = [float(v) for v in args.crop.split(",")] if args.crop else None
if CROP:  # keep the crop's aspect ratio, width rounded to a multiple of the 14 px patch
    W = round(H * (CROP[1] - CROP[0]) * 16 / ((CROP[3] - CROP[2]) * 9) / 14) * 14
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

model = torch.hub.load("facebookresearch/dinov2", args.model).cuda().eval().half()


def frames_per_second(path):
    """Yield (second, RGB uint8 array) for the first frame at or after t + 0.5 s, for each second t."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        target = 0
        for frame in container.decode(stream):
            if frame.time is None or frame.time < target + 0.5:
                continue
            if CROP:  # crop at full resolution, resize on the GPU in embed()
                img = frame.to_ndarray(format="rgb24")
                h, w = img.shape[:2]
                img = img[int(CROP[2] * h):int(CROP[3] * h), int(CROP[0] * w):int(CROP[1] * w)]
            else:
                img = frame.to_ndarray(format="rgb24", width=W, height=H)
            yield target, img
            target = int(frame.time - 0.5) + 1


@torch.no_grad()
def embed(batch):
    x = torch.from_numpy(np.stack(batch)).cuda().permute(0, 3, 1, 2).half() / 255
    if x.shape[-2:] != (H, W):
        x = torch.nn.functional.interpolate(x.float(), size=(H, W), mode="bilinear", antialias=True).half()
    out = model.forward_features((x - MEAN.half()) / STD.half())
    return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], 1).float().cpu().numpy()


only = set(args.only.split(",")) - {""}
videos = sorted(p for p in args.video_dir.glob("*_360p.mp4")
                if not only or p.name.removesuffix("_360p.mp4") in only)
for path in videos:
    rid = path.name.removesuffix("_360p.mp4")
    out = args.out_dir / f"{rid}_360p.mp4_1s_1s.npz"
    if out.exists():
        continue
    t0 = time.time()
    seconds, feats, batch = [], [], []
    for sec, img in frames_per_second(path):
        seconds.append(sec)
        batch.append(img)
        if len(batch) == args.batch:
            feats.append(embed(batch))
            batch = []
    if batch:
        feats.append(embed(batch))
    feats, seconds = np.concatenate(feats), np.array(seconds)

    # One row per second; seconds with no decoded frame (rare) repeat the previous row.
    rows = np.zeros((seconds[-1] + 1, feats.shape[1]), dtype=np.float32)
    filled = np.zeros(len(rows), dtype=bool)
    rows[seconds], filled[seconds] = feats, True
    for t in np.flatnonzero(~filled):
        rows[t] = rows[t - 1]
    np.savez(out, rows.astype(np.float16))
    print(f"{rid}: {rows.shape}, {(~filled).sum()} filled gaps, {time.time() - t0:.0f}s", flush=True)
