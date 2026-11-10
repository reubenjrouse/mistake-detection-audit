"""Download CaptainCook4D GoPro 360p videos, resumable.

Links come from the official downloader repo (metadata/download_links.json).
Files already downloaded with the right size are skipped.

Usage: python prepare/download_videos.py C:/Users/Reuben/data/cc4d/downloader/metadata/download_links.json D:/cc4d_videos [--only 1_7,2_4]
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("links", type=Path)
parser.add_argument("out_dir", type=Path)
parser.add_argument("--only", default="", help="comma-separated recording ids")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

links = json.load(open(args.links))
only = set(args.only.split(",")) - {""}
jobs = [(rid, d["gopro_360p"]) for rid, d in links.items()
        if (not only or rid in only) and d.get("gopro_360p")]
no_gopro = sorted(rid for rid, d in links.items() if (not only or rid in only) and not d.get("gopro_360p"))


def download(job):
    rid, url = job
    out = args.out_dir / f"{rid}_360p.mp4"
    with urllib.request.urlopen(url, timeout=60) as resp:
        size = int(resp.headers["Content-Length"])
        if out.exists() and out.stat().st_size == size:
            return f"{rid}: already done"
        part = out.with_suffix(".part")
        with open(part, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    if part.stat().st_size != size:
        raise IOError(f"{rid}: got {part.stat().st_size} of {size} bytes")
    part.replace(out)
    return f"{rid}: {size / 1e6:.0f} MB"


print(f"{len(jobs)} videos to fetch, {len(no_gopro)} recordings have no GoPro 360p: {no_gopro}")
failed = []
with ThreadPoolExecutor(args.workers) as pool:
    futures = {pool.submit(download, j): j[0] for j in jobs}
    for i, fut in enumerate(futures, 1):
        try:
            print(f"[{i}/{len(jobs)}] {fut.result()}", flush=True)
        except Exception as e:
            failed.append(futures[fut])
            print(f"[{i}/{len(jobs)}] FAILED {futures[fut]}: {e}", flush=True)
print(f"finished: {len(jobs) - len(failed)} ok, {len(failed)} failed {failed}")
