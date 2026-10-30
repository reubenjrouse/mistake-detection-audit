"""Blind baselines on the exact mistake-detection protocol of EgoPED / AMNAR / AEM.

Reads AMNAR's processed annotations (attachments/ in github.com/iSEE-Laboratory/AMNAR).
  CaptainCook4D: train = all normal recordings, test = all error recordings (one evaluation).
  HoloAssist: per task, train = train split, test = val split; metrics averaged over tasks.
Background segments count as normal frames and are never flagged (like those methods). Scores use
only the annotations (no video) and the ground-truth segments ("oracle segmentation"; the published
methods use ActionFormer segments). --jitter moves segment boundaries to imitate a segmentation model.

Metrics copy AMNAR's metric_vis_multiprocess.py: a segment is flagged when its score > threshold for
thresholds -2.0..2.0 step 0.1; micro AUC pools frames over videos, macro AUC averages per-video
TPR/FPR; the AUC sorts FPRs and TPRs separately and appends (1, 1); EDA is the mean over thresholds
of their segment-level accuracy (background segments included). Standard ROC AUCs (frame and step
level) are printed for reference.

Usage: python published_protocol.py D:/third_party/AMNAR/attachments [--dataset holoassist] [--jitter 5]
       python published_protocol.py D:/data/egoper/EgoPER --dataset egoper --amnar D:/data/amnar_egoper_outputs --export <prefix>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_auc_score

from matching import match_pairs, pair_metrics
from safe_pickle import load as load_safe

PUBLISHED = {"cc4d": "AUC: EgoPED 54.9, AMNAR 60.2, AEM 62.5   EDA: 69.8 / 72.3 / 71.9",
             "holoassist": "AUC: EgoPED 50.6, AMNAR 56.5   EDA: 49.0 / 69.9  (average of noun and verb)",
             "egoper": "AUC: EgoPED 54.9, AMNAR 68.5   EDA: 43.0 / 64.4  (average over the 5 tasks)"}

parser = argparse.ArgumentParser()
parser.add_argument("attachments", type=Path, help="AMNAR attachments/, or for egoper the folder with annotation.json")
parser.add_argument("--dataset", default="cc4d", choices=["cc4d", "holoassist", "egoper"])
parser.add_argument("--amnar", type=Path, help="egoper: folder with AMNAR's released {task}_*/pred_seg_results_*.pkl;"
                    " adds their per-step score (max z over overlapping predicted segments) as 'amnar'")
parser.add_argument("--export", default="", help="write <prefix>_table/_blind/_methods.npz for fair_eval.py and "
                    "matched_eval.py (test steps of all tasks pooled)")
parser.add_argument("--jitter", type=float, default=0.0, help="seconds of boundary noise (std)")
parser.add_argument("--fps", type=float, default=10, help="frames per second of the segment annotations")
parser.add_argument("--matched", default="", help="also report AUROC on blind-matched test pairs with these "
                    "tolerances in SDs (duration, position, order, rec. steps, rec. length), e.g. 0.75,0.75,2,2,2")
args = parser.parse_args()


def load_cc4d():
    """{rid: [(start, end, step key or None for background, is_error)]}, one train/test pair."""
    root = args.attachments / "CC4D_process"

    def convert(recordings):
        # Steps are identified by recipe + step label: for error steps the description is the error text.
        return {rid: [(a, b, None if d == "background" else f"{rid.split('_')[1]}_{lab}", e == -1)
                      for (a, b), d, e, lab in zip(v["segments"], v["descriptions"], v["labels_error"], v["labels"])]
                for rid, v in recordings.items()}

    train = convert(json.load(open(root / "non_error_samples_processed.json"))["0"])
    test = convert(json.load(open(root / "error_samples_processed.json"))["0"])
    recipe = lambda rid: rid.split("_")[1]
    return {"all": (train, test, recipe)}


def load_holoassist():
    root = args.attachments / "HoloAssist"
    ann = json.load(open(root / "holoassist_processed_annotations_v.json"))
    tasks = {}
    for task, recordings in ann.items():
        split = {s: set((root / "data-task-splits" / s / f"{task}.txt").read_text().split())
                 for s in ["train", "val"]}
        convert = lambda rids: {rid: [(a, b, None if lab == "bg" else vn, e == -1)
                                      for (a, b), lab, vn, e in zip(v["segments"], v["labels"], v["vn_pairs"],
                                                                    v["labels_error"])]
                                for rid, v in recordings.items() if rid in rids}
        train, test = convert(split["train"]), convert(split["val"])
        if train and test:
            tasks[task] = (train, test, lambda rid: "task")
    return tasks


def load_egoper():
    """Per task: test = the videos in AMNAR's released test outputs, train = the other normal videos.
    Segments are in seconds; frames at --fps (10, like the I3D features)."""
    ann = json.load(open(args.attachments / "annotation.json"))
    tasks = {}
    for task, data in ann.items():
        test_ids = set(load_safe(next(args.amnar.glob(f"{task}_*")) / "pred_seg_results_0.00.pkl"))
        recordings = {}
        for v in data["segments"]:
            lab = v["labels"]
            recordings[v["video_id"]] = [
                (int(round(a * args.fps)), int(round(b * args.fps)), None if act == 0 else f"{task}_{act}", typ != 0,
                 typ) for (a, b), act, typ in zip(lab["time_stamp"], lab["action"], lab["action_type"])]
        train = {r: s for r, s in recordings.items() if r not in test_ids and "normal" in r}
        test = {r: s for r, s in recordings.items() if r in test_ids}
        tasks[task] = (train, test, lambda rid: rid.split("_")[0])
    types = {i: n for n, i in next(iter(ann.values()))["actiontype2idx"].items()}
    return tasks, types


def amnar_step_scores(task, test):
    """Max z = (distance - mean) / std over AMNAR's predicted segments overlapping each test step."""
    d = next(args.amnar.glob(f"{task}_*"))
    res0, res1 = load_safe(d / "pred_seg_results_0.00.pkl"), load_safe(d / "pred_seg_results_1.00.pkl")
    q = 0.85 if 0.85 in next(iter(res0.values()))["diff_and_threshold"] else max(next(iter(res0.values()))["diff_and_threshold"])
    out = []
    for rid, segs in test.items():
        d0, d1 = res0[rid]["diff_and_threshold"][q], res1[rid]["diff_and_threshold"][q]
        dist, mean = np.array([float(x[0]) for x in d0]), np.array([float(x[1]) for x in d0])
        std = np.array([float(x[1]) for x in d1]) - mean
        z = np.where(std > 0, (dist - mean) / np.where(std > 0, std, 1), np.nan)
        pred = np.asarray(res0[rid]["segments"], float).reshape(-1, 2)[:len(z)]
        for a, b, key, *_ in segs:
            if key is not None:
                hit = (pred[:, 0] <= b) & (pred[:, 1] >= a) & np.isfinite(z)
                out.append(z[hit].max() if hit.any() else np.nan)
    out = np.array(out)
    return np.where(np.isnan(out), np.nanmin(out) - 1, out)


def steps_of(recordings, recipe):
    """One row per step segment: recording, recipe, step key, frames, relative position, previous step."""
    rows = []
    for rid, segs in recordings.items():
        steps = [s for s in segs if s[2] is not None]
        prev = "<start>"
        for i, (a, b, key, err, *typ) in enumerate(steps):
            rows.append(dict(rec=rid, recipe=recipe(rid), step=key, frames=max(b - a + 1, 1),
                             pos=i / max(len(steps) - 1, 1), prev=prev, n_steps=len(steps),
                             rec_frames=segs[-1][1], error=err, first=i == 0, start=a, end=b,
                             type=typ[0] if typ else int(err)))
            prev = key
    return rows


col = lambda rows, k: np.array([r[k] for r in rows])


def blind_scores(tr, te):
    """Each blind score as (train values, test values); stats come from correct training steps."""
    fit = [r for r in tr if not r["error"]]

    def abs_z(key, value, transform=lambda x: x, per_recording=False):
        ref = [r for r in (tr if per_recording else fit) if not per_recording or r["first"]]
        ref_k, ref_v = col(ref, key), transform(col(ref, value).astype(float))
        stats = {k: (ref_v[ref_k == k].mean(), ref_v[ref_k == k].std() + 1e-6)
                 for k in np.unique(ref_k) if (ref_k == k).sum() >= 3}
        glob = (ref_v.mean(), ref_v.std() + 1e-6)
        out = []
        for rows in (tr, te):
            mu_sd = np.array([stats.get(x, glob) for x in col(rows, key)])
            out.append(np.abs(transform(col(rows, value).astype(float)) - mu_sd[:, 0]) / mu_sd[:, 1])
        return out

    counts = {}
    for r in fit:
        counts.setdefault(r["prev"], {}).setdefault(r["step"], 0)
        counts[r["prev"]][r["step"]] += 1
    transition = [np.array([-np.log((counts.get(r["prev"], {}).get(r["step"], 0) + 0.1)
                                    / (sum(counts.get(r["prev"], {}).values()) + 1)) for r in rows])
                  for rows in (tr, te)]
    scores = {
        "duration": abs_z("step", "frames", np.log),
        "position": abs_z("step", "pos"),
        "transition": transition,
        "rec_length": abs_z("recipe", "n_steps", per_recording=True),
        "rec_duration": abs_z("recipe", "rec_frames", np.log, per_recording=True),
    }
    zs = lambda pair: [(s - pair[0].mean()) / (pair[0].std() + 1e-6) for s in pair]
    scores = {k: zs(v) for k, v in scores.items()}
    scores["sum"] = zs([sum(v[i] for v in scores.values()) for i in (0, 1)])
    # Controls: every step gets the same score (only background vs step separates frames), no segment
    # is ever flagged, and "sum" at half the scale (the AUC only tries thresholds -2..2, so scale matters).
    scores["constant"] = [np.zeros(len(tr)), np.zeros(len(te))]
    scores["flag nothing"] = [np.full(len(tr), -np.inf), np.full(len(te), -np.inf)]
    scores["sum / 2"] = [s / 2 for s in scores["sum"]]
    return scores


def error_acc(pred, gt, gt_error):
    """AMNAR's segment-level accuracy ("loose version" in metric_vis_multiprocess.py), vectorized.

    pred / gt_error: -1 = error, otherwise normal; gt: frame step labels (a run of one label = one segment).
    Like the original loop, the first frame of each new run is counted toward the previous run, and a
    run's true error status is read from its first frame.
    """
    starts = np.concatenate([[0], np.flatnonzero(gt[1:] != gt[:-1]) + 1])
    ends = np.append(starts[1:], len(gt) - 1)  # inclusive: the next run's first frame counts here
    lo = np.append(0, starts[1:] + 1)
    err_cum = np.concatenate([[0], np.cumsum(pred == -1)])
    num_error = err_cum[ends + 1] - err_cum[lo]
    num_nonerror = (ends + 1 - lo) - num_error
    is_pred_error, is_gt_error = num_error > 0, gt_error[starts] == -1
    last = np.arange(len(starts)) == len(starts) - 1
    correct = ((is_pred_error & is_gt_error & (num_error > num_nonerror))
               | (~is_pred_error & ~is_gt_error)
               | ((num_error < num_nonerror) & (~is_gt_error | last)))
    return correct.sum(), len(starts)


def frame_arrays(test, score, jitter=0.0, seed=0):
    """Per test video: frame error labels, frame scores, frame step labels.

    jitter > 0 moves each step's start and end by N(0, jitter seconds); frames are scored by the moved
    segments while labels stay on the true ones. Background frames are never flagged.
    """
    rng = np.random.default_rng(seed)
    videos, i = [], 0
    for segs in test.values():
        n = segs[-1][1]
        gt, sc, lab = np.zeros(n, bool), np.full(n, -np.inf), np.zeros(n, int)
        for a, b, key, err, *_ in segs:
            gt[a:b + 1] = err
            lab[a:b + 1] = 0 if key is None else hash(key) % 10**9 + 1
            if key is not None:
                s, i = score[i], i + 1
                if jitter:
                    a, b = sorted(np.clip(np.round([a, b] + rng.normal(0, jitter * args.fps, 2)), 0, n - 1).astype(int))
                sc[a:b + 1] = np.maximum(sc[a:b + 1], s)
        videos.append((gt, sc, lab))
    return videos


def published_metrics(videos):
    micro, macro, eda = [], [], []
    for th in [i / 10 for i in range(-20, 21)]:
        tp = fp = pos = neg = 0
        tprs, fprs, correct, total = [], [], 0, 0
        for gt, sc, lab in videos:
            pred = sc > th
            v_tp, v_fp, v_pos, v_neg = (pred & gt).sum(), (pred & ~gt).sum(), gt.sum(), (~gt).sum()
            tp, fp, pos, neg = tp + v_tp, fp + v_fp, pos + v_pos, neg + v_neg
            tprs.append(v_tp / v_pos if v_pos else float(v_tp == 0))
            fprs.append(v_fp / v_neg if v_neg else float(v_fp == 0))
            c, t = error_acc(np.where(pred, -1, 1), lab, np.where(gt, -1, 1))
            correct, total = correct + c, total + t
        micro.append((fp / neg if neg else float(fp == 0), tp / pos if pos else float(tp == 0)))
        macro.append((np.mean(fprs), np.mean(tprs)))
        eda.append(correct / total)
    curve = lambda pts: auc(np.append(np.sort([p[0] for p in pts]), 1), np.append(np.sort([p[1] for p in pts]), 1))
    return curve(micro), curve(macro), np.mean(eda)


def evaluate(task, train, test, recipe):
    """Metrics per score name: published micro AUC, macro AUC, EDA, standard frame and step ROC AUC."""
    tr, te = steps_of(train, recipe), steps_of(test, recipe)
    y = col(te, "error")
    out, scores = {}, blind_scores(tr, te)
    if args.amnar:
        scores["amnar"] = [np.full(len(tr), np.nan), amnar_step_scores(task, test)]
    exported.append((te, scores))
    if args.matched:  # pair test error steps with test correct steps that look the same to the blind scores
        features = np.column_stack([scores[k][1] for k in ["duration", "position", "transition", "rec_length",
                                                         "rec_duration"]])
        tol = [float(v) for v in args.matched.split(",")]
        pairs = match_pairs(col(te, "recipe"), features, col(te, "rec"), y, ~y, tol)
        for name, (_, sc) in scores.items():
            matched_scores.setdefault(name, []).append((sc[pairs[:, 0]], sc[pairs[:, 1]]))
    for name, (_, s) in scores.items():
        videos = frame_arrays(test, s, args.jitter)
        gt = np.concatenate([v[0] for v in videos])
        sc = np.concatenate([v[1] for v in videos])
        finite = np.isfinite(s)
        sc = np.where(np.isinf(sc), (s[finite].min() if finite.any() else 0) - 1, sc)
        std_step = roc_auc_score(y, s) if finite.all() and s.std() and 0 < y.sum() < len(y) else 0.5
        out[name] = (*published_metrics(videos), roc_auc_score(gt, sc), std_step)
    return out, len(tr), len(te), y.sum()


results, matched_scores, exported = {}, {}, []
types = {0: "Normal", 1: "Error"}
if args.dataset == "egoper":
    datasets, types = load_egoper()
else:
    datasets = load_cc4d() if args.dataset == "cc4d" else load_holoassist()
for task, data in datasets.items():
    results[task], n_tr, n_te, n_err = evaluate(task, *data)
    print(f"{task}: {len(data[0])} train recordings ({n_tr} steps), {len(data[1])} test recordings "
          f"({n_te} steps, {n_err} errors)")

print(f"\n{args.dataset}, boundary jitter {args.jitter} s, averaged over {len(results)} task(s)")
print(f"published  {PUBLISHED[args.dataset]}")
print(f"{'blind score':<14} {'micro AUC':>9} {'macro AUC':>9} {'EDA':>6} {'std frame AUC':>14} {'std step AUC':>13}")
for name in next(iter(results.values())):
    m = np.nanmean([r[name] for r in results.values()], 0) * 100
    print(f"{name:<14} {m[0]:9.1f} {m[1]:9.1f} {m[2]:6.1f} {m[3]:14.1f} {m[4]:13.1f}")

if args.matched:
    n_pairs = sum(len(e) for e, _ in matched_scores["constant"])
    print(f"\nblind-matched test pairs (tolerances {args.matched} SD): {n_pairs} pairs, pooled over tasks")
    print(f"{'blind score':<14} {'AUROC [95% CI]':>22}")
    for name, parts in matched_scores.items():
        e = np.concatenate([p[0] for p in parts])
        c = np.concatenate([p[1] for p in parts])
        if not np.isfinite(np.r_[e, c]).all():
            continue
        (a, _), lo, hi = pair_metrics(e, c)
        print(f"{name:<14} {a:6.3f} [{lo[0]:.3f}, {hi[0]:.3f}]")

if args.export:  # the format of build_table / shortcuts.py / evaluate.py --scores_out, test steps only
    rows = [r for te, _ in exported for r in te]
    cats = np.array([types[k] for k in sorted(types) if k != 0])
    typ = col(rows, "type")
    np.savez(f"{args.export}_table.npz", recording_id=col(rows, "rec"), step_id=col(rows, "step"),
             activity_id=col(rows, "recipe"), start=col(rows, "start") / args.fps, end=col(rows, "end") / args.fps,
             categories=cats, errors=np.stack([typ == k for k in sorted(types) if k != 0], 1),
             recording_error=np.ones(len(rows), bool))
    blind_keys = ["duration", "position", "transition", "rec_length", "rec_duration"]
    pooled = lambda k: np.concatenate([s[k][1] for _, s in exported])
    np.savez(f"{args.export}_blind.npz", features=np.column_stack([pooled(k) for k in blind_keys]),
             **{k: pooled(k) for k in blind_keys + ["sum"]})
    methods = [k for k in exported[0][1] if k not in blind_keys + ["sum", "constant", "flag nothing", "sum / 2"]]
    np.savez(f"{args.export}_methods.npz", eval_idx=np.arange(len(rows)), **{f"{k}_seed0": pooled(k) for k in methods})
    print(f"\nwrote {args.export}_table/_blind/_methods.npz: {len(rows)} test steps, methods {methods}")
