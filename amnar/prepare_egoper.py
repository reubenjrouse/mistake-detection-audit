"""Lay out EgoPER for AMNAR: {task}/features_10fps_new/*.npy (the released I3D features) and split lists.
test = the videos in AMNAR's released test outputs; training = all other (normal) videos."""
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from safe_pickle import load

src, out, outputs = Path("D:/data/egoper/EgoPER"), Path("D:/data/amnar_egoper"), Path("D:/data/amnar_egoper_outputs")
ann = json.load(open(src / "annotation.json"))
for name in ["annotation.json", "active_object.json"]:
    shutil.copy(src / name, out / name)
for task in ann:
    d = out / task
    feats = d / "features_10fps_new"
    feats.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src / task.capitalize() / f"{task}_i3d_2048.zip") as z:
        for m in z.namelist():
            if m.endswith(".npy"):
                (feats / Path(m).name).write_bytes(z.read(m))
    test = sorted(load(next(outputs.glob(f"{task}_*")) / "pred_seg_results_0.00.pkl"))
    ids = [s["video_id"] for s in ann[task]["segments"]]
    train = sorted(v for v in ids if v not in test and (feats / f"{v}.npy").exists())
    (d / "training.txt").write_text("\n".join(train))
    (d / "validation.txt").write_text("\n".join(test))
    (d / "test.txt").write_text("\n".join(test))
    print(task, len(train), "train", len(test), "test", len(list(feats.glob("*.npy"))), "feature files")
