# Mistake detection audit

The standard scoring for offline mistake detection in procedural video (EDA and threshold-based frame AUC, used by EgoPED, AMNAR and AEM) rewards models that never look at the video. This repo shows that, adds a test that only rewards seeing the mistake, and re-scores AMNAR (CVPR 2025) with it.

## Findings

**Trivial scores on the standard scoring** ([results/published_protocol_*.txt](results/)):

| | CaptainCook4D AUC / EDA | EgoPER AUC / EDA |
|---|---|---|
| Never flag a mistake | 51.4 / 71.3 | 54.8 / **86.9** |
| Same score for every step | **72.2** / 72.9 | 67.4 / 72.8 |
| Step duration only (no video) | **75.0** / **75.4** | 74.4 / 73.4 |
| AMNAR (published) | 60.2 / 72.3 | 68.5 / 64.4 |
| AEM (published) | 62.5 / 71.9 | 73.8 / 66.7 |

Our rows show macro AUC. Trivial scores use ground-truth step boundaries; published methods predict their own.

**Fair test.** Each error step is paired with a correct step from the same recipe whose duration, position, step order and recording length all match, so annotation-only scores sit at chance ([test_sets/](test_sets/)). Pair accuracy = how often the error step scores higher. *Gain over timing* = AUROC added to a regression on those timing scores.

| AMNAR | Pair accuracy (all errors) | Gain over timing |
|---|---|---|
| EgoPER, authors' outputs | 62.8% [56.7, 68.8], 231 pairs | +0.052 [+0.031, +0.072] |
| EgoPER, our retrained copy | 63.0% [56.7, 68.8] | +0.043 [+0.022, +0.062] |
| CaptainCook4D, our copy, 5 runs (3 feature types) | 48.2–55.8%, every CI includes 50%, 164 pairs | −0.000 to +0.001, every CI includes 0 |

No CaptainCook4D model was released. On the standard scoring our copy reaches 52.6 AUC (paper: 60.2), so the CaptainCook4D result is for our copy. Our own five one-class detectors are also at chance ([results/audit_cc4d.txt](results/audit_cc4d.txt), [results/detectors_person_test.log](results/detectors_person_test.log)).

## Layout

```
prepare/     download CaptainCook4D videos, extract DINOv2 features, build the per-step table
detectors/   our five one-class detectors and their evaluation
audit/       blind (timing-only) scores, the published protocol, the fair test (audit.py), AMNAR score export
amnar/       patch, configs and scripts to retrain AMNAR (github.com/iSEE-Laboratory/AMNAR, commit ae726f0)
test_sets/   timing-matched pairs
results/     outputs behind every number above
```

## Reproduce

Requires `pip install -r requirements.txt`. Data paths are hard-coded for our machine (`C:/Users/Reuben/data`, `D:/data`); change them first.

```bash
# CaptainCook4D steps table and our detectors
python prepare/download_videos.py <download_links.json> <videos dir>
python prepare/extract_dinov2.py <videos dir> <data>/features/dinov2_s
python prepare/build_steps.py <data>                     # -> <data>/steps_dinov2_s_w2.npz
python detectors/evaluate.py <table> --test --seeds 0,1,2,3,4 --out results/detectors_person_test.csv
python detectors/evaluate.py <table> --protocol normal_vs_error --seeds 0,1,2,3,4 --scores_out methods.npz
python audit/blind_scores.py <table> --scores_out blind.npz

# Standard scoring with trivial scores (AMNAR's processed annotations in its attachments/ folder)
python audit/published_protocol.py <AMNAR>/attachments                           # CaptainCook4D
python audit/published_protocol.py <AMNAR>/attachments --dataset holoassist
python audit/published_protocol.py <EgoPER> --dataset egoper --amnar <AMNAR outputs> --export egoper

# AMNAR: inside a clone at ae726f0
git apply amnar/patch.diff && cp -r amnar/configs/* configs/
python amnar/prepare_cc4d.py attachments <data>/features <out>
bash amnar/run_cc4d.sh <this repo> <table> <scores dir>
python amnar/prepare_egoper.py && bash amnar/run_egoper.sh <out>

# Fair test
python audit/audit.py <table> blind.npz methods.npz <scores dir>/amnar_*.npz --pairs_out test_sets/cc4d_pairs.csv
python audit/audit.py egoper_table.npz egoper_blind.npz egoper_methods.npz --pairs_out test_sets/egoper_pairs.csv
```
