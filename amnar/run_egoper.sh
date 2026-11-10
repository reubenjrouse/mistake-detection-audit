# AMNAR on EgoPER, retrained per recipe with the same two-stage recipe as run_cc4d.sh, then tested.
# The predicted segments go to <out>/<task>_ours/ for audit/published_protocol.py --amnar <out>.
# Run from the patched AMNAR folder (with amnar/configs copied into configs/).
# Usage: bash run_egoper.sh <out dir>
OUT=$1; PY=${PY:-python}
mkdir -p logs
for t in coffee oatmeal pinwheels quesadilla tea; do
  D=ckpt/EgoPER/ours_${t}_rebuild_e60
  $PY train.py configs/EgoPER/ours_${t}_aod.yaml --output af200 > logs/egoper_${t}_aod.txt 2>&1 &&
  $PY train.py configs/EgoPER/ours_${t}_rebuild.yaml --output e60 \
    --resume ckpt/EgoPER/ours_${t}_aod_af200/epoch_205.pth.tar > logs/egoper_${t}_rebuild.txt 2>&1 &&
  $PY test.py configs/EgoPER/ours_${t}_rebuild.yaml $D > logs/egoper_${t}_test.txt 2>&1 &&
  $PY test_ed.py configs/EgoPER/ours_${t}_rebuild.yaml $D > logs/egoper_${t}_test_ed.txt 2>&1 || { echo "$t failed"; continue; }
  mkdir -p $OUT/${t}_ours && cp $D/pred_seg_results_0.00.pkl $D/pred_seg_results_1.00.pkl $OUT/${t}_ours/
  $PY metric_vis_multiprocess.py --task $t --dirname ours_${t}_rebuild_e60 -ed > logs/egoper_${t}_metric.txt 2>&1 &&
    grep EDA logs/egoper_${t}_metric.txt | tail -1
done
