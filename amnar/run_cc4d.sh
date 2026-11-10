# AMNAR on CaptainCook4D: 5 runs (DINOv2-S; Omnivore with seeds 0-2; SlowFast), each trained in two stages,
# tested, and turned into per-step scores for audit/audit.py. Paper-style metrics go to logs/.
# Run from the patched AMNAR folder (with amnar/configs copied into configs/).
# Usage: bash run_cc4d.sh <this repo> <steps table> <scores out dir>
REPO=$1; TABLE=$2; OUT=$3; PY=${PY:-python}
mkdir -p logs $OUT
for run in dinov2:dinov2_s:768:0 omnivore_s0:omnivore:1024:0 omnivore_s1:omnivore:1024:1 omnivore_s2:omnivore:1024:2 slowfast:slowfast:400:0; do
  IFS=: read name feat dim seed <<< "$run"
  for stage in aod rebuild; do
    { echo "init_rand_seed: $seed"; sed -e "s/features_subdir: dinov2_s/features_subdir: $feat/" -e "s/input_dim: 768/input_dim: $dim/" \
      configs/CaptainCook4D/cc4d_$stage.yaml; } > configs/CaptainCook4D/cc4d_${stage}_$name.yaml
  done
  D=ckpt/CaptainCook4D/cc4d_rebuild_${name}_e60
  $PY train.py configs/CaptainCook4D/cc4d_aod_$name.yaml --output af200 > logs/cc4d_${name}_aod.txt 2>&1 &&
  $PY train.py configs/CaptainCook4D/cc4d_rebuild_$name.yaml --output e60 \
    --resume ckpt/CaptainCook4D/cc4d_aod_${name}_af200/epoch_205.pth.tar > logs/cc4d_${name}_rebuild.txt 2>&1 &&
  $PY test.py configs/CaptainCook4D/cc4d_rebuild_$name.yaml $D > logs/cc4d_${name}_test.txt 2>&1 &&
  $PY test_ed.py configs/CaptainCook4D/cc4d_rebuild_$name.yaml $D > logs/cc4d_${name}_test_ed.txt 2>&1 || { echo "$name failed"; continue; }
  $PY $REPO/audit/amnar_scores.py $D $TABLE $OUT/amnar_$name.npz --name amnar_$name
  # metric_vis_multiprocess.py needs actseg_src from github.com/robert80203/EgoPER_official on PYTHONPATH
  $PY metric_vis_multiprocess.py --dataset CaptainCook4D --task 0 --fps 1 -ed -p 0 --dirname cc4d_rebuild_${name}_e60 \
    > logs/cc4d_${name}_metric.txt 2>&1 && grep EDA logs/cc4d_${name}_metric.txt | tail -1
done
