set -ex

python ./codes/fluenceMap_optimization.py \
  --patient_ID Cervic_30Beam\
  --exp_name 0928_fluenceMapOptim \
  --is_check_ray_idx_order \
  --steps 5000 \
  --plateau_patience 500 \
  --device cuda \
