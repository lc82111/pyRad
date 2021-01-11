set -ex

python ./codes/aperture_shape_refine.py \
  --patient_ID Cervic_30Beam\
  --exp_name 0927_5Aperture_refine \
  --is_check_ray_idx_order \
  --max_master_subproblem_iter 8 \
  --plateau_patience 200 \
