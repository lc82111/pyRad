set -ex

  #--is_check_ray_idx_order \
python ./codes/columnGen_and_apertureRefine.py \
  --patient_ID NPC_hard \
  --exp_name 0928_Aperture_refine \
  --max_master_subproblem_iter 1 \
  --plateau_patience 200 \
  --device cpu \
  --optimization_continue

