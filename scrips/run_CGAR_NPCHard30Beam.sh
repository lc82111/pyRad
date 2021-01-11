set -ex

  #--is_check_ray_idx_order \
  # --patient_ID NPC_hard \
  # --exp_name 0928_Aperture_refine \

python ./codes/columnGen_and_apertureRefine.py \
  --patient_ID NPC_hard_30Beam \
  --exp_name 1009_CGAR \
  --max_master_subproblem_iter 15 \
  --plateau_patience 200 \
  --device cuda \

