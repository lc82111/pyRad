set -ex

  #--is_check_ray_idx_order \
  #--optimization_continue
python ./codes/columnGen_and_apertureRefine.py \
  --patient_ID Chest_original_Pa26Plan12Rx14GPU \
  --exp_name 1101_CGAR_Chest \
  --nb_apertures 10 \
  --plateau_patience 50 \
  --device cpu \
