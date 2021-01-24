set -ex

  #--is_check_ray_idx_order \
  # --patient_ID NPC_hard \
  # --exp_name 0928_Aperture_refine \
python ./pyRad/convert_optimizedSegMU_to_TPS.py \
  --patient_ID Chest_neuralDose_Pa34Plan30Rx31GPU \
  --exp_name 20210123 \
  --device cpu \
  --MCDose_shape 122,256,256 \

