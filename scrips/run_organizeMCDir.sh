set -ex

  #--mcpbDose2npz
  #--npz2h5
  # --mcpbDose2npz_noRotation_noInterp
  #--test_pbmcDoses
  # --test_mcDose
  # --Calculate_MC_unit_doses
  # --patient_ID Chest_Pa26Plan12Rx14GPU \
  # --MCDose_shape 126,256,256 \
  # --patient_ID Chest_skin_Pa34Plan30Rx31GPU \
python ./codes/organize_MontelCarlo_dir.py \
  --patient_ID Chest_skin_Pa34Plan30Rx31GPU \
  --exp_name tmp \
  --MCDose_shape 122,256,256 \
