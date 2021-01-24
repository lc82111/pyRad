set -ex

  # --mcpbDose2npz
  # --mcpbDose2npz_Interp
  # --npz2h5
  # --mcpbDose2npz_noRotation_noInterp
  # --test_pbmcDoses
  # --test_mcDose \
  # --Calculate_MC_unit_doses
  # --patient_ID Chest_Pa26Plan12Rx14GPU \
  # --MCDose_shape 126,256,256 \
  # --mcpbDose2npz_Interp
python ./pyRad/generate_trainingset.py \
  --test_pbmcDoses \
  --patient_ID  Lung_LvJiCheng_Pa38Plan30Rx31GPU_skin \
  --exp_name tmp \
  --MCDose_shape 122,256,256 \
  --net_output_shape 61,128,128 
  --winServer_nb_threads 12 \
