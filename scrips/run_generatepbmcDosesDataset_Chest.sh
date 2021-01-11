set -ex

  # --mcpbDose2npz
  # --mcpbDose2npz_Interp
  # --npz2h5
  # --mcpbDose2npz_noRotation_noInterp
  # --test_pbmcDoses
  # --test_mcDose
  # --Calculate_MC_unit_doses
  # --patient_ID Chest_Pa26Plan12Rx14GPU \
  # --MCDose_shape 126,256,256 \
python ./codes/generate_trainingset.py \
  --patient_ID Chest_skin_Pa34Plan30Rx31GPU \
  --exp_name 1101_CGAR_Chest \
  --MCDose_shape 122,256,256 \
  --winServer_nb_threads 12 \
  --mcpbDose2npz_Interp
