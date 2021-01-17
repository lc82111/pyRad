set -ex

  #--patient_ID Pa14Plan53Rx53GPU_3 \
  #--exp_name 0925_15Aperture_refine \
  #--MCPlan \
  #--CGDeposPlan_doseScale 1.0 \

  #--patient_ID Cervic_30Beam\
  #--exp_name 0927_5Aperture_refine \
  #--organ_filter PTV-smallupper PGTV-plan R1.5 Extended_PTV peripheral_tissue \

python evaluation.py \
  --patient_ID Chest_original_Pa34Plan30Rx31GPU \
  --exp_name 20210114 \
  --MCDose_shape 122,256,256 \
  --NeuralDose \
  --TPSFluenceOptimPlan \
  --organ_filter PTV.1 PTV.2 PTV.3 R2.5 R1.5 R0.5 applicator1 applicator2 Stomach-PTV.1 Stomach-PTV.2 peripheral_tissue peripheral_tissue.1 \
