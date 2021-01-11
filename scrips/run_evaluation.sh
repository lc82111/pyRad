set -ex

  #--patient_ID Pa14Plan53Rx53GPU_3 \
  #--exp_name 0925_15Aperture_refine \
  #--MCPlan \

  #--patient_ID Cervic_30Beam\
  #--exp_name 0927_5Aperture_refine \
  #--organ_filter PTV-smallupper PGTV-plan R1.5 Extended_PTV peripheral_tissue \

python ./codes/evaluation.py \
  --patient_ID NPC_hard \
  --exp_name 0928_Aperture_refine \
  --CGDeposPlan \
  --CGDeposPlan_doseScale 1.075 \
  --TPSFluenceOptimPlan \
  --organ_filter PTV60PLAN PGTVnd PGTV R2.5 R1 Pituitary Parotid_L Parotid_R BrainStem SpinalCord \
