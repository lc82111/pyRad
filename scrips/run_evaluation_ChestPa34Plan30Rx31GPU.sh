set -ex

  #--patient_ID Pa14Plan53Rx53GPU_3 \
  #--exp_name 0925_15Aperture_refine \
  #--MCPlan \
  #--CGDeposPlan_doseScale 1.0 \

  #--patient_ID Cervic_30Beam\
  #--exp_name 0927_5Aperture_refine \
  #--organ_filter PTV-smallupper PGTV-plan R1.5 Extended_PTV peripheral_tissue \
  #--TPSFluenceOptimPlan \
  #--cal_gamma \

python ./pyRad/evaluation.py \
  --cal_gamma \
  --NeuralDosePlan \
  --MCPlan \
  --patient_ID Chest_neuralDose_Pa34Plan30Rx31GPU \
  --net Unet3D \
  --data_dir /mnt/ssd/tps_optimization/patients_data/Chest_skin_Pa34Plan30Rx31GPU/pbmcDoses_npz_Interp/ \
  --ckpt_path './pyRad/neuralDose/lightning_logs/version_2/epoch=499-val_loss=0.00003918.ckpt_interval.ckpt' \
  --norm_type 'GroupNorm' \
  --num_depth 64 \
  --exp_name 20210123 \
  --MCDose_shape 122,256,256 \
  --net_output_shape 61,128,128 \
  --nb_apertures 5 \
  --consider_organs PTV Cord Stomach-PTV R0.5 \
  --device cpu \
