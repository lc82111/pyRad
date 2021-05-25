set -ex

  #--patient_ID Pa14Plan53Rx53GPU_3 \
  #--exp_name 0925_15Aperture_refine \
  #--MCPlan \
  #--CGDeposPlan_doseScale 1.0 \

  #--patient_ID Cervic_30Beam\
  #--exp_name 0927_5Aperture_refine \
  #--organ_filter PTV-smallupper PGTV-plan R1.5 Extended_PTV peripheral_tissue \
  #--TPSFluenceOptimPlan \

  #--ckpt_path './pyRad/neuralDose/lightning_logs/version_4/epoch=347-val_loss=0.00003148.ckpt_interval.ckpt' \

  #--PBPlan
  #--exp_name 20210218_noARND \

python ./pyRad/evaluation.py \
  --NeuralDosePlan \
  --exp_name 202100202 \
  --patient_ID Lung_Zhengyurui_neuralDose \
  --net Unet3D \
  --data_dir /mnt/ssd/tps_optimization/patients_data/Lung_Zhengyurui_skin/pbmcDoses_npz_Interp \
  --ckpt_path './pyRad/neuralDose/lightning_logs/version_6/epoch=249-val_loss=0.00003452.ckpt_interval.ckpt' \
  --norm_type 'GroupNorm' \
  --num_depth 64 \
  --MCDose_shape 86,256,256 \
  --net_output_shape 43,128,128 \
  --nb_apertures 6 \
  --consider_organs 'PTV_plan' 'Esophagus' 'peripheral_tissue' 'Spinal canal' 'Lung_R' 'Lung_L' \
  --device cpu \
  --PTV_name 'PTV_plan' \
