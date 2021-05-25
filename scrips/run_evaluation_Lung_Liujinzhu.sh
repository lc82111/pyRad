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

  #--NeuralDosePlan \
  #--neuralDoseMCPlan \
  #--originalMCPlan \
  #--originalPBPlan \
  #--gamma_plot_original \
  #--gamma_plot_neuralDose \
  #--NeuralDosePlan \
  #--PBPlan \

python ./pyRad/evaluation.py \
  --NeuralDosePlan \
  --patient_ID Lung_Liujinzhu_neuralDose \
  --net Unet3D \
  --data_dir /mnt/ssd/tps_optimization/patients_data/Lung_Liujinzhu_skin/pbmcDoses_npz_Interp \
  --ckpt_path './pyRad/neuralDose/lightning_logs/version_7/epoch=249-val_loss=0.00003012.ckpt_interval.ckpt' \
  --norm_type 'GroupNorm' \
  --num_depth 64 \
  --exp_name 20210204 \
  --MCDose_shape 102,256,256 \
  --net_output_shape 51,128,128 \
  --consider_organs 'PTV' 'Lung-L' 'RING2' \
  --PTV_name 'PTV' \
  --device cpu \
