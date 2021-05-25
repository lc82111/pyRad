set -ex

  #--is_check_ray_idx_order \
  #--optimization_continue
  #--patient_ID Chest_original_Pa26Plan12Rx14GPU \
  #--exp_name 0106_CGAR_Chest \
  #--exp_name 20210114 \
  # 3090 --ckpt_path './pyRad/neuralDose/lightning_logs/pretrained_UNet3D/epoch=199-val_loss=0.00002720.ckpt_interval.ckpt' \
  # titan --ckpt_path './pyRad/neuralDose/lightning_logs/version_3/epoch=199-val_loss=0.00002720.ckpt_interval.ckpt' \
python3 ./pyRad/columnGen_apertureRefine_NeuralDose.py \
  --device cuda:0 \
  --data_dir /mnt/ssd/tps_optimization/patients_data/Lung_QuLi_skin/pbmcDoses_npz_Interp \
  --ckpt_path './pyRad/neuralDose/lightning_logs/version_3/epoch=199-val_loss=0.00002720.ckpt_interval.ckpt' \
  --net Unet3D \
  --norm_type 'GroupNorm' \
  --num_depth 64 \
  --net_output_shape 51,128,128 \
  --MCDose_shape 102,256,256 \
  --patient_ID Lung_QuLi_neuralDose \
  --exp_name test_noARND \
  --nb_apertures 15 \
  --master_steps 50000 \
  --learning_rate 0.1 \
  --plateau_patience 20000 \
  --not_use_apertureRefine \
  --not_use_NeuralDose \
