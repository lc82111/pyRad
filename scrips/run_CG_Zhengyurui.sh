set -ex

  #--is_check_ray_idx_order \
  #--optimization_continue
  #--patient_ID Chest_original_Pa26Plan12Rx14GPU \
  #--exp_name 0106_CGAR_Chest \
  #--exp_name 20210114 \
  # --not_use_apertureRefine \
python3 ./pyRad/columnGen_apertureRefine_NeuralDose.py \
  --data_dir /mnt/ssd/tps_optimization/patients_data/Lung_Zhengyurui_neuralDose/pbmcDoses_npz_Interp \
  --ckpt_path './pyRad/neuralDose/lightning_logs/pretrained_UNet3D/epoch=249-val_loss=0.00003452.ckpt_interval.ckpt' \
  --net Unet3D \
  --norm_type 'GroupNorm' \
  --num_depth 64 \
  --net_output_shape 43,128,128 \
  --MCDose_shape 86,256,256 \
  --patient_ID Lung_Zhengyurui_neuralDose \
  --exp_name 20210218_noARND \
  --nb_apertures 8 \
  --master_steps 1500 \
  --learning_rate 0.01 \
  --plateau_patience 100 \
  --device cuda:0 \
  --not_use_NeuralDose \
  --logs_interval 100 \
