set -ex

  #--is_check_ray_idx_order \
  #--optimization_continue
  #--patient_ID Chest_original_Pa26Plan12Rx14GPU \
  #--exp_name 0106_CGAR_Chest \
  #--exp_name 20210114 \
python3 ./pyRad/columnGen_apertureRefine_NeuralDose.py \
  --not_use_apertureRefine \
  --not_use_NeuralDose \
  --data_dir /mnt/ssd/tps_optimization/patients_data/Lung_Liujinzhu_neuralDose/pbmcDoses_npz_Interp \
  --ckpt_path './pyRad/neuralDose/lightning_logs/pretrained_UNet3D/epoch=249-val_loss=0.00003012.ckpt_interval.ckpt' \
  --net Unet3D \
  --norm_type 'GroupNorm' \
  --num_depth 64 \
  --net_output_shape 51,128,128 \
  --MCDose_shape 102,256,256 \
  --patient_ID Lung_Liujinzhu_neuralDose \
  --exp_name 20210221_noARND \
  --nb_apertures 5 \
  --master_steps 3000 \
  --learning_rate 0.01 \
  --plateau_patience 300 \
  --device cuda:0 \
