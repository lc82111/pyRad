set -ex

  #--is_check_ray_idx_order \
  #--optimization_continue
  #--patient_ID Chest_original_Pa26Plan12Rx14GPU \
  #--exp_name 0106_CGAR_Chest \
  #--exp_name 20210114 \
  # --device cuda:0 \
python3 ./pyRad/columnGen_apertureRefine_NeuralDose.py \
  --data_dir /mnt/ssd/tps_optimization/patients_data/Lung_LvJiCheng_Pa38Plan30Rx31GPU_skin/pbmcDoses_npz_Interp \
  --ckpt_path './pyRad/neuralDose/lightning_logs/version_2/epoch=347-val_loss=0.00003148.ckpt_interval.ckpt' \
  --net Unet3D \
  --norm_type 'GroupNorm' \
  --num_depth 64 \
  --net_output_shape 61,128,128 \
  --patient_ID Lung_LvJiCheng_Pa38Plan30Rx31GPU_neuralDose \
  --MCDose_shape 122,256,256 \
  --exp_name 20210219 \
  --nb_apertures 5 \
  --master_steps 1500 \
  --learning_rate 0.01 \
  --plateau_patience 30 \
  --device cpu \
  --logs_interval 100 \
  --not_use_NeuralDose \
  --not_use_apertureRefine \
