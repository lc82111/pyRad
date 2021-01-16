set -ex

  #--is_check_ray_idx_order \
  #--optimization_continue
  #--patient_ID Chest_original_Pa26Plan12Rx14GPU \
  #--exp_name 0106_CGAR_Chest \
python3 ./columnGen_apertureRefine_NeuralDose.py \
  --data_dir /mnt/ssd/tps_optimization/Chest_skin_Pa34Plan30Rx31GPU/pbmcDoses_npz_Interp/ \
  --ckpt_path './neuralDose/lightning_logs/version_2/epoch=499-val_loss=0.00003918.ckpt_interval.ckpt' \
  --net Unet3D \
  --norm_type 'GroupNorm' \
  --num_depth 64 \
  --net_output_shape 61,128,128 \
  --patient_ID Chest_original_Pa34Plan30Rx31GPU \
  --MCDose_shape 122,256,256 \
  --exp_name 20210114 \
  --nb_apertures 5 \
  --master_steps 1500 \
  --learning_rate 0.01 \
  --plateau_patience 30 \
  --device cuda:0 \
