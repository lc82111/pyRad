set -ex

  #--patient_ID  Lung_Zhengyurui_skin \
python ./pyRad/organize_MontelCarlo_dir.py \
  --patient_ID Lung_Zhengyurui_skin \
  --exp_name tmp \
  --MCDose_shape 86,256,256 \
