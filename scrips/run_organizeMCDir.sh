set -ex

  #--patient_ID  Lung_Zhengyurui_skin \
python ./pyRad/organize_MontelCarlo_dir.py \
  --patient_ID Lung_Liujinzhu_skin \
  --exp_name tmp \
  --MCDose_shape 102,256,256 \
