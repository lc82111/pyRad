set -ex

python ./MU_MonteCarloRefine.py \ 
  --eval \
  --exp_name 0926_15Aperture_MUMCRefine \
  --optimized_segments_MUs_file ./results/0902_aperture_refine/ \
  --Calculate_MC_unit_doses \
  --patient_ID Pa14Plan53Rx53GPU_2 \
