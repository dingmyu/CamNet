#!/bin/sh
exp=res-psp50_val
EXP_DIR=exp/drivable/$exp
mkdir -p ${EXP_DIR}/result
now=$(date +"%Y%m%d_%H%M%S")

part=AD
numGPU=6
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --mpi=pmi2 --gres=gpu:$numGPU -n1 --ntasks-per-node=$numGPU --partition=$part --job-name=${exp} --kill-on-bad-exit=1 \
python eval_train.py \
  --data_root=/mnt/lustre/dingmingyu/Research/ICCV19/NNnet/data/sevenscenes/ \
  --val_list=/mnt/lustre/dingmingyu/Research/ICCV19/CamNet/preTrain/data_list/test.txt \
  --split=test \
  --layers=34 \
  --backbone=resnet \
  --crop_h=224 \
  --crop_w=224 \
  --scales 1.0 \
  --has_prediction=0 \
  --gpu 0 1 2 3 4 5\
  --model_path=/mnt/lustre/dingmingyu/Research/ICCV19/CamNet/Train/exp/drivable/res34_train/model/train_epoch_300.pth \
  --save_folder=${EXP_DIR}/result/epoch_50/val_scale/ms \
  2>&1 | tee ${EXP_DIR}/result/epoch_50-val-ms-$now.log