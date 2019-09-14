#!/bin/sh
exp=res34_train
EXP_DIR=exp/drivable/$exp
mkdir -p ${EXP_DIR}/model
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${EXP_DIR}
part=AD
numGPU=8
nodeGPU=8
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $part --gres=gpu:$nodeGPU -n$numGPU --ntasks-per-node=$nodeGPU --job-name=${exp} \
python -u train.py \
  --data_root=/mnt/lustre/dingmingyu/Research/ICCV19/NNnet/data/sevenscenes/ \
  --train_list=data_list/sample.list \
  --val_list=/mnt/lustre/dingmingyu/Research/ICCV19/NNnet/data/sevenscenes/db_all_med_hard_valid.txt \
  --layers=34 \
  --backbone=resnet \
  --port=12345 \
  --syncbn=1 \
  --crop_h=224 \
  --crop_w=224 \
  --base_lr=1e-2 \
  --epochs=300 \
  --start_epoch=1 \
  --batch_size=16 \
  --bn_group=8 \
  --save_step=10 \
  --save_path=${EXP_DIR}/model \
  --evaluate=0 \
  --ignore_label 255 \
  --workers 2 \
  --weight train_epoch_300.pth \
  2>&1 | tee ${EXP_DIR}/model/train-$now.log
