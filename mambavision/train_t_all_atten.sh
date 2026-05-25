#!/bin/bash

MODEL=mamba_vision_T
DATA_PATH_TRAIN="/home/lqz25zhj/data/ImageNet1k/train"
DATA_PATH_VAL="/home/lqz25zhj/data/ImageNet1k/val"
BS=256  # per-GPU batch size(128) * 2
EXP=t_all_atten
LR=0.0025  # 按比例缩放: 0.005 × (512/1024)
WD=0.05
DR=0.2

CUDA_VISIBLE_DEVICES=5,2 torchrun --nproc_per_node=2 --master_port=12345 train.py \
--data_dir /home/lqz25zhj/data/ImageNet1k \
--input-size 3 224 224 \
--crop-pct=0.875 \
--train-split=$DATA_PATH_TRAIN \
--val-split=$DATA_PATH_VAL \
--model $MODEL \
--amp \
--weight-decay ${WD} \
--drop-path ${DR} \
--batch-size $BS \
--tag $EXP \
--lr $LR \
--opt lamb \
--sched cosine \
--warmup-epochs 20 \
--epochs 310 \
--model-ema \
--channels-last