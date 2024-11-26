#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train_stage1.py \
--name word4per_stage1 \
--root_dir 'your_data_path' \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id' \
--num_epoch 60