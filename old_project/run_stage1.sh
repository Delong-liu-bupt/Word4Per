#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=1 \
python train_stage1.py \
--name word4per \
--root_dir 'your_data_path' \
--img_aug \
--batch_size 64 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id' \
--num_epoch 60