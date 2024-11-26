#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train_stage2.py \
--name word4per_stage2 \
--root_dir 'your_data_path' \
--img_aug \
--batch_size 128 \
--lr 1e-4 \
--optimizer AdamW \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id' \
--toword_loss 'text' \
--num_epoch 60
