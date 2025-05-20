#!/bin/bash
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=1 \
python train_stage2.py \
--name word4per \
--root_dir '/mnt/cache/liudelong/data/datasets' \
--img_aug \
--batch_size 128 \
--MLM \
--lr 1e-4 \
--optimizer AdamW \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+id+mlm' \
--toword_loss 'text' \
--num_epoch 60