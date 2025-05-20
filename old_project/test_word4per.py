from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader,build_dataloader_toword
from processor.processor import do_inference,do_inference_toword
from utils.checkpoint import Checkpointer,Checkpointer_Toword
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs
from model.word4per import IM2TEXT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Word4Per Test")
    parser.add_argument("--config_file", default='/home/ldl/ReID/Com_Reid/logs/CUHK-PEDES/20230918_193706_iira/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('Word4Per', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    _, _, num_classes = build_dataloader(args)
    test_query_loader, test_gallery_loader = build_dataloader_toword(args)
    model = build_model(args, num_classes=num_classes)
    if args.pretrain_choice == 'ViT-L/14':
        img2text = IM2TEXT(embed_dim=768, 
                        middle_dim=512, 
                        output_dim=768,
                        n_layer=args.mlp_depth)
    else:
        img2text = IM2TEXT(embed_dim=512, 
                        middle_dim=512, 
                        output_dim=512,
                        n_layer=args.mlp_depth)
    
    checkpointer = Checkpointer_Toword(model,img2text)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))

    model.to(device)
    img2text.to(device)
    do_inference_toword(model, img2text, test_query_loader, test_gallery_loader)

    