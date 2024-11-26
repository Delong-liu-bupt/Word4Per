from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader,build_dataloader_toword
from processor.processor import do_inference, do_inference_toword, do_inference_fuse_p2w
from utils.checkpoint import Checkpointer,Checkpointer_Toword
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs
from model.word4per import IM2TEXT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Word4Per Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/word4per/configs.yaml')
    parser.add_argument("--model2_file", default='logs/CUHK-PEDES/word4per/best.pth')
    args = parser.parse_args()
    model2_file = args.model2_file
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('Word4Per', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    _, _, num_classes = build_dataloader(args)
    test_query_loader, test_gallery_loader = build_dataloader_toword(args)
    model = build_model(args, num_classes=num_classes)
    if args.pretrain_choice == 'ViT-L/14':
        img2text_text = IM2TEXT(embed_dim=768, 
                        middle_dim=512, 
                        output_dim=768,
                        n_layer=args.mlp_depth)

        img2text_img = IM2TEXT(embed_dim=768, 
                        middle_dim=512, 
                        output_dim=768,
                        n_layer=args.mlp_depth)
    else:
        img2text_text = IM2TEXT(embed_dim=512, 
                        middle_dim=512, 
                        output_dim=512,
                        n_layer=args.mlp_depth)

        img2text_img = IM2TEXT(embed_dim=512, 
                        middle_dim=512, 
                        output_dim=512,
                        n_layer=args.mlp_depth)
    
    checkpointer = Checkpointer_Toword(model,img2text_text)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))

    com_img_model = torch.load(model2_file, map_location='cuda')
    sd_img2text = com_img_model['img2text']
    if next(iter(sd_img2text.items()))[0].startswith('module'):
        sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
    img2text_img.load_state_dict(sd_img2text)

    model.to(device)
    img2text_text.to(device)
    img2text_img.to(device)
    do_inference_fuse_p2w(model, img2text_text, img2text_img, test_query_loader, test_gallery_loader)