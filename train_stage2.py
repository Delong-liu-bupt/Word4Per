import sys
from model.build import build_model
from model.word4per import IM2TEXT
import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader, build_dataloader_toword
from processor.processor import do_train, do_train_toword
from utils.checkpoint import Checkpointer,Checkpointer_Toword
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator, Evaluator_toword
from utils.options import get_args
from utils.comm import get_rank, synchronize
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_text_features(model, token_features):
    text = tokenize("a photo of", SimpleTokenizer())
    text = text.to("cuda")
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    # print(text, token_features)
    text_features = model.encode_text_img(text, token_features)
    return text_features

if __name__ == '__main__':
    args = get_args()
    set_seed(1+get_rank())
    name = args.name

    if 'ViT-L' in args.pretrain_choice:
        img2text = IM2TEXT(embed_dim=768, 
                        middle_dim=512, 
                        output_dim=768,
                        n_layer=args.mlp_depth)

    elif 'ViT-B' in args.pretrain_choice:
        img2text = IM2TEXT(embed_dim=512, 
                        middle_dim=512, 
                        output_dim=512,
                        n_layer=args.mlp_depth)

        
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    device = "cuda"
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')
    logger = setup_logger('Word4Per', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(str(args).replace(',', '\n'))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader
    train_loader, _, _, num_classes = build_dataloader(args)
    test_query_loader, test_gallery_loader = build_dataloader_toword(args)
    model = build_model(args, num_classes)
    logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    img2text.to(device)
    if 'ViT-L' in args.pretrain_choice:
        base_model_path = '/data1/Com_ReID/model/base_model.pth'
    elif 'ViT-B' in args.pretrain_choice:
        base_model_path = '/data1/Com_ReID/model/base_model_vitb.pth'
    base_model = torch.load(base_model_path, map_location='cuda')
    sd_model = base_model['model']
    model.load_state_dict(sd_model, strict=False)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    optimizer = build_optimizer(args, img2text)
    scheduler = build_lr_scheduler(args, optimizer)

    is_master = get_rank() == 0
    checkpointer = Checkpointer_Toword(model, img2text, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator_toword(test_query_loader, test_gallery_loader)

    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']
    model.eval()
    img2text.train()

    do_train_toword(start_epoch, args, model, img2text, train_loader, evaluator, optimizer, scheduler, checkpointer)
