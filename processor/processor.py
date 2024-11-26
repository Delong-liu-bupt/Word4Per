import logging
import sys
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator, Evaluator_toword,Evaluator_fuse_p2w
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize
from model import objectives

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("Word4Per.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss', 0), batch_size)

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def get_text_features(model, token_features):
    text = tokenize("a photo of", SimpleTokenizer())
    text = text.to("cuda")
    text = text.view(1, -1)
    text = text.repeat(token_features.size(0), 1)
    # print(text, token_features)
    text_features = model.encode_text_img(text, token_features)
    return text_features

def do_train_toword(start_epoch, args, model, img2text, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("Word4Per.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "text_loss": AverageMeter(),
        "img_loss": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        
        img2text.train()


        for n_iter, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            images = batch['images']
            caption_ids = batch['caption_ids']

            person_ids = batch['pids']
            image_feats, text_feats = model.base_model(images, caption_ids)
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            token_features = img2text(i_feats)
            text_features = get_text_features(model, token_features)
            logit_scale = torch.ones([]) * (1 / args.temperature) 
            if 'text' in args.toword_loss and 'img' in args.toword_loss:
                text_loss = objectives.compute_sdm(text_features, t_feats, person_ids, logit_scale)
                img_loss = objectives.compute_sdm(text_features, i_feats, person_ids, logit_scale)
                total_loss = text_loss + img_loss
                batch_size = batch['images'].shape[0]
                meters['loss'].update(total_loss.item(), batch_size)
                meters['text_loss'].update(text_loss.item(), batch_size)
                meters['img_loss'].update(img_loss.item(), batch_size)
            elif 'text' in args.toword_loss:
                text_loss = objectives.compute_sdm(text_features, t_feats, person_ids, logit_scale)
                total_loss = text_loss
                batch_size = batch['images'].shape[0]
                meters['loss'].update(total_loss.item(), batch_size)
                meters['text_loss'].update(text_loss.item(), batch_size)
            elif 'img' in args.toword_loss:
                img_loss = objectives.compute_sdm(text_features, i_feats, person_ids, logit_scale)
                total_loss = img_loss
                batch_size = batch['images'].shape[0]
                meters['loss'].update(total_loss.item(), batch_size)
                meters['img_loss'].update(img_loss.item(), batch_size)
            else:
                text_loss = objectives.compute_sdm(text_features, t_feats, person_ids, logit_scale)
                img_loss = objectives.compute_sdm(text_features, i_feats, person_ids, logit_scale)
                total_loss = text_loss + img_loss
                batch_size = batch['images'].shape[0]
                meters['loss'].update(total_loss.item(), batch_size)
                meters['text_loss'].update(text_loss.item(), batch_size)
                meters['img_loss'].update(img_loss.item(), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', args.temperature, epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval(),img2text.module.eval())
                else:
                    top1 = evaluator.eval(model.eval(),img2text.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
    

    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("Word4Per.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval())


def do_inference_toword(model, img2text, test_query_loader, test_gallery_loader):

    logger = logging.getLogger("Word4Per.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator_toword(test_query_loader, test_gallery_loader)
    top1 = evaluator.eval(model.eval(),img2text.eval())

def do_inference_fuse_p2w(model, img2text_text, img2text_img, test_query_loader, test_gallery_loader):

    logger = logging.getLogger("Word4Per.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator_fuse_p2w(test_query_loader, test_gallery_loader)
    top1 = evaluator.eval(model.eval(),img2text_text.eval(), img2text_img.eval())