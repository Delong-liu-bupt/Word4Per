from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from utils.simple_tokenizer import SimpleTokenizer
from datasets.bases import tokenize

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("Word4Per.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]




class Evaluator_toword():
    def __init__(self, test_query_loader, test_gallery_loader):
        self.query_loader = test_query_loader # query
        self.gallery_loader = test_gallery_loader # gallery
        self.logger = logging.getLogger("Word4Per.eval")

    def _compute_embedding(self, model, img2text):
        model = model.eval()
        img2text = img2text.eval()

        device = next(model.parameters()).device
        tokenizer = SimpleTokenizer()

        qids, gids, qfeats_text, qfeats_img, qfeats_com, gfeats = [], [], [], [], [], []
        # query
        for iid, img, caption, caption_with_blank in self.query_loader:
            caption = caption.to(device)
            img = img.to(device)
            caption_with_blank = caption_with_blank.to(device)
            id_split = tokenize("*",tokenizer)[1]
            # print(id_split,tokenize("*",tokenizer))

            with torch.no_grad():
                query_text_feat = model.encode_text(caption)
                query_img_feat = model.encode_image(img)
                query_img_token = img2text(query_img_feat)
                composed_feat = model.encode_text_img_retrieval(caption_with_blank, query_img_token, split_ind=id_split, repeat=False)
            qids.append(iid.view(-1)) # flatten 
            qfeats_text.append(query_text_feat)
            qfeats_img.append(query_img_feat)
            qfeats_com.append(composed_feat)

        qids = torch.cat(qids, 0)
        qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_com = torch.cat(qfeats_com, 0)

        # gallery
        for iid, img in self.gallery_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(iid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qids, gids, qfeats_text, qfeats_img, qfeats_com, gfeats
    
    def eval(self, model, img2text):

        qids, gids, qfeats_text, qfeats_img, qfeats_com, gfeats = self._compute_embedding(model, img2text)

        qfeats_text = F.normalize(qfeats_text, p=2, dim=1) # query text features
        qfeats_img = F.normalize(qfeats_img, p=2, dim=1) # query img features
        qfeats_fuse = qfeats_text + qfeats_img
        qfeats_fuse = F.normalize(qfeats_fuse, p=2, dim=1) # query fuse features
        qfeats_com = F.normalize(qfeats_com, p=2, dim=1) # composed features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity_text = qfeats_text @ gfeats.t()
        similarity_img = qfeats_img @ gfeats.t()
        similarity_fuse = qfeats_fuse @ gfeats.t()
        similarity = qfeats_com.type(gfeats.dtype) @ gfeats.t()


        img_cmc, img_mAP, img_mINP, _ = rank(similarity=similarity_img, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        img_cmc, img_mAP, img_mINP = img_cmc.numpy(), img_mAP.numpy(), img_mINP.numpy()
        table_img = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table_img.add_row(['img', img_cmc[0], img_cmc[4], img_cmc[9], img_mAP, img_mINP])

        # table.float_format = '.4'
        table_img.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table_img.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table_img.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table_img.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table_img.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table_img))


        text_cmc, text_mAP, text_mINP, _ = rank(similarity=similarity_text, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        text_cmc, text_mAP, text_mINP = text_cmc.numpy(), text_mAP.numpy(), text_mINP.numpy()
        table_text = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table_text.add_row(['text', text_cmc[0], text_cmc[4], text_cmc[9], text_mAP, text_mINP])

        # table.float_format = '.4'
        table_text.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table_text.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table_text.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table_text.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table_text.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table_text))

        fuse_cmc, fuse_mAP, fuse_mINP, _ = rank(similarity=similarity_fuse, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        fuse_cmc, fuse_mAP, fuse_mINP = fuse_cmc.numpy(), fuse_mAP.numpy(), fuse_mINP.numpy()
        table_fuse = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table_fuse.add_row(['fuse', fuse_cmc[0], fuse_cmc[4], fuse_cmc[9], fuse_mAP, fuse_mINP])

        # table.float_format = '.4'
        table_fuse.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table_fuse.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table_fuse.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table_fuse.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table_fuse.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table_fuse))
        
        com_cmc, com_mAP, com_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        com_cmc, com_mAP, com_mINP = com_cmc.numpy(), com_mAP.numpy(), com_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['com', com_cmc[0], com_cmc[4], com_cmc[9], com_mAP, com_mINP])

        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return com_cmc[0]

class Evaluator_fuse_p2w():
    def __init__(self, test_query_loader, test_gallery_loader):
        self.query_loader = test_query_loader # query
        self.gallery_loader = test_gallery_loader # gallery
        self.logger = logging.getLogger("Word4Per.eval")

    def _compute_embedding(self, model, img2text_text, img2text_img):
        model = model.eval()
        img2text_text = img2text_text.eval()
        img2text_img = img2text_img.eval()

        device = next(model.parameters()).device
        tokenizer = SimpleTokenizer()

        qids, gids, qfeats_text, qfeats_img, qfeats_com_text, qfeats_com_img, gfeats = [], [], [], [], [], [], []
        # query
        for iid, img, caption, caption_with_blank in self.query_loader:
            caption = caption.to(device)
            img = img.to(device)
            caption_with_blank = caption_with_blank.to(device)
            id_split = tokenize("*",tokenizer)[1]
            # print(id_split,tokenize("*",tokenizer))

            with torch.no_grad():
                query_text_feat = model.encode_text(caption)
                query_img_feat = model.encode_image(img)
                query_img_token_t = img2text_text(query_img_feat)
                query_img_token_i = img2text_img(query_img_feat)
                composed_feat_text = model.encode_text_img_retrieval(caption_with_blank, query_img_token_t, split_ind=id_split, repeat=False)
                composed_feat_img = model.encode_text_img_retrieval(caption_with_blank, query_img_token_i, split_ind=id_split, repeat=False)
            qids.append(iid.view(-1)) # flatten 
            qfeats_text.append(query_text_feat)
            qfeats_img.append(query_img_feat)
            qfeats_com_text.append(composed_feat_text)
            qfeats_com_img.append(composed_feat_img)

        qids = torch.cat(qids, 0)
        qfeats_text = torch.cat(qfeats_text, 0)
        qfeats_img = torch.cat(qfeats_img, 0)
        qfeats_com_text = torch.cat(qfeats_com_text, 0)
        qfeats_com_img = torch.cat(qfeats_com_img, 0)

        # gallery
        for iid, img in self.gallery_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(iid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qids, gids, qfeats_text, qfeats_img, qfeats_com_text, qfeats_com_img, gfeats
    
    def eval(self, model, img2text_text, img2text_img):

        qids, gids, qfeats_text, qfeats_img, qfeats_com_text, qfeats_com_img, gfeats = self._compute_embedding(model, img2text_text, img2text_img)

        qfeats_text = F.normalize(qfeats_text, p=2, dim=1) # query text features
        qfeats_img = F.normalize(qfeats_img, p=2, dim=1) # query img features
        qfeats_fuse = qfeats_text + qfeats_img
        qfeats_fuse = F.normalize(qfeats_fuse, p=2, dim=1) # query fuse features
        qfeats_com_text = F.normalize(qfeats_com_text, p=2, dim=1) # composed features
        qfeats_com_img = F.normalize(qfeats_com_img, p=2, dim=1) # composed features
        qfeats_com = qfeats_com_text + qfeats_com_img
        qfeats_com = F.normalize(qfeats_com, p=2, dim=1) # query fuse features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity_text = qfeats_text @ gfeats.t()
        similarity_img = qfeats_img @ gfeats.t()
        similarity_fuse = qfeats_fuse @ gfeats.t()
        similarity_com = qfeats_com.type(gfeats.dtype) @ gfeats.t()
        similarity_com_text = qfeats_com_text.type(gfeats.dtype) @ gfeats.t()
        similarity_com_img = qfeats_com_img.type(gfeats.dtype) @ gfeats.t()


        img_cmc, img_mAP, img_mINP, _ = rank(similarity=similarity_img, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        img_cmc, img_mAP, img_mINP = img_cmc.numpy(), img_mAP.numpy(), img_mINP.numpy()
        table_img = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table_img.add_row(['img', img_cmc[0], img_cmc[4], img_cmc[9], img_mAP, img_mINP])

        # table.float_format = '.4'
        table_img.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table_img.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table_img.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table_img.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table_img.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table_img))


        text_cmc, text_mAP, text_mINP, _ = rank(similarity=similarity_text, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        text_cmc, text_mAP, text_mINP = text_cmc.numpy(), text_mAP.numpy(), text_mINP.numpy()
        table_text = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table_text.add_row(['text', text_cmc[0], text_cmc[4], text_cmc[9], text_mAP, text_mINP])

        # table.float_format = '.4'
        table_text.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table_text.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table_text.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table_text.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table_text.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table_text))

        fuse_cmc, fuse_mAP, fuse_mINP, _ = rank(similarity=similarity_fuse, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        fuse_cmc, fuse_mAP, fuse_mINP = fuse_cmc.numpy(), fuse_mAP.numpy(), fuse_mINP.numpy()
        table_fuse = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table_fuse.add_row(['fuse', fuse_cmc[0], fuse_cmc[4], fuse_cmc[9], fuse_mAP, fuse_mINP])

        # table.float_format = '.4'
        table_fuse.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table_fuse.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table_fuse.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table_fuse.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table_fuse.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table_fuse))

        com_text_cmc, com_text_mAP, com_text_mINP, _ = rank(similarity=similarity_com_text, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        com_text_cmc, com_text_mAP, com_text_mINP = com_text_cmc.numpy(), com_text_mAP.numpy(), com_text_mINP.numpy()
        table_com_text = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table_com_text.add_row(['com_t', com_text_cmc[0], com_text_cmc[4], com_text_cmc[9], com_text_mAP, com_text_mINP])

        # table.float_format = '.4'
        table_com_text.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table_com_text.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table_com_text.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table_com_text.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table_com_text.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table_com_text))

        com_img_cmc, com_img_mAP, com_img_mINP, _ = rank(similarity=similarity_com_img, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        com_img_cmc, com_img_mAP, com_img_mINP = com_img_cmc.numpy(), com_img_mAP.numpy(), com_img_mINP.numpy()
        table_com_img = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table_com_img.add_row(['com_i', com_img_cmc[0], com_img_cmc[4], com_img_cmc[9], com_img_mAP, com_img_mINP])

        # table.float_format = '.4'
        table_com_img.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table_com_img.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table_com_img.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table_com_img.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table_com_img.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table_com_img))
        
        com_cmc, com_mAP, com_mINP, _ = rank(similarity=similarity_com, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        com_cmc, com_mAP, com_mINP = com_cmc.numpy(), com_mAP.numpy(), com_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['com', com_cmc[0], com_cmc[4], com_cmc[9], com_mAP, com_mINP])

        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return com_cmc[0]

