import os
import sys

import torch
import torch.nn.functional as F
# project
from atomsurf.tasks.pip_site.model import PIPsiteNet,PINDERNet,PINDERNet_seed,PINDERNet_seed_ppi
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.utils.metrics import compute_auroc, compute_accuracy,compute_f1metrics,compute_auc_metrics
def focal_bce( inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** gamma

        # Apply alpha if provided
        if alpha is not None:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss
# def compute_BCE(pred, target):
#     """
#     Compute Binary Cross Entropy Loss with class imbalance handling.

#     :param pred:    Tensor of predictions (logits).
#     :param target:  Tensor of target labels (0 or 1).
#     :return:        Computed BCE loss.
#     """
#     num_pos = target.sum()
#     numels = len(target)

#     # Avoid division by zero
#     if num_pos.item() == 0:
#         return torch.tensor(0.0, device=pred.device)

#     # Compute positive weight (balancing factor)
#     weight = (numels - num_pos) / max(num_pos, 1)
#     weight = torch.tensor(weight, dtype=torch.float, device=pred.device)

#     # Compute BCE Loss with pos_weight adjustment
#     loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)
#     return loss 
def compute_BCE(pred, target):
    """
    Compute Binary Cross Entropy Loss with dynamic positive class weighting.

    Args:
        pred (Tensor): Raw logits, shape [N] or [N, 1]
        target (Tensor): Binary labels (0 or 1), same shape as pred

    Returns:
        Tensor: Scalar loss value
    """
    # Ensure target is float (required by BCEWithLogits)
    target = target.float()

    # Number of positive and total elements
    num_pos = target.sum()
    num_total = target.numel()

    # If no positive samples, return 0 to avoid invalid weight
    if num_pos == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Calculate pos_weight: weight for positive examples
    pos_weight = (num_total - num_pos) / num_pos
    pos_weight = pos_weight.clamp(min=1.0)  # optional: prevent tiny values

    # Use BCE with logits and pos_weight
    loss = F.binary_cross_entropy_with_logits(
        pred, target, pos_weight=pos_weight.detach()
    )

    return loss

class PIPsiteModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([hparams.model.pos_weight])
        self.model = PIPsiteNet(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)

    def step(self, batch):
        runtype = 'both'
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None
        if isinstance(batch.label_l, list):
            labels_l = torch.cat(batch.label_l).reshape(-1, 1)
            labels_r = torch.cat(batch.label_r).reshape(-1, 1)
        else:
            labels_l = batch.label_l.reshape(-1, 1)
            labels_r = batch.label_r.reshape(-1, 1)
        outputs = self(batch)
        if runtype=='both':
            labels = torch.cat([labels_l,labels_r])
            loss = self.criterion(outputs, labels)
        elif runtype=='l':
            labels= labels_l
            loss = self.criterion(outputs, labels)
        elif runtype=='r':
            labels= labels_r
            loss = self.criterion(outputs, labels)
        if torch.isnan(loss).any():
            print('Nan loss')
            return None, None, None
        return loss, outputs, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        acc = compute_accuracy(logits, labels, add_sigmoid=True)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/train": acc, "auroc/train": auroc}, on_epoch=True, batch_size=len(logits))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None:
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        acc = compute_accuracy(logits, labels, add_sigmoid=True)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/val": acc, "auroc/val": auroc}, on_epoch=True, batch_size=len(logits))
        self.log("auroc_val", auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None:
            self.log("acc/test", 0.5, on_epoch=True)
            return None
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        acc = compute_accuracy(logits, labels, add_sigmoid=True)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/test": acc, "auroc/test": auroc}, on_epoch=True, batch_size=len(logits))



class PINDERModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        # self.criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([hparams.model.pos_weight])
        self.model = PINDERNet(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)

    def step(self, batch):
        runtype = 'both'
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return  None, None, None,None, None, None,None
        if isinstance(batch.label_l, list):
            labels_l = torch.cat(batch.label_l).reshape(-1, 1)
            labels_r = torch.cat(batch.label_r).reshape(-1, 1)
        else:
            labels_l = batch.label_l.reshape(-1, 1)
            labels_r = batch.label_r.reshape(-1, 1)
        outputs_site,outputs_pair = self(batch)
        if runtype=='both':
            labels = torch.cat([labels_l,labels_r])
            loss_site = compute_BCE(outputs_site, labels)
        elif runtype=='l':
            labels= labels_l
            loss_site = compute_BCE(outputs_site, labels)
        elif runtype=='r':
            labels= labels_r
            loss_site = compute_BCE(outputs_site, labels)
        if isinstance(batch.labels_pair, list):
            labels_pair = torch.cat(batch.labels_pair).reshape(-1, 1)
        else:
            labels_pair = batch.labels_pair.reshape(-1, 1)
        loss_pair = compute_BCE(outputs_pair, labels_pair)
        loss = loss_site + loss_pair
        
        if torch.isnan(loss_site).any() or torch.isnan(loss_pair).any():
            print('Nan loss')
            return None, None, None,None, None, None,None
        return loss,loss_site,loss_pair, outputs_site,outputs_pair, labels,labels_pair

    def training_step(self, batch, batch_idx):
        loss,loss_site,loss_pair, outputs_site,outputs_pair, labels,labels_pair = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item(),"loss_site/train":loss_site.item(),"loss_pair/train":loss_pair.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        self.log_dict({"acc_site/train": acc_site, "auroc_site/train": auroc_site,"acc_pair/train":acc_pair,"auroc_pair/train":auroc_pair}, on_epoch=True, batch_size=len(outputs_site))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss,loss_site,loss_pair, outputs_site,outputs_pair, labels,labels_pair = self.step(batch)
        if loss is None:
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item(),"loss_site/val":loss_site.item(),"loss_pair/val":loss_pair.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        self.log_dict({"acc_site/val": acc_site, "auroc_site/val": auroc_site,"acc_pair/val":acc_pair,"auroc_pair/val":auroc_pair}, on_epoch=True, batch_size=len(outputs_site))
        # self.log("auroc_val", auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss,loss_site,loss_pair, outputs_site,outputs_pair, labels,labels_pair = self.step(batch)
        if loss is None:
            self.log_dict({"acc_site/val": 0.5, "auroc_site/val": 0.5,"acc_pair/val":0.5,"auroc_pair/val":0.5}, on_epoch=True, batch_size=0)
            return None
        self.log_dict({"loss/test": loss.item(),"loss_site/test":loss_site.item(),"loss_pair/test":loss_pair.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        self.log_dict({"acc_site/test": acc_site, "auroc_site/test": auroc_site,"acc_pair/test":acc_pair,"auroc_pair/test":auroc_pair}, on_epoch=True, batch_size=len(outputs_site))
        # self.log_dict({"acc/test": acc, "auroc/test": auroc}, on_epoch=True, batch_size=len(logits))


class PINDERModule_seed(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__() 
        self.cfg= cfg   
        self.save_hyperparameters()            
        # self.criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([hparams.model.pos_weight])
        self.model = PINDERNet_seed(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)

    def step(self, batch,save_emb=False):
        runtype = 'both'
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None,None, None, None,None,None, None,None,None,None, None,None,None,None,None,None,None
        if isinstance(batch.label_l, list):
            labels_l = torch.cat(batch.label_l).reshape(-1, 1)
            labels_r = torch.cat(batch.label_r).reshape(-1, 1)
            s_site_label_L = torch.cat(batch.s_site_label_L).reshape(-1, 1)
            s_site_label_R = torch.cat(batch.s_site_label_R).reshape(-1, 1)
        else:
            labels_l = batch.label_l.reshape(-1, 1)
            labels_r = batch.label_r.reshape(-1, 1)
            s_site_label_L = batch.s_site_label_L.reshape(-1, 1)
            s_site_label_R = batch.s_site_label_R.reshape(-1, 1)
        outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair,emb1,emb2,emb1_inv,emb2_inv,s_emb1,s_emb2,s_emb1_inv,s_emb2_inv = self.model(batch,save_emb)
                # -------- Hard negative mining on surface site --------
        
        if  self.current_epoch >30 and self.training:
            focal_alpha = 0.5
            pos_weight = 1.0
            s_site_label = torch.cat([s_site_label_L,s_site_label_R])
            with torch.no_grad():
                probs = torch.sigmoid(outputs_surface_site).view(-1)
                labels = s_site_label.view(-1)
                
                # Hard positives: label==1, predicted < 0.5
                # hard_pos_mask = (probs < 0.5) & (labels == 1)
                # hard_pos_idxs = torch.where(hard_pos_mask)[0]
                # print('hard pos num: ',len(hard_pos_idxs),'/',sum(labels))
                pos_mask = (labels == 1)
                pos_idxs = torch.where(pos_mask)[0]
                # Hard negatives: label==0, predicted > 0.5
                hard_neg_mask = (probs > 0.5) & (labels == 0)
                hard_neg_idxs = torch.where(hard_neg_mask)[0]

                # Easy negatives (optional): label==0, predicted < 0.5
                easy_neg_mask = (probs < 0.5) & (labels == 0)
                easy_neg_idxs = torch.where(easy_neg_mask)[0]

                # (Optional) sample an equal number of easy negatives
                if len(pos_idxs) > len(hard_neg_idxs):
                    n_easy = len(pos_idxs) -len(hard_neg_idxs)
                    easy_neg_sampled = easy_neg_idxs[torch.randperm(len(easy_neg_idxs))[:n_easy]]
                    selected_idxs = torch.cat([pos_idxs, hard_neg_idxs, easy_neg_sampled])
                else:
                    selected_idxs = torch.cat([pos_idxs, hard_neg_idxs[torch.randperm(len(hard_neg_idxs))[:len(pos_idxs)]]])
                outputs_surface_site = outputs_surface_site[selected_idxs]
                s_site_label = s_site_label[selected_idxs]
            if len(selected_idxs)<5:
                print('too less select idx skip train')
                return None, None, None,None, None, None,None,None, None,None,None,None, None,None,None,None,None,None,None
        else:
            # focal_alpha = 0.99
            # pos_weight = 100
            focal_alpha = 0.99
            pos_weight = 100.0
            s_site_label = torch.cat([s_site_label_L,s_site_label_R])
        # Select a subset: hard + sampled easy
        if runtype=='both':
            labels = torch.cat([labels_l,labels_r])
            loss_site = compute_BCE(outputs_site, labels)
            # s_site_label = torch.cat([s_site_label_L,s_site_label_R])
            s_site_label= s_site_label
            # loss_surf_site = compute_BCE(outputs_surface_site, s_site_label)
            loss_surf_site = focal_bce(
            outputs_surface_site, 
            s_site_label, 
            alpha=focal_alpha,  # Adjust based on class balance
            gamma=3.0   # Focuses on hard examples
        ) 
        elif runtype=='l':
            labels= labels_l
            loss_site = compute_BCE(outputs_site, labels)
            s_site_label = s_site_label_L
            loss_surf_site = compute_BCE(outputs_surface_site, s_site_label_L)
        elif runtype=='r':
            labels= labels_r
            s_site_label = s_site_label_R
            loss_site = compute_BCE(outputs_site, labels)
            loss_surf_site = compute_BCE(outputs_surface_site, s_site_label_R)
        if isinstance(batch.labels_pair, list):
            label_batch = torch.cat([torch.tensor([i] * len(labels)) for i, labels in enumerate(batch.labels_pair)]).to(batch.labels_pair[0].device)
            labels_pair = torch.cat(batch.labels_pair).reshape(-1, 1)
        else:
            labels_pair = batch.labels_pair.reshape(-1, 1)
        if isinstance(batch.s_label, list):
            s_label_batch = torch.cat([torch.tensor([i] * len(labels)) for i, labels in enumerate(batch.s_label)]).to(batch.s_label[0].device)
            s_label = torch.cat(batch.s_label).reshape(-1, 1)
        else:
            s_label = batch.s_label.reshape(-1, 1)
        loss_pair = compute_BCE(outputs_pair, labels_pair)
        loss_surface_pair= compute_BCE(outputs_surface_pair, s_label)
        ## add complementarity loss, dot product for similarity
        ## old loss for complementary
        complementray =  'euclidean' #'cosine'
        if complementray=='cosine':
            dists_1_2= F.cosine_similarity(emb1,emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
            dists_2_1= F.cosine_similarity(emb2,emb1_inv).reshape(-1, 1)
            loss_complementarity_g = compute_BCE(dists_1_2, labels_pair) + compute_BCE(dists_2_1, labels_pair) 

            s_dists_1_2= F.cosine_similarity(s_emb1,s_emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
            s_dists_2_1= F.cosine_similarity(s_emb2,s_emb1_inv).reshape(-1, 1)
            loss_complementarity_s = compute_BCE(s_dists_1_2, s_label) + compute_BCE(s_dists_2_1, s_label)
        elif complementray=='euclidean':
            emb1 = F.normalize(emb1, dim=1)
            emb1_inv = F.normalize(emb1_inv, dim=1)
            emb2 = F.normalize(emb2, dim=1)
            emb2_inv = F.normalize(emb2_inv, dim=1)
            s_emb1 = F.normalize(s_emb1, dim=1)
            s_emb1_inv = F.normalize(s_emb1_inv, dim=1)
            s_emb2 = F.normalize(s_emb2, dim=1)
            s_emb2_inv = F.normalize(s_emb2_inv, dim=1)

            # cosine similarities as “logits”
            cos_dists_1_2 = F.cosine_similarity(emb1, emb2_inv).view(-1,1)
            cos_dists_2_1 = F.cosine_similarity(emb2, emb1_inv).view(-1,1)
            cos_s_dists_1_2 = F.cosine_similarity(s_emb1, s_emb2_inv).view(-1,1)
            cos_s_dists_2_1 = F.cosine_similarity(s_emb2, s_emb1_inv).view(-1,1)

            # replace compute_BCE with focal_bce
            loss_complementarity_g_cos = (
                focal_bce(cos_dists_1_2, labels_pair, alpha=focal_alpha, gamma=2.0)
            + focal_bce(cos_dists_2_1, labels_pair, alpha=focal_alpha, gamma=2.0)
            )

            loss_complementarity_s_cos = (
                focal_bce(cos_s_dists_1_2, s_label,    alpha=focal_alpha, gamma=2.0)
            + focal_bce(cos_s_dists_2_1, s_label,    alpha=focal_alpha, gamma=2.0)
            )

            # your existing (non-BCE) contrastive losses
            def contrastive_loss(distances, labels, margin=1.0, pos_weight=1.0):
                loss_pos = pos_weight * labels * distances.pow(2)
                loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)
                return (loss_pos + loss_neg).mean()

            def cos_contrastive_loss(similarities, labels, margin=-0.5, pos_weight=1.0):
                loss_pos = pos_weight * labels * (1 - similarities).pow(2)
                loss_neg = (1 - labels) * F.relu(similarities - margin).pow(2)
                return (loss_pos + loss_neg).mean()

            # Euclidean distances
            dists_1_2 = torch.norm(emb1 - emb2_inv, p=2, dim=1, keepdim=True)
            dists_2_1 = torch.norm(emb2 - emb1_inv, p=2, dim=1, keepdim=True)
            s_dists_1_2 = torch.norm(s_emb1 - s_emb2_inv, p=2, dim=1, keepdim=True)
            s_dists_2_1 = torch.norm(s_emb2 - s_emb1_inv, p=2, dim=1, keepdim=True)

            # final loss aggregation
            loss_complementarity_g = (
                contrastive_loss(dists_1_2,   labels_pair, margin=2.0)
                + contrastive_loss(dists_2_1,   labels_pair, margin=2.0)
                + cos_contrastive_loss(cos_dists_1_2, labels_pair)
                + cos_contrastive_loss(cos_dists_2_1, labels_pair)
            )

            loss_complementarity_s = (
                contrastive_loss(s_dists_1_2, s_label, margin=2.0, pos_weight=pos_weight)
                + contrastive_loss(s_dists_2_1, s_label, margin=2.0, pos_weight=pos_weight)
                + cos_contrastive_loss(cos_s_dists_1_2, s_label, pos_weight=pos_weight)
                + cos_contrastive_loss(cos_s_dists_2_1, s_label, pos_weight=pos_weight)
            )

        # w_site, w_pair, w_surf_site, w_surf_pair = 1.0, 1.0, 20.0, 5.0
        # w_compl_g, w_compl_s = 1.0, 1.0
        # w_cos = 5.0
        w_site, w_pair, w_surf_site, w_surf_pair = 1.0, 0.5, 10.0, 1.0
        w_compl_g, w_compl_s = 10.0, 10.0
        w_cos = 10.0
        loss  = (
                w_site * loss_site
                + w_pair * loss_pair
                + w_surf_site * loss_surf_site
                + w_surf_pair * loss_surface_pair
                + w_compl_g * loss_complementarity_g
                + w_compl_s * loss_complementarity_s
                + w_cos  * (loss_complementarity_g_cos + loss_complementarity_s_cos)
                )
        # loss = loss_site + loss_pair +loss_surf_site+loss_surface_pair+ loss_complementartiy_g + loss_complementartiy_s +inverse_loss_g+inverse_loss_s+loss_comp_g_new+loss_comp_s_new
        # loss = loss_site + loss_pair +loss_surf_site+loss_surface_pair+ loss_comp_g_new + loss_comp_s_new+ inverse_loss_g+inverse_loss_s
        # loss = loss_complementartiy_g + loss_complementartiy_s+ inverse_loss_g+inverse_loss_s
        # TODO: change loss pass in return and in each step logging
        
        if torch.isnan(loss_site).any() or torch.isnan(loss_pair).any() or torch.isnan(loss_surface_pair).any() :
            print('Nan loss')
            return None, None, None,None, None, None,None,None, None,None,None,None, None,None,None,None,None,None,None
        return loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementarity_g,loss_complementarity_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,cos_dists_1_2,cos_dists_2_1,cos_s_dists_1_2,cos_s_dists_2_1

    def training_step(self, batch, batch_idx):
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1= self.step(batch)
        if loss is None:
            return None

        self.log_dict({"loss/train": loss.item(),"loss_site/train":loss_site.item(),"loss_pair/train":loss_pair.item(),"loss_surface_site/train":loss_surf_site.item(),"loss_surface/train":loss_surface_pair.item(),"loss_complementartiy_g/train":loss_complementartiy_g.item(),"loss_complementartiy_s/train":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site,auprc_site = compute_auc_metrics(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair,auprc_pair = compute_auc_metrics(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        f1_surface_site =compute_f1metrics(outputs_surface_site,s_site_label, add_sigmoid=True)
        print('f1 metric surface site train:',f1_surface_site,'; pos ratio:',sum(s_site_label)/len(s_site_label))
        auroc_surface_site,auprc_surface_site = compute_auc_metrics(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface,auprc_surface = compute_auc_metrics(outputs_surface_pair, s_label)
        #
        auroc_graph_seed1,auprc_graph_seed1 = compute_auc_metrics(dists_1_2, labels_pair)
        auroc_graph_seed2,auprc_graph_seed2 = compute_auc_metrics(dists_2_1, labels_pair)
        auroc_surface_seed1,auprc_surface_seed1 = compute_auc_metrics(s_dists_1_2, s_label)
        auroc_surface_seed2,auprc_surface_seed2 = compute_auc_metrics(s_dists_2_1, s_label)

        self.log_dict({"acc_site/train": acc_site, "auroc_site/train": auroc_site,"acc_pair/train":acc_pair,"auroc_pair/train":auroc_pair,'acc_surface_site/train':acc_surface_site,'auroc_surface_site/train':auroc_surface_site,"acc_surface/train":acc_surface,"auroc_surface/train":auroc_surface,'auroc_graph_seed1/train':auroc_graph_seed1,'auroc_graph_seed2/train':auroc_graph_seed2,'auroc_surface_seed1/train':auroc_surface_seed1,'auroc_surface_seed2/train':auroc_surface_seed2,'auprc_site/train':auprc_site,'auprc_pair/train':auprc_pair,'auprc_surface_site/train':auprc_surface_site,'auprc_surface/train':auprc_surface,'auprc_graph_seed1/train':auprc_graph_seed1,'auprc_graph_seed2/train':auprc_graph_seed2,'auprc_surface_seed1/train':auprc_surface_seed1,'auprc_surface_seed2/train':auprc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.eval()
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1 = self.step(batch)
        if loss is None:
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item(),"loss_site/val":loss_site.item(),"loss_pair/val":loss_pair.item(),"loss_surface/val":loss_surface_pair.item(),"loss_complementartiy_g/val":loss_complementartiy_g.item(),"loss_complementartiy_s/val":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site,auprc_site = compute_auc_metrics(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair,auprc_pair = compute_auc_metrics(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        f1_surface_site =compute_f1metrics(outputs_surface_site,s_site_label, add_sigmoid=True)
        print('f1 metric surface site val:',f1_surface_site,'; pos ratio:',sum(s_site_label)/len(s_site_label))
        auroc_surface_site,auprc_surface_site = compute_auc_metrics(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface,auprc_surface = compute_auc_metrics(outputs_surface_pair, s_label)
        #
        auroc_graph_seed1,auprc_graph_seed1 = compute_auc_metrics(dists_1_2, labels_pair)
        auroc_graph_seed2,auprc_graph_seed2 = compute_auc_metrics(dists_2_1, labels_pair)
        auroc_surface_seed1,auprc_surface_seed1 = compute_auc_metrics(s_dists_1_2, s_label)
        auroc_surface_seed2,auprc_surface_seed2 = compute_auc_metrics(s_dists_2_1, s_label)

        self.log_dict({"acc_site/val": acc_site, "auroc_site/val": auroc_site,"acc_pair/val":acc_pair,"auroc_pair/val":auroc_pair,'acc_surface_site/val':acc_surface_site,'auroc_surface_site/val':auroc_surface_site,"acc_surface/val":acc_surface,"auroc_surface/val":auroc_surface,'auroc_graph_seed1/val':auroc_graph_seed1,'auroc_graph_seed2/val':auroc_graph_seed2,'auroc_surface_seed1/val':auroc_surface_seed1,'auroc_surface_seed2/val':auroc_surface_seed2,'auprc_site/val':auprc_site,'auprc_pair/val':auprc_pair,'auprc_surface_site/val':auprc_surface_site,'auprc_surface/val':auprc_surface,'auprc_graph_seed1/val':auprc_graph_seed1,'auprc_graph_seed2/val':auprc_graph_seed2,'auprc_surface_seed1/val':auprc_surface_seed1,'auprc_surface_seed2/val':auprc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))

    def test_step(self, batch, batch_idx: int):
        self.model.eval()
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1 = self.step(batch,save_emb=True)
        if loss is None:
            print("test step skipped!")
            self.log("auroc/test", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/test": loss.item(),"loss_site/test":loss_site.item(),"loss_pair/test":loss_pair.item(),"loss_surface/test":loss_surface_pair.item(),"loss_complementartiy_g/test":loss_complementartiy_g.item(),"loss_complementartiy_s/test":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site,auprc_site = compute_auc_metrics(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair,auprc_pair = compute_auc_metrics(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        f1_surface_site =compute_f1metrics(outputs_surface_site,s_site_label, add_sigmoid=True)
        print('f1 metric surface site test:',f1_surface_site,'; pos ratio:',sum(s_site_label)/len(s_site_label))
        auroc_surface_site,auprc_surface_site = compute_auc_metrics(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface,auprc_surface = compute_auc_metrics(outputs_surface_pair, s_label)
        #
        auroc_graph_seed1,auprc_graph_seed1 = compute_auc_metrics(dists_1_2, labels_pair)
        auroc_graph_seed2,auprc_graph_seed2 = compute_auc_metrics(dists_2_1, labels_pair)
        auroc_surface_seed1,auprc_surface_seed1 = compute_auc_metrics(s_dists_1_2, s_label)
        auroc_surface_seed2,auprc_surface_seed2 = compute_auc_metrics(s_dists_2_1, s_label)

        self.log_dict({"acc_site/test": acc_site, "auroc_site/test": auroc_site,"acc_pair/test":acc_pair,"auroc_pair/test":auroc_pair,'acc_surface_site/test':acc_surface_site,'auroc_surface_site/test':auroc_surface_site,"acc_surface/test":acc_surface,"auroc_surface/test":auroc_surface,'auroc_graph_seed1/test':auroc_graph_seed1,'auroc_graph_seed2/test':auroc_graph_seed2,'auroc_surface_seed1/test':auroc_surface_seed1,'auroc_surface_seed2/test':auroc_surface_seed2,'auprc_site/test':auprc_site,'auprc_pair/test':auprc_pair,'auprc_surface_site/test':auprc_surface_site,'auprc_surface/test':auprc_surface,'auprc_graph_seed1/test':auprc_graph_seed1,'auprc_graph_seed2/test':auprc_graph_seed2,'auprc_surface_seed1/test':auprc_surface_seed1,'auprc_surface_seed2/test':auprc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))
    
    def predict_step(self, batch, batch_idx: int):
        self.model.eval()
        # self.model.train()
        import h5py
        prediction_type,prediction_save_dir,prediction_name =  'single','/work/lpdi/users/ymiao/Novoproj/target_proj/dataset/raw/sabdab/nb_target/','test_embedding'
        if prediction_type=='single':
            prediction_dataset = 'db5' 
        else:
            prediction_dataset = 'pinder' 

        prediction_save_dir = prediction_save_dir
        prediction_name = prediction_name
        if batch is  None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            print('None batch')
        elif prediction_dataset =='pinder':
            graph_1,graph_2,surface_1,surface_2,g_site_L_pred,g_site_R_pred,s_site_L_pred,s_site_R_pred= self.model.extract_embedding(batch)
            save_dir = prediction_save_dir#"/work/lpdi/users/ymiao/code/sbatch/pinderlog/"
            os.makedirs(save_dir, exist_ok=True)
            hdf5_path = os.path.join(save_dir, prediction_name+'.h5')
            #change it to  self.hparams.save_dir and hdf5_path later
            # Open HDF5 file in append mode

            graph_1_list= graph_1.to_data_list()
            graph_2_list= graph_2.to_data_list()
            surface_1_list= surface_1.to_data_list()
            surface_2_list= surface_2.to_data_list()
            with h5py.File(hdf5_path, "a") as hf:
                for batch_idx in graph_1.batch.unique():
                    tmp_id2, tmp_id1 = batch.id[batch_idx].split('--')
                    # Create a single large group for everything
                    grp = hf.require_group("embedding")#hf.require_group(f"batch_{batch_idx}")
                    if f"graph/{tmp_id1}" not in grp:
                        tmp_graph= graph_1_list[batch_idx]
                        tmp_graph_d= {}
                        tmp_graph_d['edge_index']=tmp_graph.edge_index.cpu().numpy()
                        tmp_graph_d['edge_attr']=tmp_graph.edge_attr.cpu().numpy()
                        tmp_graph_d['node_pos']=tmp_graph.node_pos.cpu().numpy()
                        tmp_graph_d['emb']=graph_1.x_emb[graph_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['inv_emb']=graph_1.x_inv[graph_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['site_pred']=g_site_L_pred[graph_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['label']=batch.label_l[batch_idx].detach().cpu().numpy()
                        grp_graph1 = grp.require_group(f"graph/{tmp_id1}")
                        save_dict_to_group(grp_graph1, tmp_graph_d)
                        tmp_surface1 = surface_1_list[batch_idx]
                        tmp_surface1_d = {}
                        tmp_surface1_d['verts'] = tmp_surface1.verts.cpu().numpy() 
                        tmp_surface1_d['faces'] = tmp_surface1.faces.cpu().numpy() 
                        tmp_surface1_d['vnormals'] = tmp_surface1.vnormals.cpu().numpy() 
                        tmp_surface1_d['emb'] = surface_1.x_emb[surface_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface1_d['inv_emb'] = surface_1.x_inv[surface_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface1_d['site_pred'] = s_site_L_pred[surface_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface1_d['label'] = batch.s_site_label_L[batch_idx].detach().cpu().numpy()
                        grp_surface1 = grp.require_group(f"surface/{tmp_id1}")
                        save_dict_to_group(grp_surface1, tmp_surface1_d)

                    if f"graph/{tmp_id2}" not in grp:
                        tmp_graph= graph_2_list[batch_idx]
                        tmp_graph_d= {}
                        tmp_graph_d['edge_index'] = tmp_graph.edge_index.cpu().numpy()
                        tmp_graph_d['edge_attr']=tmp_graph.edge_attr.cpu().numpy()
                        tmp_graph_d['node_pos']=tmp_graph.node_pos.cpu().numpy()
                        tmp_graph_d['emb']=graph_2.x_emb[graph_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['inv_emb']=graph_2.x_inv[graph_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['site_pred']=g_site_R_pred[graph_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['label']=batch.label_r[batch_idx].detach().cpu().numpy()
                        grp_graph2 = grp.require_group(f"graph/{tmp_id2}")
                        save_dict_to_group(grp_graph2, tmp_graph_d)
                        tmp_surface2 = surface_2_list[batch_idx]
                        tmp_surface2_d = {}
                        tmp_surface2_d['verts'] = tmp_surface2.verts.cpu().numpy() 
                        tmp_surface2_d['faces'] = tmp_surface2.faces.cpu().numpy() 
                        tmp_surface2_d['vnormals'] = tmp_surface2.vnormals.cpu().numpy() 
                        tmp_surface2_d['emb'] = surface_2.x_emb[surface_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface2_d['inv_emb'] = surface_2.x_inv[surface_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface2_d['site_pred'] = s_site_R_pred[surface_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface2_d['label'] = batch.s_site_label_R[batch_idx].detach().cpu().numpy()
                        grp_surface2 = grp.require_group(f"surface/{tmp_id2}")
                        save_dict_to_group(grp_surface2, tmp_surface2_d)
        elif prediction_dataset =='db5':
            graph_1,surface_1,g_site_L_pred,s_site_L_pred= self.model.extract_embedding_single(batch)
            save_dir = prediction_save_dir#"/work/lpdi/users/ymiao/code/sbatch/pinderlog/"
            os.makedirs(save_dir, exist_ok=True)
            hdf5_path = os.path.join(save_dir, prediction_name+'.h5')
            #change it to  self.hparams.save_dir and hdf5_path later
            # Open HDF5 file in append mode
            graph_1_list= graph_1.to_data_list()
            surface_1_list= surface_1.to_data_list()
            
            with h5py.File(hdf5_path, "a") as hf:
                for batch_idx in graph_1.batch.unique():
                    tmp_id1 = batch.id[batch_idx]
                    # print(tmp_id2,tmp_id1)
                    # Create a single large group for everything
                    grp = hf.require_group("embedding")#hf.require_group(f"batch_{batch_idx}")
                    if f"graph/{tmp_id1}" not in grp:
                        tmp_graph= graph_1_list[batch_idx]
                        tmp_graph_d= {}
                        tmp_graph_d['edge_index']=tmp_graph.edge_index.cpu().numpy()
                        tmp_graph_d['edge_attr']=tmp_graph.edge_attr.cpu().numpy()
                        tmp_graph_d['node_pos']=tmp_graph.node_pos.cpu().numpy()
                        tmp_graph_d['emb']=graph_1.x_emb[graph_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['inv_emb']=graph_1.x_inv[graph_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['site_pred']=g_site_L_pred[graph_1.batch == batch_idx].detach().cpu().numpy()
                        # tmp_graph_d['label']=batch.label_l[batch_idx].detach().cpu().numpy()
                        grp_graph1 = grp.require_group(f"graph/{tmp_id1}")
                        save_dict_to_group(grp_graph1, tmp_graph_d)
                        tmp_surface1 = surface_1_list[batch_idx]
                        tmp_surface1_d = {}
                        tmp_surface1_d['verts'] = tmp_surface1.verts.cpu().numpy() 
                        tmp_surface1_d['faces'] = tmp_surface1.faces.cpu().numpy() 
                        tmp_surface1_d['vnormals'] = tmp_surface1.vnormals.cpu().numpy() 
                        tmp_surface1_d['emb'] = surface_1.x_emb[surface_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface1_d['inv_emb'] = surface_1.x_inv[surface_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface1_d['site_pred'] = s_site_L_pred[surface_1.batch == batch_idx].detach().cpu().numpy()
                        # tmp_surface1_d['label'] = batch.s_site_label_L[batch_idx].detach().cpu().numpy()
                        grp_surface1 = grp.require_group(f"surface/{tmp_id1}")
                        save_dict_to_group(grp_surface1, tmp_surface1_d)


            print(f"Saved batch {batch_idx} embeddings to {hdf5_path}")

def save_dict_to_group(h5group, data_dict):
    """
    Save each key-value pair from data_dict into the provided h5group.
    Assumes values are NumPy arrays.
    """
    for key, value in data_dict.items():
        if value is not None:
            h5group.create_dataset(key, data=value)
        else:
            print(f"Warning: Value for key {key} is None; skipping.")


def freeze_except_ppi(model: torch.nn.Module):
    for name, param in model.named_parameters():
        # 如果名字里包含 cross_attn 或 ppi_mlp 则保留可训练，其它都 freeze
        if ('cross_attn' in name) or ('ppi_mlp' in name) or ('ppi' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
class PINDERModule_seed_ppi(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__() 
        self.cfg= cfg   
        self.save_hyperparameters()            
        # self.criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([hparams.model.pos_weight])
        self.model = PINDERNet_seed_ppi(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)
        self.freeze = cfg.freeze
        # if self.freeze:
            # freeze_except_ppi(self.model)
            # print("Trainable parameter names:")
            # for name, p in self.model.named_parameters():
            #     if p.requires_grad:
            #         print(name, p.shape)
    def compute_infonce_loss(self, ppi_all_outputs, ppi_all_labels, temperature=0.07):
        """
        InfoNCE Loss for contrastive learning
        Args:
            ppi_all_outputs: [batch_size * 3, 1] 模型预测的相似度分数
            ppi_all_labels: [batch_size * 3, 1] 标签 (1=正样本, 0=负样本)
        """
        batch_size = int(ppi_all_labels.sum().item())  # 正样本数量
        
        # 重塑为 [batch_size, 3] - 每行是 1个正样本 + 2个负样本
        # 假设数据排列是: pos_1, pos_2, ..., neg_1, neg_2, ...
        scores = ppi_all_outputs.squeeze()  # [batch_size * 3]
        
        # 分离正负样本
        pos_mask = ppi_all_labels.squeeze().bool()
        pos_scores = scores[pos_mask]  # [batch_size]
        neg_scores = scores[~pos_mask].view(batch_size, -1)  # [batch_size, 2]
        
        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        pos_exp = torch.exp(pos_scores / temperature)
        neg_exp = torch.exp(neg_scores / temperature).sum(dim=1)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp)).mean()
        
        return loss
    def focal_bce_ppi(self,outputs, labels, alpha=0.75, gamma=2.0):
        """
        Focal loss for PPI prediction
        alpha: 正样本权重 (0.75 表示正样本更重要)
        gamma: 难样本聚焦参数
        """
        bce = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
        probs = torch.sigmoid(outputs)
        
        # pt = p if label=1, else 1-p
        pt = torch.where(labels == 1, probs, 1 - probs)
        
        # alpha_t = alpha if label=1, else 1-alpha
        alpha_t = torch.where(labels == 1, alpha, 1 - alpha)
        
        focal_weight = alpha_t * (1 - pt) ** gamma
        
        return (focal_weight * bce).mean()
    def margin_loss(self, outputs, labels, margin=0.5):
        """
        Margin loss: 强制正负样本之间至少相差 margin
        正样本应该 > 0.5 + margin/2
        负样本应该 < 0.5 - margin/2
        """
        probs = torch.sigmoid(outputs)
        
        # 正样本: 希望 prob > 0.5 + margin/2
        pos_mask = (labels == 1)
        pos_target = 0.5 + margin / 2
        pos_loss = F.relu(pos_target - probs[pos_mask]).mean() if pos_mask.any() else 0.0
        
        # 负样本: 希望 prob < 0.5 - margin/2
        neg_mask = (labels == 0)
        neg_target = 0.5 - margin / 2
        neg_loss = F.relu(probs[neg_mask] - neg_target).mean() if neg_mask.any() else 0.0
        
        return pos_loss + neg_loss


    def center_loss(self, outputs, labels):
        """
        Center loss: 让正样本聚集在1附近，负样本聚集在0附近
        减小类内方差
        """
        probs = torch.sigmoid(outputs)
        
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        # 正样本距离1的距离
        pos_center_loss = ((probs[pos_mask] - 1) ** 2).mean() if pos_mask.any() else 0.0
        
        # 负样本距离0的距离
        neg_center_loss = ((probs[neg_mask] - 0) ** 2).mean() if neg_mask.any() else 0.0
        
        return pos_center_loss + neg_center_loss


    def separation_loss(self, outputs, labels, margin=0.3):
        """
        直接最大化正负样本均值的距离
        """
        probs = torch.sigmoid(outputs)
        
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        if not pos_mask.any() or not neg_mask.any():
            return 0.0
        
        pos_mean = probs[pos_mask].mean()
        neg_mean = probs[neg_mask].mean()
        
        # 希望 pos_mean - neg_mean > margin
        # 如果差距不够，就惩罚
        return F.relu(margin - (pos_mean - neg_mean))


    def triplet_margin_loss(self, outputs, labels, margin=0.3):
        """
        Triplet-like loss: 对于每个正样本，找到最难的负样本
        强制 pos_score > hardest_neg_score + margin
        """
        probs = torch.sigmoid(outputs)
        
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        if not pos_mask.any() or not neg_mask.any():
            return 0.0
        
        pos_probs = probs[pos_mask]
        neg_probs = probs[neg_mask]
        
        # 对每个正样本，找最高的负样本概率（最难区分的）
        hardest_neg = neg_probs.max()
        
        # 要求所有正样本都比最难的负样本高出margin
        losses = F.relu(hardest_neg + margin - pos_probs)
        
        return losses.mean()


    def variance_loss(self, outputs, labels):
        """
        减小正负样本各自的方差，增加类间距离
        """
        probs = torch.sigmoid(outputs)
        
        pos_mask = (labels == 1)
        neg_mask = (labels == 0)
        
        pos_var = probs[pos_mask].var() if pos_mask.sum() > 1 else 0.0
        neg_var = probs[neg_mask].var() if neg_mask.sum() > 1 else 0.0
        
        # 同时减小正负样本的方差
        return pos_var + neg_var


    def compute_ppi_loss(self, ppi_all_outputs, ppi_all_labels):
        """综合多种对比学习损失"""
        
        # 1. Focal BCE (基础分类损失)
        loss_bce = self.focal_bce_ppi(ppi_all_outputs, ppi_all_labels, alpha=0.75, gamma=2.0)
        
        # 2. InfoNCE (对比学习损失)
        loss_infonce = self.compute_infonce_loss(ppi_all_outputs, ppi_all_labels)
        
        # 3. Margin Loss (强制正负样本分离)
        loss_margin = self.margin_loss(ppi_all_outputs, ppi_all_labels, margin=0.5)
        
        # 4. Separation Loss (最大化正负样本均值差距)
        loss_separation = self.separation_loss(ppi_all_outputs, ppi_all_labels, margin=0.3)
        
        # 5. Center Loss (减小类内方差)
        loss_center = self.center_loss(ppi_all_outputs, ppi_all_labels)
        
        # 组合损失 - 可以根据效果调整权重
        total_ppi_loss = (
            loss_bce + 
            0.5 * loss_infonce + 
            0.3 * loss_margin +      # 强制分离
            0.2 * loss_separation +   # 最大化类间距
            0.1 * loss_center         # 减小类内方差
        )
        
        # # 可选: 返回详细loss用于监控
        # loss_dict = {
        #     'bce': loss_bce.item() if isinstance(loss_bce, torch.Tensor) else loss_bce,
        #     'infonce': loss_infonce.item() if isinstance(loss_infonce, torch.Tensor) else loss_infonce,
        #     'margin': loss_margin.item() if isinstance(loss_margin, torch.Tensor) else loss_margin,
        #     'separation': loss_separation.item() if isinstance(loss_separation, torch.Tensor) else loss_separation,
        #     'center': loss_center.item() if isinstance(loss_center, torch.Tensor) else loss_center,
        #     'total': total_ppi_loss.item()
        # }
        
        return total_ppi_loss
    def step(self, batch,save_emb=False):
        runtype = 'both'
        result = {
            'loss': None,
            'loss_site': None,
            'loss_pair': None,
            'loss_surf_site': None,
            'loss_surf_pair': None,
            'loss_complementarity_g': None,
            'loss_complementarity_s': None,
            'loss_ppi': None,
            'outputs_site': None,
            'outputs_pair': None,
            'outputs_surface_site': None,
            'outputs_surface_pair': None,
            'labels': None,
            'labels_pair': None,
            's_site_label': None,
            's_label': None,
            'cos_dists_1_2': None,
            'cos_dists_2_1': None,
            'cos_s_dists_1_2': None,
            'cos_s_dists_2_1': None,
            'ppi_all_outputs': None,
            'ppi_all_labels': None,
        }
        
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return result
        if isinstance(batch.label_l, list):
            labels_l = torch.cat(batch.label_l).reshape(-1, 1)
            labels_r = torch.cat(batch.label_r).reshape(-1, 1)
            s_site_label_L = torch.cat(batch.s_site_label_L).reshape(-1, 1)
            s_site_label_R = torch.cat(batch.s_site_label_R).reshape(-1, 1)
        else:
            labels_l = batch.label_l.reshape(-1, 1)
            labels_r = batch.label_r.reshape(-1, 1)
            s_site_label_L = batch.s_site_label_L.reshape(-1, 1)
            s_site_label_R = batch.s_site_label_R.reshape(-1, 1)

        outputs_site, outputs_pair, outputs_surface_site, outputs_surface_pair, \
        emb1, emb2, emb1_inv, emb2_inv, s_emb1, s_emb2, s_emb1_inv, s_emb2_inv, \
        ppi_all_outputs, ppi_all_labels = self.model(batch, save_emb)
                # -------- Hard negative mining on surface site --------
        
        if  self.current_epoch >100 and self.training:
            focal_alpha = 0.5
            pos_weight = 1.0
            s_site_label = torch.cat([s_site_label_L,s_site_label_R])
            with torch.no_grad():
                probs = torch.sigmoid(outputs_surface_site).view(-1)
                labels = s_site_label.view(-1)

                pos_mask = (labels == 1)
                pos_idxs = torch.where(pos_mask)[0]
                # Hard negatives: label==0, predicted > 0.5
                hard_neg_mask = (probs > 0.5) & (labels == 0)
                hard_neg_idxs = torch.where(hard_neg_mask)[0]

                # Easy negatives (optional): label==0, predicted < 0.5
                easy_neg_mask = (probs < 0.5) & (labels == 0)
                easy_neg_idxs = torch.where(easy_neg_mask)[0]

                # (Optional) sample an equal number of easy negatives
                if len(pos_idxs) > len(hard_neg_idxs):
                    n_easy = len(pos_idxs) -len(hard_neg_idxs)
                    easy_neg_sampled = easy_neg_idxs[torch.randperm(len(easy_neg_idxs))[:n_easy]]
                    selected_idxs = torch.cat([pos_idxs, hard_neg_idxs, easy_neg_sampled])
                else:
                    selected_idxs = torch.cat([pos_idxs, hard_neg_idxs[torch.randperm(len(hard_neg_idxs))[:len(pos_idxs)]]])
                outputs_surface_site = outputs_surface_site[selected_idxs]
                s_site_label = s_site_label[selected_idxs]
            if len(selected_idxs)<5:
                print('too less select idx skip train')
                return result
        else:
            focal_alpha = 0.99
            pos_weight = 100.0
            s_site_label = torch.cat([s_site_label_L,s_site_label_R])
        # Select a subset: hard + sampled easy
        if runtype=='both':
            labels = torch.cat([labels_l,labels_r])
            loss_site = compute_BCE(outputs_site, labels)
            # s_site_label = torch.cat([s_site_label_L,s_site_label_R])
            s_site_label= s_site_label
            # loss_surf_site = compute_BCE(outputs_surface_site, s_site_label)
            loss_surf_site = focal_bce(
            outputs_surface_site, 
            s_site_label, 
            alpha=focal_alpha,  # Adjust based on class balance
            gamma=3.0   # Focuses on hard examples
        ) 
        elif runtype=='l':
            labels= labels_l
            loss_site = compute_BCE(outputs_site, labels)
            s_site_label = s_site_label_L
            loss_surf_site = compute_BCE(outputs_surface_site, s_site_label_L)
        elif runtype=='r':
            labels= labels_r
            s_site_label = s_site_label_R
            loss_site = compute_BCE(outputs_site, labels)
            loss_surf_site = compute_BCE(outputs_surface_site, s_site_label_R)
        if isinstance(batch.labels_pair, list):
            label_batch = torch.cat([torch.tensor([i] * len(labels)) for i, labels in enumerate(batch.labels_pair)]).to(batch.labels_pair[0].device)
            labels_pair = torch.cat(batch.labels_pair).reshape(-1, 1)
        else:
            labels_pair = batch.labels_pair.reshape(-1, 1)
        if isinstance(batch.s_label, list):
            s_label_batch = torch.cat([torch.tensor([i] * len(labels)) for i, labels in enumerate(batch.s_label)]).to(batch.s_label[0].device)
            s_label = torch.cat(batch.s_label).reshape(-1, 1)
        else:
            s_label = batch.s_label.reshape(-1, 1)
        loss_pair = compute_BCE(outputs_pair, labels_pair)
        loss_surface_pair= compute_BCE(outputs_surface_pair, s_label)
        ## add complementarity loss, dot product for similarity
        ## old loss for complementary
        complementray =  'euclidean' #'cosine'
        if complementray=='cosine':
            dists_1_2= F.cosine_similarity(emb1,emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
            dists_2_1= F.cosine_similarity(emb2,emb1_inv).reshape(-1, 1)
            loss_complementarity_g = compute_BCE(dists_1_2, labels_pair) + compute_BCE(dists_2_1, labels_pair) 

            s_dists_1_2= F.cosine_similarity(s_emb1,s_emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
            s_dists_2_1= F.cosine_similarity(s_emb2,s_emb1_inv).reshape(-1, 1)
            loss_complementarity_s = compute_BCE(s_dists_1_2, s_label) + compute_BCE(s_dists_2_1, s_label)
        elif complementray=='euclidean':
            emb1 = F.normalize(emb1, dim=1)
            emb1_inv = F.normalize(emb1_inv, dim=1)
            emb2 = F.normalize(emb2, dim=1)
            emb2_inv = F.normalize(emb2_inv, dim=1)
            s_emb1 = F.normalize(s_emb1, dim=1)
            s_emb1_inv = F.normalize(s_emb1_inv, dim=1)
            s_emb2 = F.normalize(s_emb2, dim=1)
            s_emb2_inv = F.normalize(s_emb2_inv, dim=1)

            # cosine similarities as “logits”
            cos_dists_1_2 = F.cosine_similarity(emb1, emb2_inv).view(-1,1)
            cos_dists_2_1 = F.cosine_similarity(emb2, emb1_inv).view(-1,1)
            cos_s_dists_1_2 = F.cosine_similarity(s_emb1, s_emb2_inv).view(-1,1)
            cos_s_dists_2_1 = F.cosine_similarity(s_emb2, s_emb1_inv).view(-1,1)

            # replace compute_BCE with focal_bce
            loss_complementarity_g_cos = (
                focal_bce(cos_dists_1_2, labels_pair, alpha=focal_alpha, gamma=2.0)
            + focal_bce(cos_dists_2_1, labels_pair, alpha=focal_alpha, gamma=2.0)
            )

            loss_complementarity_s_cos = (
                focal_bce(cos_s_dists_1_2, s_label,    alpha=focal_alpha, gamma=2.0)
            + focal_bce(cos_s_dists_2_1, s_label,    alpha=focal_alpha, gamma=2.0)
            )

            # your existing (non-BCE) contrastive losses
            def contrastive_loss(distances, labels, margin=1.0, pos_weight=1.0):
                loss_pos = pos_weight * labels * distances.pow(2)
                loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)
                return (loss_pos + loss_neg).mean()

            def cos_contrastive_loss(similarities, labels, margin=-0.5, pos_weight=1.0):
                loss_pos = pos_weight * labels * (1 - similarities).pow(2)
                loss_neg = (1 - labels) * F.relu(similarities - margin).pow(2)
                return (loss_pos + loss_neg).mean()

            # Euclidean distances
            dists_1_2 = torch.norm(emb1 - emb2_inv, p=2, dim=1, keepdim=True)
            dists_2_1 = torch.norm(emb2 - emb1_inv, p=2, dim=1, keepdim=True)
            s_dists_1_2 = torch.norm(s_emb1 - s_emb2_inv, p=2, dim=1, keepdim=True)
            s_dists_2_1 = torch.norm(s_emb2 - s_emb1_inv, p=2, dim=1, keepdim=True)

            # final loss aggregation
            loss_complementarity_g = (
                contrastive_loss(dists_1_2,   labels_pair, margin=2.0)
                + contrastive_loss(dists_2_1,   labels_pair, margin=2.0)
                + cos_contrastive_loss(cos_dists_1_2, labels_pair)
                + cos_contrastive_loss(cos_dists_2_1, labels_pair)
            )

            loss_complementarity_s = (
                contrastive_loss(s_dists_1_2, s_label, margin=2.0, pos_weight=pos_weight)
                + contrastive_loss(s_dists_2_1, s_label, margin=2.0, pos_weight=pos_weight)
                + cos_contrastive_loss(cos_s_dists_1_2, s_label, pos_weight=pos_weight)
                + cos_contrastive_loss(cos_s_dists_2_1, s_label, pos_weight=pos_weight)
            )
        loss_ppi = self.compute_ppi_loss(ppi_all_outputs, ppi_all_labels)
        # 综合总损失
        w_site, w_pair, w_surf_site, w_surf_pair = 1.0, 0.5, 10.0, 1.0
        w_compl_g, w_compl_s = 10.0, 10.0
        w_cos = 10.0
        w_ppi = 5.0  # 🔥 PPI 权重
        if not self.freeze:
            loss = (
                w_site * loss_site
                + w_pair * loss_pair
                + w_surf_site * loss_surf_site
                + w_surf_pair * loss_surface_pair
                + w_compl_g * loss_complementarity_g
                + w_compl_s * loss_complementarity_s
                + w_cos * (loss_complementarity_g_cos + loss_complementarity_s_cos)
                + w_ppi * loss_ppi  # 🔥 添加 PPI loss
            )
        else:
            loss = loss_ppi
        
        if torch.isnan(loss_site).any() or torch.isnan(loss_pair).any() or torch.isnan(loss_surface_pair).any() :
            print('Nan loss')
            return result
        # 更新结果
        result.update({
            'loss': loss,
            'loss_site': loss_site,
            'loss_pair': loss_pair,
            'loss_surf_site': loss_surf_site,
            'loss_surf_pair': loss_surface_pair,
            'loss_complementarity_g': loss_complementarity_g,
            'loss_complementarity_s': loss_complementarity_s,
            'loss_ppi': loss_ppi,
            'outputs_site': outputs_site,
            'outputs_pair': outputs_pair,
            'outputs_surface_site': outputs_surface_site,
            'outputs_surface_pair': outputs_surface_pair,
            'labels': labels,
            'labels_pair': labels_pair,
            's_site_label': s_site_label,
            's_label': s_label,
            'cos_dists_1_2': cos_dists_1_2,
            'cos_dists_2_1': cos_dists_2_1,
            'cos_s_dists_1_2': cos_s_dists_1_2,
            'cos_s_dists_2_1': cos_s_dists_2_1,
            'ppi_all_outputs': ppi_all_outputs,
            'ppi_all_labels': ppi_all_labels,
        })
        return result

    def training_step(self, batch, batch_idx):
        result = self.step(batch)
        
        # 解包所有结果
        loss = result['loss']
        loss_site = result['loss_site']
        loss_pair = result['loss_pair']
        loss_surf_site = result['loss_surf_site']
        loss_surf_pair = result['loss_surf_pair']
        loss_complementarity_g = result['loss_complementarity_g']
        loss_complementarity_s = result['loss_complementarity_s']
        loss_ppi = result['loss_ppi']
        
        outputs_site = result['outputs_site']
        outputs_pair = result['outputs_pair']
        outputs_surface_site = result['outputs_surface_site']
        outputs_surface_pair = result['outputs_surface_pair']
        
        labels = result['labels']
        labels_pair = result['labels_pair']
        s_site_label = result['s_site_label']
        s_label = result['s_label']
        
        cos_dists_1_2 = result['cos_dists_1_2']
        cos_dists_2_1 = result['cos_dists_2_1']
        cos_s_dists_1_2 = result['cos_s_dists_1_2']
        cos_s_dists_2_1 = result['cos_s_dists_2_1']
        
        ppi_all_outputs = result['ppi_all_outputs']
        ppi_all_labels = result['ppi_all_labels']
        
        # 早期返回检查
        if loss is None:
            return None
        
        # === 记录损失 ===
        self.log_dict({
            "loss/train": loss.item(),
            "loss_site/train": loss_site.item(),
            "loss_pair/train": loss_pair.item(),
            "loss_surface_site/train": loss_surf_site.item(),
            "loss_surface/train": loss_surf_pair.item(),
            "loss_complementarity_g/train": loss_complementarity_g.item(),
            "loss_complementarity_s/train": loss_complementarity_s.item(),
            "loss_ppi/train": loss_ppi.item(),  # 🔥 新增 PPI loss
        }, on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        
        # === Site 预测指标 ===
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site, auprc_site = compute_auc_metrics(outputs_site, labels)
        
        # === Pair 预测指标 ===
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair, auprc_pair = compute_auc_metrics(outputs_pair, labels_pair)
        
        # === Surface site 预测指标 ===
        acc_surface_site = compute_accuracy(outputs_surface_site, s_site_label, add_sigmoid=True)
        f1_surface_site = compute_f1metrics(outputs_surface_site, s_site_label, add_sigmoid=True)
        f1_surface_site = f1_surface_site['f1']
        # print(f'f1 metric surface site train: {f1_surface_site}; pos ratio: {sum(s_site_label)/len(s_site_label):.4f}')
        auroc_surface_site, auprc_surface_site = compute_auc_metrics(outputs_surface_site, s_site_label)
        
        # === Surface pair 预测指标 ===
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface, auprc_surface = compute_auc_metrics(outputs_surface_pair, s_label)
        
        # === Seed complementarity 指标 ===
        auroc_graph_seed1, auprc_graph_seed1 = compute_auc_metrics(cos_dists_1_2, labels_pair)
        auroc_graph_seed2, auprc_graph_seed2 = compute_auc_metrics(cos_dists_2_1, labels_pair)
        auroc_surface_seed1, auprc_surface_seed1 = compute_auc_metrics(cos_s_dists_1_2, s_label)
        auroc_surface_seed2, auprc_surface_seed2 = compute_auc_metrics(cos_s_dists_2_1, s_label)
        
        # 🔥 === PPI 预测指标 ===
        acc_ppi = compute_accuracy(ppi_all_outputs, ppi_all_labels, add_sigmoid=True)
        auroc_ppi, auprc_ppi = compute_auc_metrics(ppi_all_outputs, ppi_all_labels)
        
        # 分别计算正样本和负样本的准确率
        pos_mask = (ppi_all_labels == 1).squeeze()
        neg_mask = (ppi_all_labels == 0).squeeze()
        
        if pos_mask.sum() > 0:
            acc_ppi_pos = compute_accuracy(
                ppi_all_outputs[pos_mask], 
                ppi_all_labels[pos_mask], 
                add_sigmoid=True
            )
        else:
            acc_ppi_pos = 0.0
        
        if neg_mask.sum() > 0:
            acc_ppi_neg = compute_accuracy(
                ppi_all_outputs[neg_mask], 
                ppi_all_labels[neg_mask], 
                add_sigmoid=True
            )
        else:
            acc_ppi_neg = 0.0
        
        # print(f'PPI - Pos samples: {pos_mask.sum()}, Neg samples: {neg_mask.sum()}, '
        #     f'Acc: {acc_ppi:.4f}, Pos Acc: {acc_ppi_pos:.4f}, Neg Acc: {acc_ppi_neg:.4f}, '
        #     f'AUROC: {auroc_ppi:.4f}, AUPRC: {auprc_ppi:.4f}')
        
        # === 记录所有指标 ===
        self.log_dict({
            # Site metrics
            "acc_site/train": acc_site,
            "auroc_site/train": auroc_site,
            "auprc_site/train": auprc_site,
            
            # Pair metrics
            "acc_pair/train": acc_pair,
            "auroc_pair/train": auroc_pair,
            "auprc_pair/train": auprc_pair,
            
            # Surface site metrics
            "acc_surface_site/train": acc_surface_site,
            "auroc_surface_site/train": auroc_surface_site,
            "auprc_surface_site/train": auprc_surface_site,
            "f1_surface_site/train": f1_surface_site,  # 🔥 添加 F1
            
            # Surface pair metrics
            "acc_surface/train": acc_surface,
            "auroc_surface/train": auroc_surface,
            "auprc_surface/train": auprc_surface,
            
            # Graph seed complementarity
            "auroc_graph_seed1/train": auroc_graph_seed1,
            "auroc_graph_seed2/train": auroc_graph_seed2,
            "auprc_graph_seed1/train": auprc_graph_seed1,
            "auprc_graph_seed2/train": auprc_graph_seed2,
            
            # Surface seed complementarity
            "auroc_surface_seed1/train": auroc_surface_seed1,
            "auroc_surface_seed2/train": auroc_surface_seed2,
            "auprc_surface_seed1/train": auprc_surface_seed1,
            "auprc_surface_seed2/train": auprc_surface_seed2,
            
            # 🔥 PPI metrics
            "acc_ppi/train": acc_ppi,
            "acc_ppi_pos/train": acc_ppi_pos,  # 正样本准确率
            "acc_ppi_neg/train": acc_ppi_neg,  # 负样本准确率
            "auroc_ppi/train": auroc_ppi,
            "auprc_ppi/train": auprc_ppi,
        }, on_epoch=True, batch_size=len(outputs_site))
        
        return loss

    def validation_step(self, batch, batch_idx):
        result = self.step(batch, save_emb=False)
        
        # 解包所有结果
        loss = result['loss']
        loss_site = result['loss_site']
        loss_pair = result['loss_pair']
        loss_surf_site = result['loss_surf_site']
        loss_surf_pair = result['loss_surf_pair']
        loss_complementarity_g = result['loss_complementarity_g']
        loss_complementarity_s = result['loss_complementarity_s']
        loss_ppi = result['loss_ppi']
        
        outputs_site = result['outputs_site']
        outputs_pair = result['outputs_pair']
        outputs_surface_site = result['outputs_surface_site']
        outputs_surface_pair = result['outputs_surface_pair']
        
        labels = result['labels']
        labels_pair = result['labels_pair']
        s_site_label = result['s_site_label']
        s_label = result['s_label']
        
        cos_dists_1_2 = result['cos_dists_1_2']
        cos_dists_2_1 = result['cos_dists_2_1']
        cos_s_dists_1_2 = result['cos_s_dists_1_2']
        cos_s_dists_2_1 = result['cos_s_dists_2_1']
        
        ppi_all_outputs = result['ppi_all_outputs']
        ppi_all_labels = result['ppi_all_labels']
        
        # 早期返回检查
        if loss is None:
            return None
        
        # === 记录验证损失 ===
        self.log_dict({
            "loss/val": loss.item(),
            "loss_site/val": loss_site.item(),
            "loss_pair/val": loss_pair.item(),
            "loss_surface_site/val": loss_surf_site.item(),
            "loss_surface/val": loss_surf_pair.item(),
            "loss_complementarity_g/val": loss_complementarity_g.item(),
            "loss_complementarity_s/val": loss_complementarity_s.item(),
            "loss_ppi/val": loss_ppi.item(),
        }, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(outputs_site))
        
        # === Site 预测指标 ===
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site, auprc_site = compute_auc_metrics(outputs_site, labels)
        
        # === Pair 预测指标 ===
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair, auprc_pair = compute_auc_metrics(outputs_pair, labels_pair)
        
        # === Surface site 预测指标 ===
        acc_surface_site = compute_accuracy(outputs_surface_site, s_site_label, add_sigmoid=True)
        f1_surface_site = compute_f1metrics(outputs_surface_site, s_site_label, add_sigmoid=True)
        f1_surface_site = f1_surface_site['f1']
        auroc_surface_site, auprc_surface_site = compute_auc_metrics(outputs_surface_site, s_site_label)
        
        # === Surface pair 预测指标 ===
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface, auprc_surface = compute_auc_metrics(outputs_surface_pair, s_label)
        
        # === Seed complementarity 指标 ===
        auroc_graph_seed1, auprc_graph_seed1 = compute_auc_metrics(cos_dists_1_2, labels_pair)
        auroc_graph_seed2, auprc_graph_seed2 = compute_auc_metrics(cos_dists_2_1, labels_pair)
        auroc_surface_seed1, auprc_surface_seed1 = compute_auc_metrics(cos_s_dists_1_2, s_label)
        auroc_surface_seed2, auprc_surface_seed2 = compute_auc_metrics(cos_s_dists_2_1, s_label)
        
        # === PPI 预测指标 ===
        acc_ppi = compute_accuracy(ppi_all_outputs, ppi_all_labels, add_sigmoid=True)
        auroc_ppi, auprc_ppi = compute_auc_metrics(ppi_all_outputs, ppi_all_labels)
        
        # 分别计算正样本和负样本的准确率
        pos_mask = (ppi_all_labels == 1).squeeze()
        neg_mask = (ppi_all_labels == 0).squeeze()
        
        if pos_mask.sum() > 0:
            acc_ppi_pos = compute_accuracy(
                ppi_all_outputs[pos_mask], 
                ppi_all_labels[pos_mask], 
                add_sigmoid=True
            )
        else:
            acc_ppi_pos = 0.0
        
        if neg_mask.sum() > 0:
            acc_ppi_neg = compute_accuracy(
                ppi_all_outputs[neg_mask], 
                ppi_all_labels[neg_mask], 
                add_sigmoid=True
            )
        else:
            acc_ppi_neg = 0.0
        
        print(f'[VAL] PPI - Pos samples: {pos_mask.sum()}, Neg samples: {neg_mask.sum()}, '
            f'Acc: {acc_ppi:.4f}, Pos Acc: {acc_ppi_pos:.4f}, Neg Acc: {acc_ppi_neg:.4f}, '
            f'AUROC: {auroc_ppi:.4f}, AUPRC: {auprc_ppi:.4f}')
        
        # === 记录所有验证指标 ===
        self.log_dict({
            # Site metrics
            "acc_site/val": acc_site,
            "auroc_site/val": auroc_site,
            "auprc_site/val": auprc_site,
            
            # Pair metrics
            "acc_pair/val": acc_pair,
            "auroc_pair/val": auroc_pair,
            "auprc_pair/val": auprc_pair,
            
            # Surface site metrics
            "acc_surface_site/val": acc_surface_site,
            "auroc_surface_site/val": auroc_surface_site,
            "auprc_surface_site/val": auprc_surface_site,
            "f1_surface_site/val": f1_surface_site,
            
            # Surface pair metrics
            "acc_surface/val": acc_surface,
            "auroc_surface/val": auroc_surface,
            "auprc_surface/val": auprc_surface,
            
            # Graph seed complementarity
            "auroc_graph_seed1/val": auroc_graph_seed1,
            "auroc_graph_seed2/val": auroc_graph_seed2,
            "auprc_graph_seed1/val": auprc_graph_seed1,
            "auprc_graph_seed2/val": auprc_graph_seed2,
            
            # Surface seed complementarity
            "auroc_surface_seed1/val": auroc_surface_seed1,
            "auroc_surface_seed2/val": auroc_surface_seed2,
            "auprc_surface_seed1/val": auprc_surface_seed1,
            "auprc_surface_seed2/val": auprc_surface_seed2,
            
            # PPI metrics
            "acc_ppi/val": acc_ppi,
            "acc_ppi_pos/val": acc_ppi_pos,
            "acc_ppi_neg/val": acc_ppi_neg,
            "auroc_ppi/val": auroc_ppi,
            "auprc_ppi/val": auprc_ppi,
        }, on_epoch=True, batch_size=len(outputs_site))
        
        return loss


    def test_step(self, batch, batch_idx):
        result = self.step(batch, save_emb=False)
        
        # 解包所有结果
        loss = result['loss']
        loss_site = result['loss_site']
        loss_pair = result['loss_pair']
        loss_surf_site = result['loss_surf_site']
        loss_surf_pair = result['loss_surf_pair']
        loss_complementarity_g = result['loss_complementarity_g']
        loss_complementarity_s = result['loss_complementarity_s']
        loss_ppi = result['loss_ppi']
        
        outputs_site = result['outputs_site']
        outputs_pair = result['outputs_pair']
        outputs_surface_site = result['outputs_surface_site']
        outputs_surface_pair = result['outputs_surface_pair']
        
        labels = result['labels']
        labels_pair = result['labels_pair']
        s_site_label = result['s_site_label']
        s_label = result['s_label']
        
        cos_dists_1_2 = result['cos_dists_1_2']
        cos_dists_2_1 = result['cos_dists_2_1']
        cos_s_dists_1_2 = result['cos_s_dists_1_2']
        cos_s_dists_2_1 = result['cos_s_dists_2_1']
        
        ppi_all_outputs = result['ppi_all_outputs']
        ppi_all_labels = result['ppi_all_labels']
        
        # 早期返回检查
        if loss is None:
            return None
        
        # === 记录测试损失 ===
        self.log_dict({
            "loss/test": loss.item(),
            "loss_site/test": loss_site.item(),
            "loss_pair/test": loss_pair.item(),
            "loss_surface_site/test": loss_surf_site.item(),
            "loss_surface/test": loss_surf_pair.item(),
            "loss_complementarity_g/test": loss_complementarity_g.item(),
            "loss_complementarity_s/test": loss_complementarity_s.item(),
            "loss_ppi/test": loss_ppi.item(),
        }, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(outputs_site))
        
        # === Site 预测指标 ===
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site, auprc_site = compute_auc_metrics(outputs_site, labels)
        
        # === Pair 预测指标 ===
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair, auprc_pair = compute_auc_metrics(outputs_pair, labels_pair)
        
        # === Surface site 预测指标 ===
        acc_surface_site = compute_accuracy(outputs_surface_site, s_site_label, add_sigmoid=True)
        f1_surface_site = compute_f1metrics(outputs_surface_site, s_site_label, add_sigmoid=True)
        auroc_surface_site, auprc_surface_site = compute_auc_metrics(outputs_surface_site, s_site_label)
        f1_surface_site = f1_surface_site['f1']
        # print(f'[TEST] f1 metric surface site: {f1_surface_site:.4f}; pos ratio: {sum(s_site_label)/len(s_site_label):.4f}')
        
        # === Surface pair 预测指标 ===
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface, auprc_surface = compute_auc_metrics(outputs_surface_pair, s_label)
        
        # === Seed complementarity 指标 ===
        auroc_graph_seed1, auprc_graph_seed1 = compute_auc_metrics(cos_dists_1_2, labels_pair)
        auroc_graph_seed2, auprc_graph_seed2 = compute_auc_metrics(cos_dists_2_1, labels_pair)
        auroc_surface_seed1, auprc_surface_seed1 = compute_auc_metrics(cos_s_dists_1_2, s_label)
        auroc_surface_seed2, auprc_surface_seed2 = compute_auc_metrics(cos_s_dists_2_1, s_label)
        
        # === PPI 预测指标 ===
        acc_ppi = compute_accuracy(ppi_all_outputs, ppi_all_labels, add_sigmoid=True)
        auroc_ppi, auprc_ppi = compute_auc_metrics(ppi_all_outputs, ppi_all_labels)
        
        # 分别计算正样本和负样本的准确率
        pos_mask = (ppi_all_labels == 1).squeeze()
        neg_mask = (ppi_all_labels == 0).squeeze()
        
        if pos_mask.sum() > 0:
            acc_ppi_pos = compute_accuracy(
                ppi_all_outputs[pos_mask], 
                ppi_all_labels[pos_mask], 
                add_sigmoid=True
            )
            auroc_ppi_pos, auprc_ppi_pos = compute_auc_metrics(
                ppi_all_outputs[pos_mask], 
                ppi_all_labels[pos_mask]
            )
        else:
            acc_ppi_pos = 0.0
            auroc_ppi_pos = 0.0
            auprc_ppi_pos = 0.0
        
        if neg_mask.sum() > 0:
            acc_ppi_neg = compute_accuracy(
                ppi_all_outputs[neg_mask], 
                ppi_all_labels[neg_mask], 
                add_sigmoid=True
            )
            auroc_ppi_neg, auprc_ppi_neg = compute_auc_metrics(
                ppi_all_outputs[neg_mask], 
                ppi_all_labels[neg_mask]
            )
        else:
            acc_ppi_neg = 0.0
            auroc_ppi_neg = 0.0
            auprc_ppi_neg = 0.0
        
        print(f'[TEST] PPI Results:')
        print(f'  Total - Pos: {pos_mask.sum()}, Neg: {neg_mask.sum()}')
        print(f'  Overall - Acc: {acc_ppi:.4f}, AUROC: {auroc_ppi:.4f}, AUPRC: {auprc_ppi:.4f}')
        print(f'  Positive - Acc: {acc_ppi_pos:.4f}, AUROC: {auroc_ppi_pos:.4f}, AUPRC: {auprc_ppi_pos:.4f}')
        print(f'  Negative - Acc: {acc_ppi_neg:.4f}, AUROC: {auroc_ppi_neg:.4f}, AUPRC: {auprc_ppi_neg:.4f}')
        
        # === 记录所有测试指标 ===
        self.log_dict({
            # Site metrics
            "acc_site/test": acc_site,
            "auroc_site/test": auroc_site,
            "auprc_site/test": auprc_site,
            
            # Pair metrics
            "acc_pair/test": acc_pair,
            "auroc_pair/test": auroc_pair,
            "auprc_pair/test": auprc_pair,
            
            # Surface site metrics
            "acc_surface_site/test": acc_surface_site,
            "auroc_surface_site/test": auroc_surface_site,
            "auprc_surface_site/test": auprc_surface_site,
            "f1_surface_site/test": f1_surface_site,
            
            # Surface pair metrics
            "acc_surface/test": acc_surface,
            "auroc_surface/test": auroc_surface,
            "auprc_surface/test": auprc_surface,
            
            # Graph seed complementarity
            "auroc_graph_seed1/test": auroc_graph_seed1,
            "auroc_graph_seed2/test": auroc_graph_seed2,
            "auprc_graph_seed1/test": auprc_graph_seed1,
            "auprc_graph_seed2/test": auprc_graph_seed2,
            
            # Surface seed complementarity
            "auroc_surface_seed1/test": auroc_surface_seed1,
            "auroc_surface_seed2/test": auroc_surface_seed2,
            "auprc_surface_seed1/test": auprc_surface_seed1,
            "auprc_surface_seed2/test": auprc_surface_seed2,
            
            # PPI metrics - Overall
            "acc_ppi/test": acc_ppi,
            "auroc_ppi/test": auroc_ppi,
            "auprc_ppi/test": auprc_ppi,
            
            # PPI metrics - Positive samples
            "acc_ppi_pos/test": acc_ppi_pos,
            "auroc_ppi_pos/test": auroc_ppi_pos,
            "auprc_ppi_pos/test": auprc_ppi_pos,
            
            # PPI metrics - Negative samples
            "acc_ppi_neg/test": acc_ppi_neg,
            "auroc_ppi_neg/test": auroc_ppi_neg,
            "auprc_ppi_neg/test": auprc_ppi_neg,
        }, on_epoch=True, batch_size=len(outputs_site))
        
        return loss