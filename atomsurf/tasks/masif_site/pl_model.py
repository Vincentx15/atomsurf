import os
import sys

import torch
import torch.nn.functional as F

# project
from atomsurf.tasks.masif_site.model import MasifSiteNet
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.utils.metrics import compute_accuracy, compute_auroc


def masif_site_loss(preds, labels):
    # Inspired from dmasif https://github.com/FreyrS/dMaSIF/blob/master/data_iteration.py#L158

    # Get our predictions and corresponding binary labels
    pos_preds = preds[labels == 1]
    neg_preds = preds[labels == 0]
    pos_labels = torch.ones_like(pos_preds)
    neg_labels = torch.zeros_like(neg_preds)

    # Subsample majority class to get balanced loss
    n_points_sample = min(len(pos_labels), len(neg_labels))
    pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
    neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]
    pos_preds = pos_preds[pos_indices]
    pos_labels = pos_labels[pos_indices]
    neg_preds = neg_preds[neg_indices]
    neg_labels = neg_labels[neg_indices]

    # Compute loss on these prediction/GT pairs
    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])
    loss = F.binary_cross_entropy_with_logits(preds_concat, labels_concat)
    return loss, preds_concat, labels_concat


class MasifSiteModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = MasifSiteNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)

    def step(self, batch):
        if batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None
        labels = torch.concatenate(batch.label)
        out_surface_batch = self(batch)
        outputs = out_surface_batch.x.flatten()
        loss, outputs_concat, labels_concat = masif_site_loss(outputs, labels)
        # if torch.isnan(loss).any():
        #     print('Nan loss')
        #     return None, None, None
        return loss, outputs_concat, labels_concat

    def get_metrics(self, logits, labels, prefix):
        logits, labels = torch.cat(logits, dim=0), torch.cat(labels, dim=0)
        auroc = compute_auroc(predictions=logits, labels=labels)
        acc = compute_accuracy(predictions=logits, labels=labels, add_sigmoid=True)
        self.log_dict({f"auroc/{prefix}": auroc, f"acc/{prefix}": acc, }, on_epoch=True, batch_size=len(logits))
