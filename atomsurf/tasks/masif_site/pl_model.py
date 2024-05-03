import os
import sys

import torch
import torch.nn.functional as F

# project
from atomsurf.tasks.masif_site.model import MasifSiteNet
from atomsurf.utils.data_utils import AtomPLModule
from atomsurf.utils.metrics import compute_accuracy, compute_auroc


def masif_site_loss(preds, labels):
    # Inspired from dmasif
    pos_preds = preds[labels == 1]
    pos_labels = torch.ones_like(pos_preds)
    neg_preds = preds[labels == 0]
    neg_labels = torch.zeros_like(pos_preds)
    n_points_sample = min(len(pos_labels), len(neg_labels))
    pos_indices = torch.randperm(len(pos_labels))[:n_points_sample]
    neg_indices = torch.randperm(len(neg_labels))[:n_points_sample]
    pos_preds = pos_preds[pos_indices]
    pos_labels = pos_labels[pos_indices]
    neg_preds = neg_preds[neg_indices]
    neg_labels = neg_labels[neg_indices]
    preds_concat = torch.cat([pos_preds, neg_preds])
    labels_concat = torch.cat([pos_labels, neg_labels])
    loss = F.binary_cross_entropy_with_logits(preds_concat, labels_concat)
    return loss, preds_concat, labels_concat


class MasifSiteModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.save_hyperparameters()
        self.model = MasifSiteNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        if batch is None:
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
        auroc = compute_auroc(labels, logits)
        acc = compute_accuracy(labels, logits)
        self.log_dict({
            f"auroc/{prefix}": auroc,
            f"acc/{prefix}": acc,
        }, on_epoch=True, batch_size=len(logits))
