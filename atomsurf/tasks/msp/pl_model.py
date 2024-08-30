import os
import sys

import torch

# project
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.tasks.msp.model import MSPNet
from atomsurf.utils.metrics import compute_auroc


class MSPModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.model = MSPNet(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)

    def step(self, batch):
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None
        labels = batch.label.reshape(-1, 1)
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        # names = batch.name
        if torch.isnan(loss).any():
            print('Nan loss')
            return None, None, None, None
        return loss, outputs, labels

    def get_metrics(self, logits, labels, prefix):
        logits, labels = torch.cat(logits, dim=0), torch.cat(labels, dim=0)
        auroc = compute_auroc(predictions=logits, labels=labels)
        self.log_dict({f"auroc/{prefix}": auroc}, on_epoch=True)