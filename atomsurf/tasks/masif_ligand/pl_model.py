import os
import sys
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# project
from atomsurf.tasks.masif_ligand.model import MasifLigandNet
from atomsurf.utils.data_utils import AtomPLModule
from atomsurf.utils.metrics import multi_class_eval


class MasifLigandModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model = MasifLigandNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)

    def step(self, batch):
        labels = batch.label
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        return loss, outputs, labels

    def get_metrics(self, logits, labels, prefix):
        logits, labels = torch.cat(logits, dim=0), torch.cat(labels, dim=0)
        accuracy_macro, accuracy_micro, accuracy_balanced, \
            precision_macro, precision_micro, \
            recall_macro, recall_micro, \
            f1_macro, f1_micro, \
            auroc_macro = multi_class_eval(logits, labels, K=7)
        self.log_dict({
            f"accuracy_balanced/{prefix}": accuracy_balanced,
            f"precision_micro/{prefix}": precision_micro,
            f"recall_micro/{prefix}": recall_micro,
            f"f1_micro/{prefix}": f1_micro,
            f"auroc_macro/{prefix}": auroc_macro,
        }, on_epoch=True, batch_size=len(logits))
