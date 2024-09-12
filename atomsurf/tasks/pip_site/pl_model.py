import os
import sys

import torch

# project
from atomsurf.tasks.pip_site.model import PIPsiteNet
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.utils.metrics import compute_auroc, compute_accuracy


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
