import os
import sys

from collections import defaultdict
import numpy as np
import scipy
import torch

# project
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.tasks.abag.model import AbAgNet

class AbAgModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.criterion = torch.nn.MSELoss()
        self.model = AbAgNet(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)
        self.val_resdict = defaultdict(list)
        self.test_resdict = defaultdict(list)

    def step(self, batch):
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None, None
        
        positive_ab = batch.positive_abs_cdr
        positive_ag = batch.positive_ag
        pred_ab, pred_ag = self(batch)
        loss = self.criterion(outputs, scores)
        names = batch.name
        if torch.isnan(loss).any():
            print('Nan loss')
            return None, None, None, None
        return loss, outputs, scores, names

    def training_step(self, batch, batch_idx):
        loss, logits, scores, names = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.eval()
        loss, logits, scores, names = self.step(batch)
        if loss is None:
            return None
        for name, logit, score in zip(names, logits, scores):
            reslist = [logit.cpu().numpy(), score.cpu().numpy()]
            self.val_reslist.append(reslist)
            self.val_resdict[name[0:5]].append(reslist)
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        self.model.eval()
        loss, logits, scores, names = self.step(batch)
        if loss is None:
            return None
        for name, logit, score in zip(names, logits, scores):
            reslist = [logit.cpu().numpy(), score.cpu().numpy()]
            self.test_reslist.append(reslist)
            self.test_resdict[name[0:5]].append(reslist)
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
