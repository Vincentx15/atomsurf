import os
import sys

from collections import defaultdict
import numpy as np
import scipy
import torch

# project
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.tasks.psr.model import PSRNet


def safe_spearman(gt, pred):
    if np.all(np.isclose(pred, pred[0])):
        return 0
    return scipy.stats.spearmanr(pred, gt).statistic


def rs_metric(reslist, resdict):
    if len(reslist) == 0:
        return 0, 0
    all_lists = np.array(reslist)
    gt, pred = all_lists[:, 0], all_lists[:, 1]
    global_r = safe_spearman(gt, pred)
    local_r = []
    for system, lists in resdict.items():
        lists = np.array(lists)
        gt, pred = lists[:, 0], lists[:, 1]
        local_r.append(safe_spearman(gt, pred))
    local_r = float(np.mean(local_r))
    return global_r, local_r


class PSRModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.criterion = torch.nn.MSELoss()
        self.model = PSRNet(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)

        self.val_reslist = list()
        self.val_resdict = defaultdict(list)
        self.test_reslist = list()
        self.test_resdict = defaultdict(list)

    def step(self, batch):
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None, None
        scores = batch.score.reshape(-1, 1)
        outputs = self(batch)
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

    def on_validation_epoch_end(self):
        global_r, local_r = rs_metric(self.val_reslist, self.val_resdict)
        self.val_reslist = list()
        self.val_resdict = defaultdict(list)
        print(f" Global R validation: {global_r}")
        print(f" Local R validation : {local_r}")
        self.log_dict({"global_r/val": global_r})
        self.log_dict({"local_r/val": local_r})
        self.log("global_r_val", global_r, prog_bar=True, on_step=False, on_epoch=True, logger=False)

    def on_test_epoch_end(self) -> None:
        global_r, local_r = rs_metric(self.test_reslist, self.test_resdict)
        self.test_reslist = list()
        self.test_resdict = defaultdict(list)
        print(f" Global R test: {global_r}")
        print(f" Local R test : {local_r}")
        self.log_dict({"global_r/test": global_r})
        self.log_dict({"local_r/test": local_r})
