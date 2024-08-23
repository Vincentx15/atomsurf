import os
import sys

from collections import defaultdict
import numpy as np
import pytorch_lightning as pl
import scipy
import torch

# project
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


class PSRModule(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.MSELoss()
        self.model = PSRNet(hparams_encoder=hparams.encoder, hparams_head=hparams.cfg_head, use_graph_only=True)
        self.val_reslist = list()
        self.val_resdict = defaultdict(list)

        self.test_reslist = list()
        self.test_resdict = defaultdict(list)

    def forward(self, x):
        return self.model(x)

    def step(self, batch):

        if batch is None:
            return None, None, None
        if len(set(batch.graph.batch.cpu().numpy())) < 2:
            return None, None, None
        scores = batch.score.reshape(-1, 1)
        outputs = self(batch)
        loss = self.criterion(outputs, scores)
        names = batch.name
        if torch.isnan(loss).any():
            print('Nan loss')
            return None, None, None
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
        if loss is None or logits.isnan().any() or scores.isnan().any():
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
        if loss is None or logits.isnan().any() or scores.isnan().any():
            return None
        for name, logit, score in zip(names, logits, scores):
            reslist = [logit.cpu().numpy(), score.cpu().numpy()]
            self.test_reslist.append(reslist)
            self.test_resdict[name[0:5]].append(reslist)
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))

    def configure_optimizers(self):
        opt_params = self.hparams.hparams.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   patience=opt_params.patience,
                                                                   factor=opt_params.factor,
                                                                   mode='max')
        scheduler = {'scheduler': scheduler_obj,
                     'monitor': "loss/val",
                     'interval': "epoch",
                     'frequency': 1,
                     "strict": True,
                     'name': "epoch/lr"}
        # return optimizer
        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if batch is None:
            return None
        batch = batch.to(device)
        return batch

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

    # def get_metrics(self, logits, scores, prefix):
    #     if prefix=='val':
    #         global_r, local_r = rs_metric(self.val_reslist, self.val_resdict)
    #         self.val_reslist = list()
    #         self.val_resdict = defaultdict(list)
    #         print(f" Global R validation: {global_r}")
    #         print(f" Local R validation : {local_r}")
    #         self.log_dict({"global_r/val": global_r})
    #         self.log_dict({"local_r/val": local_r})
    #         self.log("global_r_val", global_r, prog_bar=True, on_step=False, on_epoch=True, logger=False)

    #     elif prefix=='test':
    #         global_r, local_r = rs_metric(self.test_reslist, self.test_resdict)
    #         self.test_reslist = list()
    #         self.test_resdict = defaultdict(list)
    #         print(f" Global R test: {global_r}")
    #         print(f" Local R test : {local_r}")
    #         self.log_dict({"global_r/test": global_r})
    #         self.log_dict({"local_r/test": local_r})
