import os
import sys

from collections import defaultdict
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torchmetrics import AUROC, MatthewsCorrCoef

# project
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.tasks.abag.model import AbAgNet


def compute_BCE(pred, target):
    """
    Compute Binary Cross Entropy Loss

    :param pred:  prediction
    :param target:  target
    :return:      loss
    """
    num_pos = target.sum()
    numels = len(target)
    weight = numels / num_pos - 1 if num_pos.item() > 0 else num_pos
    # TODO handle all negative values
    if num_pos.item() == 0:
        print("All negative")
    loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)
    return loss


class AbAgModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = AbAgNet(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)
        self.val_resdict = defaultdict(list)
        self.test_resdict = defaultdict(list)

    def step(self, batch):
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None, None

        # Get targets, concatenate and keep track of system sizes
        label_abs_cdr = torch.cat(batch.label_abs_cdr)
        label_ags = torch.cat(batch.label_ags)
        len_cdrs = [len(x) for x in batch.label_abs_cdr]
        len_ags = [len(x) for x in batch.label_ags]

        # Make computation and compute losses
        pred_ab, pred_ag = self(batch)
        ab_loss = compute_BCE(pred_ab, label_abs_cdr)
        ag_loss = compute_BCE(pred_ag, label_ags)
        loss = ab_loss + 2 * ag_loss

        # Split per system
        label_abs = [x.detach().cpu() for x in batch.label_abs_cdr]
        label_ags = [x.detach().cpu() for x in batch.label_ags]
        pred_ab = torch.split(pred_ab.detach().cpu(), len_cdrs)
        pred_ag = torch.split(pred_ag.detach().cpu(), len_ags)

        if torch.isnan(loss).any():
            print('Nan loss')
            return None, None, None
        return ((ab_loss, ag_loss),
                (pred_ab, label_abs),
                (pred_ag, label_ags))

    def training_step(self, batch, batch_idx):
        losses, abs, ags = self.step(batch)
        if losses is None:
            return None
        self.log_dict({"loss/ab_loss": losses[0].item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(abs[0]))
        self.log_dict({"loss/ag_loss": losses[1].item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(ags[0]))
        return losses[0] + 2 * losses[1]

    def validation_step(self, batch, batch_idx: int):
        self.model.eval()
        losses, abs, ags = self.step(batch)
        if losses is None:
            return None
        losses_values = (losses[0].item(), losses[1].item())
        self.val_res.append((losses_values, abs, ags))

    def test_step(self, batch, batch_idx: int):
        self.model.eval()
        losses, abs, ags = self.step(batch)
        if losses is None:
            return None
        losses_values = (losses[0].item(), losses[1].item())
        self.test_res.append((losses_values, abs, ags))

    def get_metrics(self, results, prefix):
        # Unwrap on at the individual level
        all_losses = torch.tensor([batch_res[0] for batch_res in results])
        ab_loss, ag_loss = torch.mean(all_losses, dim=1).tolist()

        # list of list becomes list of logits/labels
        all_abs_logits = [x for batch_res in results for x in batch_res[1][0]]
        all_abs_labels = [x for batch_res in results for x in batch_res[1][1]]

        all_ags_logits = [x for batch_res in results for x in batch_res[2][0]]
        all_ags_labels = [x for batch_res in results for x in batch_res[2][1]]

        auroc_computer = AUROC(task="binary")
        mcc_computer = MatthewsCorrCoef(task="binary")
        auroc_abs = [auroc_computer(log, lab) for log, lab in zip(all_abs_logits, all_abs_labels)]
        mcc_abs = [mcc_computer(log, lab) for log, lab in zip(all_abs_logits, all_abs_labels)]
        auroc_ags = [auroc_computer(log, lab) for log, lab in zip(all_ags_logits, all_ags_labels)]
        mcc_ags = [mcc_computer(log, lab) for log, lab in zip(all_ags_logits, all_ags_labels)]
        auroc_abs, mcc_abs, auroc_ags, mcc_ags = [torch.mean(torch.stack(x)) for x in
                                                  (auroc_abs, mcc_abs, auroc_ags, mcc_ags)]

        self.log_dict({f"ab_loss/{prefix}": ab_loss,
                       f"ag_loss/{prefix}": ag_loss,
                       f"ag_auroc/{prefix}": auroc_abs,
                       f"ag_auroc/{prefix}": auroc_ags,
                       f"ab_mcc/{prefix}": mcc_abs,
                       f"ag_mcc/{prefix}": mcc_ags,
                       }, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if len(self.val_res) > 0:
            self.get_metrics(self.val_res, 'val')
            self.val_res = list()

    def on_test_epoch_end(self) -> None:
        if len(self.test_res) > 0:
            self.get_metrics(self.test_res, 'test')
            self.test_res = list()
