import os
import sys
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# project
from atomsurf.tasks.masif_ligand.model import MasifLigandNet
from atomsurf.utils.metrics import multi_class_eval


class MasifLigandModule(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model = MasifLigandNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)
        self.train_res = list()
        self.val_res = list()
        self.test_res = list()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):

        if batch is None:
            return None, None, None
        labels = batch.label
        # return None, None, None
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        # if torch.isnan(loss).any():
        #     print('Nan loss')
        #     return None, None, None
        return loss, outputs, labels

    # def on_after_backward(self):
    #     valid_gradients = True
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
    #             if not valid_gradients:
    #                 break
    #
    #     if not valid_gradients:
    #         print(f'Detected inf or nan values in gradients. not updating model parameters')
    #         self.zero_grad()

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        self.train_res.append((logits, labels))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            print("validation step skipped!")
            return None
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        self.val_res.append((logits, labels))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            return None
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        self.test_res.append((logits, labels))

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

    def on_train_epoch_end(self) -> None:
        logits, labels = [list(i) for i in zip(*self.train_res)]
        self.get_metrics(logits, labels, 'train')
        self.train_res = list()
        pass

    def on_validation_epoch_end(self) -> None:
        logits, labels = [list(i) for i in zip(*self.val_res)]
        self.get_metrics(logits, labels, 'val')
        self.val_res = list()
        pass

    def on_test_epoch_end(self) -> None:
        logits, labels = [list(i) for i in zip(*self.test_res)]
        self.get_metrics(logits, labels, 'test')
        self.test_res = list()
        pass

    def configure_optimizers(self):
        opt_params = self.hparams.cfg.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   patience=opt_params.patience,
                                                                   factor=opt_params.factor,
                                                                   mode='max')
        scheduler = {'scheduler': scheduler_obj,
                     'monitor': self.hparams.cfg.train.to_monitor,
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
