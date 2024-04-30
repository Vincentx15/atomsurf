import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# project
from atomsurf.tasks.masif_ligand.model import MasifLigandNet
from atomsurf.utils.metrics import compute_auroc, compute_accuracy


class MasifLigandModule(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.model = MasifLigandNet(hparams_encoder=hparams.encoder, hparams_head=hparams.cfg_head)

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
        # acc = compute_accuracy(logits, labels)
        # auroc = compute_auroc(logits, labels)
        # self.log_dict({"acc/train": acc, "auroc/train": auroc}, on_epoch=True, batch_size=len(logits))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        # acc = compute_accuracy(logits, labels) # TODO FIX
        # auroc = compute_auroc(logits, labels)
        # self.log_dict({"acc/val": acc, "auroc/val": auroc}, on_epoch=True, batch_size=len(logits))
        # self.log("auroc_val", auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            self.log("acc/test", 0.5, on_epoch=True)
            return None
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        # acc = compute_accuracy(logits, labels) # TODO FIX
        # auroc = compute_auroc(logits, labels)
        # self.log_dict({"acc/test": acc, "auroc/test": auroc}, on_epoch=True, batch_size=len(logits))

    def configure_optimizers(self):
        opt_params = self.hparams.hparams.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   patience=opt_params.patience,
                                                                   factor=opt_params.factor,
                                                                   mode='max')
        scheduler = {'scheduler': scheduler_obj,
                     'monitor': "auroc_val",
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
