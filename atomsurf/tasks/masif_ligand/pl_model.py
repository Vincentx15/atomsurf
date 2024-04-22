# import torch
import pytorch_lightning as pl


class MasifLigandModule(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x):
        pass