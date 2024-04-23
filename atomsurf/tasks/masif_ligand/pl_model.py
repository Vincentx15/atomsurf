# 3p
import pytorch_lightning as pl
# project
from atomsurf.networks.protein_encoder import ProteinEncoder


class MasifLigandModule(pl.LightningModule):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ProteinEncoder(hparams)

    def forward(self, x):
        pass
