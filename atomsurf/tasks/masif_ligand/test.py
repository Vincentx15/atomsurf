# std
import sys
from pathlib import Path
# 3p
import hydra
import torch
import pytorch_lightning as pl
import numpy as np
# project
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from atomsurf.utils.callbacks import CommandLoggerCallback
from pl_model import MasifLigandModule
from data_loader import MasifLigandDataModule

@hydra.main(config_path="conf", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    seed =cfg.seed
    pl.seed_everything(seed, workers=True)

    # init datamodule
    datamodule = MasifLigandDataModule(cfg)

    # init model
    model = MasifLigandModule(cfg)

    if torch.cuda.is_available():
        params = {"accelerator": "gpu", "devices": [cfg.device]}
    else:
        params = {}

    # init trainer
    trainer = pl.Trainer(**params)

    # test
    # trainer.test(model, ckpt_path="best", datamodule=datamodule)
    # print('*******last ckpt*******')
    # trainer.test(model, ckpt_path="last", datamodule=datamodule)
    trainer.test(model, ckpt_path=cfg.path_model, datamodule=datamodule)


if __name__ == "__main__":
    main()