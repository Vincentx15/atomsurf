# std
import sys
from pathlib import Path
# 3p
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# project
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).absolute().parents[3]))


from atomsurf.utils.callbacks import CommandLoggerCallback
from pl_model import MasifLigandModule
from data_loader import MasifLigandDataModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"

    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init model
    model = MasifLigandModule(cfg)

    # init logger
    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{accuracy_val:.2f}",
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor="accuracy_val",
        mode="max",
        save_last=True,
        save_top_k=cfg.train.save_top_k,
        verbose=False,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(monitor='accuracy_val',
                                                     patience=cfg.train.early_stoping_patience,
                                                     mode='max')

    callbacks = [lr_logger, checkpoint_callback, early_stop_callback, CommandLoggerCallback(command)]

    if torch.cuda.is_available():
        params = {"accelerator": "gpu", "devices": [cfg.device]}
    else:
        params = {}

    # init trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        # epochs, batch size and when to val
        max_epochs=cfg.epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        val_check_interval=cfg.train.val_check_interval,
        # just verbose to maybe be used
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        # auto_lr_find=cfg.train.auto_lr_find,
        log_every_n_steps=cfg.train.log_every_n_steps,
        max_steps=cfg.train.max_steps,
        # gradient clipping
        gradient_clip_val=cfg.train.gradient_clip_val,
        # detect NaNs
        detect_anomaly=cfg.train.detect_anomaly,
        # debugging
        overfit_batches=cfg.train.overfit_batches,
        # gpu
        **params,
    )

    # datamodule
    datamodule = MasifLigandDataModule(cfg)

    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()
