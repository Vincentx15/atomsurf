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

from atomsurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger
from pl_model import MasifLigandModule
from data_loader import MasifLigandDataModule

torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="conf", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"

    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init datamodule
    datamodule = MasifLigandDataModule(cfg)

    # init model
    model = MasifLigandModule(cfg)

    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{cfg.run_name}" if not cfg.use_wandb else f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    if cfg.use_wandb:
        add_wandb_logger(loggers)

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{accuracy_balanced/val:.2f}",
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor=cfg.train.to_monitor,
        mode="max",
        save_last=True,
        save_top_k=cfg.train.save_top_k,
        verbose=False,
    )
    early_stop_callback = pl.callbacks.EarlyStopping(monitor=cfg.train.to_monitor,
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

    # train
    trainer.fit(model, datamodule=datamodule)

    # test
    print('*****************test best ckpt*****************')
    trainer.test(model, ckpt_path="best", datamodule=datamodule)
    print('*****************test last ckpt*****************')
    trainer.test(model, ckpt_path="last", datamodule=datamodule)


if __name__ == "__main__":
    main()
