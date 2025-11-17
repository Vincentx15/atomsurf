import os
import sys
# std
from pathlib import Path
# 3p
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
import warnings

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# project
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from atomsurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger
from pl_model import PINDERModule_seed,PINDERModule_seed_ppi
from data_loader import PINDERDataModule

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(config_path="conf", config_name="config")
def main(cfg=None):
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Process {os.getpid()}: Using GPU {local_rank}")
    command = f"python3 {' '.join(sys.argv)}"
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    # init datamodule
    datamodule = PINDERDataModule(cfg)

    # init model
    model = PINDERModule_seed_ppi(cfg)

    # init logger
    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    if cfg.use_wandb:
        add_wandb_logger(loggers, projectname='pip_site',runname=version_name)

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="epoch={epoch:02d}-auprc_val={auprc_surface_site/val:.4f}",  # 使用 4 位小数
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor="auprc_surface_site/val",
        mode="max",
        save_last=True,
        save_top_k=cfg.train.save_top_k,
        verbose=True,  
        save_on_train_epoch_end=False,  # 🔥 在验证结束后保存
        auto_insert_metric_name=False,  # 🔥 避免自动插入 metric 名称
    )

    # early_stop_callback = pl.callbacks.EarlyStopping(monitor='auprc_surface_site/val',
    #                                                  patience=cfg.train.early_stoping_patience,
    #                                                  mode='max')

    callbacks = [lr_logger, checkpoint_callback,  CommandLoggerCallback(command)] #early_stop_callback

    # if torch.cuda.is_available():
    #     params = {"accelerator": "gpu", "devices": [cfg.device]}
    # else:
    #     params = {}

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
        # monitor time
        # profiler="simple",
        # gpu
        accelerator='gpu',
        # strategy=pl.strategies.DDPStrategy(find_unused_parameters=False,),
        devices=1,
        strategy='ddp',
        num_nodes=1,
        # precision=16, 
        # **params,
    )

    # train
    if os.path.exists(cfg.path_model):
        trainer.fit(model, datamodule=datamodule,ckpt_path=cfg.path_model)
    else:
        trainer.fit(model, datamodule=datamodule)

    # test
    trainer.test(model, ckpt_path="best", datamodule=datamodule)
    trainer.test(model, ckpt_path="last", datamodule=datamodule)


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high') 
    main()
# cd /work/lpdi/users/ymiao/code/newcode_0923/atomsurf/atomsurf/tasks/pip_site/ && python train_pinder_seed.py data_dir=/work/lpdi/users/ymiao/code/pinderdata cfg_graph.use_esm=True optimizer.lr=5e-4 run_name=test_pinder_seed_addppi device=0 out_dir=/work/lpdi/users/ymiao/code/sbatch/pinderlog  log_dir=/work/lpdi/users/ymiao/code/sbatch/pinderlog/ loader.batch_size=32 loader.num_workers=30 epochs=30 encoder=pronet_hmrencoder use_wandb=False cfg_surface.data_dir=/work/lpdi/users/ymiao/code/pinderdata/ cfg_graph.data_dir=/work/lpdi/users/ymiao/code/pinderdata/ cfg_graph.esm_dir=/work/lpdi/users/ymiao/code/pinderdata/esm  loader.prefetch_factor=5 seed=42