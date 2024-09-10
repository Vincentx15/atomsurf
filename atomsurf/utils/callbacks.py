from pathlib import Path

# 3p
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger
import wandb


def add_wandb_logger(loggers, projectname,runname):
    # init logger
    wandb.init(reinit=True, entity='vincent-mallet-cri-lpi')
    wand_id = wandb.util.generate_id()
    tb_logger = loggers[-1]
    # run_name = f"{Path(tb_logger.log_dir).stem}"
    tags = []
    Path(tb_logger.log_dir).absolute().mkdir(parents=True, exist_ok=True)
    wandb_logger = WandbLogger(project=projectname, name=runname, tags=tags,
                               version=Path(tb_logger.log_dir).stem, id=wand_id,
                               save_dir=tb_logger.log_dir, log_model=False)
    loggers += [wandb_logger]


class CommandLoggerCallback(Callback):
    def __init__(self, command):
        self.command = command

    def setup(self, trainer, pl_module, stage):
        tensorboard = pl_module.loggers[0].experiment
        tensorboard.add_text("Command", self.command)
