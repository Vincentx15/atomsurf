# 3p
from pytorch_lightning import Callback


class CommandLoggerCallback(Callback):
    def __init__(self, command):
        self.command = command

    def setup(self, trainer, pl_module, stage):
        tensorboard = pl_module.loggers[0].experiment
        tensorboard.add_text("Command", self.command)
