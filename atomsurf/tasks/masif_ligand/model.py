import torch


class MasifLigandNet(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def forward(self, batch):
        raise NotImplementedError
