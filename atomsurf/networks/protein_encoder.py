import torch.nn as nn
import hydra


class ProteinEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.network = nn.Sequential([hydra.utils.instantiate(x) for x in hparams.achitecture.blocks])

    def forward(self, surface=None, graph=None):
        return self.network(surface, graph)
