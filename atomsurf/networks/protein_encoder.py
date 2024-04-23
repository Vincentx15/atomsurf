import torch.nn as nn
import hydra


class ProteinEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.blocks = [hydra.utils.instantiate(x.instanciate, x.kwargs) for x in hparams.encoder.blocks]

    def forward(self, surface=None, graph=None):
        for block in self.blocks:
            surface, graph = block(surface, graph)

        return surface, graph
