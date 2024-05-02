import torch.nn as nn
import hydra


class ProteinEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([hydra.utils.instantiate(x.instanciate, x.kwargs) for x in cfg.blocks])

    def forward(self, surface=None, graph=None):
        for block in self.blocks:
            surface, graph = block(surface, graph)
        return surface, graph
