import torch.nn as nn
import hydra


class ProteinEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([hydra.utils.instantiate(x.instanciate, x.kwargs) for x in cfg.blocks])

    def forward(self, surface=None, graph=None):
        for i, block in enumerate(self.blocks):
            import time
            import torch
            t1 = time.perf_counter()
            surface, graph = block(surface, graph)
            torch.cuda.synchronize()
            time_model = time.perf_counter() - t1
            print("Time model", i, time_model)
        return surface, graph
