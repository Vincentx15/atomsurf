import torch
import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder


class MasifSiteNet(torch.nn.Module):
    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.hparams_head = cfg_head
        self.hparams_encoder = cfg_encoder
        self.encoder = ProteinEncoder(cfg_encoder)

        self.top_net = nn.Sequential(*[
            nn.Linear(cfg_head.encoded_dims, cfg_head.encoded_dims),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(cfg_head.encoded_dims),
            nn.SiLU(),
            nn.Linear(cfg_head.encoded_dims, out_features=cfg_head.output_dims),
            nn.Sigmoid()
        ])

    def forward(self, batch):
        # forward pass
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)
        import time
        import torch
        t1 = time.perf_counter()
        surface.x = self.top_net(surface.x)
        torch.cuda.synchronize()
        time_model = time.perf_counter() - t1
        print("Time top net", time_model)
        return surface
