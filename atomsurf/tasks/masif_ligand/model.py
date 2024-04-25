import torch

from atomsurf.networks.protein_encoder import ProteinEncoder


class MasifLigandNet(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = ProteinEncoder(hparams)
        a=1


    def forward(self, batch):
        # forward pass
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)

        x = torch.stack([torch.mean(x, dim=-2) for x in processed])
        x = self.top_net(x)
        return x
