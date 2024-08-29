import torch
import torch.nn as nn
import torch_geometric

from atomsurf.networks.protein_encoder import ProteinEncoder


class PSRNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        self.sigma = 2.5
        in_features = hparams_head.encoded_dims  # 12
        self.top_net = nn.Sequential(*[
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features // 2, 1)
        ])

    def forward(self, batch):
        # forward pass
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)
        mean_result = torch_geometric.nn.pool.global_mean_pool(graph.x, graph.batch)
        predictions = self.top_net(mean_result)
        return predictions
