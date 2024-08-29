import torch
import torch.nn as nn

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
        # Calculate the mean pooling by dividing sum by count
        sum_result = torch.zeros(graph.batch.max() + 1, graph.x.shape[1]).to(graph.x.device)
        count_result = torch.zeros(graph.batch.max() + 1, graph.x.shape[1]).to(graph.x.device)
        sum_result.scatter_add_(0, graph.batch.unsqueeze(-1).expand_as(graph.x), graph.x)
        count_result.scatter_add_(0, graph.batch.unsqueeze(-1).expand_as(graph.x), torch.ones_like(graph.x))
        mean_result = sum_result / count_result
        mean_result = self.top_net(mean_result)
        return mean_result
