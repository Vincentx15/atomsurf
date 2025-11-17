import torch
import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder


class PIPNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        # in_features = 128 * 2  # 12
        in_features = hparams_head.encoded_dims * 2
        self.top_net = nn.Sequential(*[
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])

    def forward(self, batch):
        # forward pass
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        surface_2, graph_2 = self.encoder(graph=batch.graph_2, surface=batch.surface_2)
        # update idx
        base_l = torch.cumsum(batch.g1_len, dim=0)
        base_r = torch.cumsum(batch.g2_len, dim=0)
        for i in range(1, len(batch.idx_left)):
            batch.idx_left[i] = (batch.idx_left[i] + base_l[i - 1])
            batch.idx_right[i] = (batch.idx_right[i] + base_r[i - 1])
        # import pdb
        # pdb.set_trace()
        if isinstance(batch.idx_left, list):
            batch.idx_left = torch.cat(batch.idx_left)
            batch.idx_right = torch.cat(batch.idx_right)
        else:
            batch.idx_left = batch.idx_left.reshape(-1)
            batch.idx_right = batch.idx_right.reshape(-1)
        processed_left = graph_1.x[batch.idx_left]
        processed_right = graph_2.x[batch.idx_right]
        x = torch.cat([processed_left, processed_right], dim=1)
        x = self.top_net(x)
        return x
