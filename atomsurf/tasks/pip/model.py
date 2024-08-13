import torch
import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder


class PIPNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)

        self.hparams_head = hparams_head
        encoded_dims = hparams_head.encoded_dims
        self.top_net = nn.Sequential(*[
            nn.Linear(encoded_dims, encoded_dims),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(encoded_dims, 1)
        ])

    def project_processed_surface(self, pos, processed, lig_coords):
        # find nearest neighbors between doing last layers
        with torch.no_grad():
            dists = torch.cdist(pos, lig_coords.float())
            min_indices = torch.topk(-dists, k=10, dim=0).indices.unique()
        return processed[min_indices]

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
        batch.idx_left = torch.cat(batch.idx_left)
        batch.idx_right = torch.cat(batch.idx_right)
        processed_left = graph_1.x[batch.idx_left]
        # min_indices = torch.argmin(dists, dim=1)
        # processed_left = graph_1.x[min_indices]
        # dists = torch.cdist(torch.cat(batch.locs_right), graph_2.node_pos)
        # min_indices = torch.argmin(dists, dim=1)
        # processed_right = graph_2.x[min_indices]s
        processed_right = graph_2.x[batch.idx_right]
        x = torch.cat([processed_left, processed_right], dim=1)
        x = self.top_net(x)
        return x
