import torch
import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder


class MasifLigandNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        in_features = 12
        out_features = 7
        self.top_net = nn.Sequential(*[
            nn.Linear(in_features, in_features),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(in_features),
            nn.SiLU(),
            nn.Linear(in_features, out_features=out_features)
        ])

    def pool_lig(self, pos, processed, lig_coords):
        # find nearest neighbors between doing last layers
        with torch.no_grad():
            dists = torch.cdist(pos, lig_coords.float())
            min_indices = torch.topk(-dists, k=10, dim=0).indices.unique()
        return processed[min_indices]

    def forward(self, batch):
        # forward pass
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)

        pos_and_x = [(surf.verts, surf.x) for surf in surface.to_data_list()]
        pockets_embs = []
        for (pos, x), lig_coord in zip(pos_and_x, batch.lig_coord):
            selected = self.pool_lig(pos, x, lig_coord)
            pocket_emb = torch.mean(selected, dim=-2)
            pockets_embs.append(pocket_emb)
        x = torch.stack(pockets_embs)
        x = self.top_net(x)
        return x
