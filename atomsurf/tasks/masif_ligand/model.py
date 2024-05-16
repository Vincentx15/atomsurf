import torch
import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder


class MasifLigandNet(torch.nn.Module):
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
            nn.Linear(cfg_head.encoded_dims, out_features=cfg_head.output_dims)
        ])

    def pool_lig(self, pos, processed, lig_coords):
        # find nearest neighbors between last layers encodings and position of the ligand
        with torch.no_grad():
            dists = torch.cdist(pos, lig_coords.float())
            min_indices = torch.topk(-dists, k=10, dim=0).indices.unique()
        return processed[min_indices]

    def forward(self, batch):
        # forward pass
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)

        # Now select and average the encoded surface features around each ligand pocket.
        pos_and_x = [(surf.verts, surf.x) for surf in surface.to_data_list()]
        pockets_embs = []
        for (pos, x), lig_coord in zip(pos_and_x, batch.lig_coord):
            selected = self.pool_lig(pos, x, lig_coord)
            pocket_emb = torch.mean(selected, dim=-2)
            pockets_embs.append(pocket_emb)
        x = torch.stack(pockets_embs)
        x = self.top_net(x)
        return x
