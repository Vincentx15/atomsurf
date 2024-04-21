import torch
import torch.nn as nn
from torch_scatter import scatter


class ChemGeomFeatEncoder(nn.Module):
    def __init__(self, h_dim=128, dropout=0.1, num_gdf=16, num_signatures=16):
        super().__init__()

        # chem feat
        chem_feat_dim = 2 + num_gdf * 2
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * h_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # geom feats
        geom_input_dim = num_gdf * 2 + num_signatures
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_input_dim, h_dim // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, h_dim // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim // 2),
        )

        # chem + geom feats
        self.feat_mlp = nn.Sequential(
            nn.Linear(h_dim + h_dim // 2, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
        )

    def forward(self, surface, graph):
        chem_feats, geom_feats, nbr_vids = surface.chem_feats, surface.geom_feats, surface.nbr_vids
        # chemical features
        h_chem = self.chem_mlp(chem_feats)

        # self-filter
        nbr_filter, nbr_core = h_chem.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h_chem = nbr_filter * nbr_core
        h_chem = scatter(h_chem, nbr_vids, dim=0, reduce="sum")

        # geometric features
        h_geom = self.geom_mlp(geom_feats)

        # combine chemical and geometric features
        h_out = self.feat_mlp(torch.cat((h_chem, h_geom), dim=-1))

        surface.x = h_out
        return surface, graph
