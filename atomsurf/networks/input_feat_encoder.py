import torch
import torch.nn as nn
from torch_scatter import scatter


class ChemGeomFeatEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        h_dim = hparams.h_dim
        dropout = hparams.dropout
        num_gdf = hparams.num_gdf
        num_signatures = hparams.num_signatures
        self.use_neigh = hparams.use_neigh
        chem_feat_dim = hparams.graph_feat_dim
        geom_feat_dim = hparams.surface_feat_dim

        # chem feat
        # chem_feat_dim = 2 + num_gdf * 2
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim,  h_dim),#2 * h_dim
            nn.Dropout(dropout),
            nn.BatchNorm1d( h_dim),#2 * h_dim
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # # geom feats
        # # geom_input_dim = num_gdf * 2 + num_signatures
        # self.geom_mlp = nn.Sequential(
        #     nn.Linear(geom_feat_dim, h_dim // 2),
        #     nn.Dropout(dropout),
        #     nn.BatchNorm1d(h_dim // 2),
        #     nn.SiLU(),
        #     nn.Linear(h_dim // 2, h_dim // 2),
        #     nn.Dropout(dropout),
        #     nn.BatchNorm1d(h_dim // 2),
        # ) 
        self.surf_chem_mlp = nn.Sequential(
            nn.Linear(60, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2*h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2*h_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        # geom feats
        # geom_input_dim = num_gdf * 2 + num_signatures
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
        )

        if self.use_neigh:
            # chem + geom feats
            self.feat_mlp = nn.Sequential(
                nn.Linear(h_dim + h_dim, h_dim),
                # nn.Linear(h_dim + h_dim // 2, h_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(h_dim),
                nn.SiLU(),
                nn.Linear(h_dim, h_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(h_dim),
            )

    def forward(self, surface, graph):
        # chem_feats, geom_feats, nbr_vids = graph.chem_feats, surface.geom_feats, surface.nbr_vids
        # surface_in = [mini_surface.x for mini_surface in surface]
        # chem_feats, geom_feats = graph.x, torch.concatenate(surface_in, dim=-2)
        chem_feats_graph, geom_feats = graph.x, surface.x
        chem_feats_suf = surface.testchem_feats

        # geometric features
        h_geom = self.geom_mlp(geom_feats)

        # chemical features
        chem_feats_graph = self.chem_mlp(chem_feats_graph)
        chem_feats_suf=chem_feats_suf.to(geom_feats.device,dtype=geom_feats.dtype)
        chem_feats_suf=self.surf_chem_mlp(chem_feats_suf)
        nbr_filter, nbr_core = chem_feats_suf.chunk(2, dim=-1)

        b_nbr_vids = []
        base_idx = 0
        for nbrvid,numverts in zip(surface.nbr_vid,surface.n_verts):
            b_nbr_vids.append(torch.from_numpy(nbrvid+int(base_idx)))
            base_idx+=numverts
        b_nbr_vids = torch.cat(b_nbr_vids).to(geom_feats.device)
        # If self-filter
        if self.use_neigh:
            nbr_vids = b_nbr_vids  # TODO: implement/fix
            nbr_filter = self.sigmoid(nbr_filter)
            nbr_core = self.softplus(nbr_core)
            chem_feats_suf = nbr_filter * nbr_core
            chem_feats_suf = scatter(chem_feats_suf, nbr_vids, dim=0, reduce="sum")
            h_geom = self.feat_mlp(torch.cat((chem_feats_suf, h_geom), dim=-1))
        else:
            h_geom = h_geom
            h_chem = nbr_filter

        # surface_out = torch.split(h_geom, [len(x) for x in surface_in], dim=-2)
        # for mini_surf_out, mini_surf in zip(surface_out, surface):
        #     mini_surf.x = mini_surf_out
        graph.x = chem_feats_graph
        surface.x = h_geom
        return surface, graph
