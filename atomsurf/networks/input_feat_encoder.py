import torch
import torch.nn as nn
from torch_scatter import scatter

from atomsurf.network_utils.communication.surface_graph_comm import SurfaceGraphCommunication
from atomsurf.network_utils.communication.utils_blocks import init_block


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
            nn.Linear(h_dim, 2 * h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * h_dim),
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
        chem_feats, geom_feats = graph.x, surface.x

        # geometric features
        h_geom = self.geom_mlp(geom_feats)

        # chemical features
        h_chem = self.chem_mlp(chem_feats)

        nbr_filter, nbr_core = h_chem.chunk(2, dim=-1)
        # If self-filter
        if self.use_neigh:
            nbr_vids = surface.nbr_vid  # TODO: implement/fix
            nbr_filter = self.sigmoid(nbr_filter)
            nbr_core = self.softplus(nbr_core)
            h_chem = nbr_filter * nbr_core
            h_chem_geom = scatter(h_chem, nbr_vids, dim=0, reduce="sum")
            h_geom = self.feat_mlp(torch.cat((h_chem_geom, h_geom), dim=-1))
        else:
            h_geom = h_geom
            h_chem = nbr_filter

        # surface_out = torch.split(h_geom, [len(x) for x in surface_in], dim=-2)
        # for mini_surf_out, mini_surf in zip(surface_out, surface):
        #     mini_surf.x = mini_surf_out
        graph.x = h_chem
        surface.x = h_geom
        return surface, graph


class HMRChemGeomFeatEncoder(SurfaceGraphCommunication):
    def __init__(self, hparams, **kwargs):

        use_bp = hparams.use_bp
        use_gvp = hparams.use_gvp if "use_gvp" in hparams else False
        h_dim = hparams.h_dim
        dropout = hparams.dropout
        chem_feat_dim = hparams.graph_feat_dim
        geom_feat_dim = hparams.surface_feat_dim

        # chem feat
        chem_mlp = HMR2LayerMLP([chem_feat_dim, h_dim, h_dim], dropout)

        # geom feats
        geom_mlp = HMR2LayerMLP([geom_feat_dim, h_dim, h_dim], dropout)

        # preprocess blocks
        s_pre_block = geom_mlp
        g_pre_block = chem_mlp

        # message passing blocks
        # * this version does not use self-loops, because we will be summing surface-level features with graph-level features (not good apriori)
        if use_bp:
            if use_gvp:
                bp_sg_block = init_block("gvp",
                                         dim_in=h_dim,
                                         dim_out=h_dim)
                bp_gs_block = init_block("gvp",
                                         dim_in=h_dim,
                                         dim_out=h_dim)
            else:
                bp_sg_block = init_block("gcn",
                                         use_gat=hparams.use_gat, use_v2=hparams.use_v2,
                                         dim_in=hparams.bp_s_dim_in, dim_out=hparams.bp_s_dim_out,
                                         add_self_loops=hparams.bp_self_loops, fill_value=hparams.bp_fill_value)
                bp_gs_block = init_block("gcn",
                                         use_gat=hparams.use_gat, use_v2=hparams.use_v2,
                                         dim_in=hparams.bp_g_dim_in, dim_out=hparams.bp_g_dim_out,
                                         add_self_loops=hparams.bp_self_loops, fill_value=hparams.bp_fill_value)
        else:
            bp_gs_block, bp_sg_block = None, None

        # post-process blocks
        # * skip connection is a bad design, summing surface-level features with graph-level features, the skip is done in two different spaces
        # * we will use concatenation instead

        # merge SG/GS features
        merge_sg = HMR2LayerMLP([h_dim * 2, h_dim * 2, h_dim], dropout)
        merge_gs = HMR2LayerMLP([h_dim * 2, h_dim * 2, h_dim], dropout)

        s_post_block = CatMergeBlock(merge_sg)
        g_post_block = CatMergeBlock(merge_gs)

        super().__init__(use_bp, use_gvp=use_gvp, bp_sg_block=bp_sg_block, bp_gs_block=bp_gs_block,
                         s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                         s_post_block=s_post_block, g_post_block=g_post_block,
                         neigh_thresh=hparams.neigh_thresh, sigma=hparams.sigma, **kwargs)


class HMR2LayerMLP(nn.Module):
    def __init__(self, layers, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.Dropout(dropout),
            nn.BatchNorm1d(layers[1]),
            nn.SiLU(),
            nn.Linear(layers[1], layers[2]),
            nn.Dropout(dropout),
            nn.BatchNorm1d(layers[2]),
        )

    def forward(self, x):
        return self.net(x)


class CatMergeBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x_in, x_out):
        return self.net(torch.cat((x_in, x_out), dim=-1))
