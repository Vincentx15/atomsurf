import torch
import torch.nn as nn
from torch_scatter import scatter

from atomsurf.network_utils.communication.surface_graph_comm import SurfaceGraphCommunication
from atomsurf.network_utils.communication.utils_blocks import init_block
from atomsurf.network_utils.communication.passing_utils import _rbf


class ChemGeomFeatEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        h_dim = hparams.h_dim
        dropout = hparams.dropout
        self.num_gdf = hparams.num_gdf
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
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
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
            self.surf_chem_mlp = nn.Sequential(
                nn.Linear(chem_feat_dim + 2 * self.num_gdf, h_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(h_dim),
                nn.SiLU(),
                nn.Linear(h_dim, 2 * h_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(2 * h_dim),
            )

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
        chem_feats, geom_feats = graph.x, surface.x
        # First, let us get geom and chem features
        h_geom = self.geom_mlp(geom_feats)
        h_chem = self.chem_mlp(chem_feats)

        # If we additionally use neighboring info, we need to compute it and propagate a message that
        # uses dists and angles
        if self.use_neigh:
            # TODO: UNSAFE, if normals are useful, find a better way (probably by including it as a surface attribute)
            verts_list = [surf.verts for surf in surface.to_data_list()]
            nodepos_list = [gr.node_pos for gr in graph.to_data_list()]
            with torch.no_grad():
                all_dists = [torch.cdist(vert, nodepos) for vert, nodepos in zip(verts_list, nodepos_list)]
                neighbors = []
                offset_surf, offset_graph = 0, 0
                for x in all_dists:
                    k = 16
                    # these are the 16 closest point in the graph for each vertex
                    min_indices = torch.topk(-x, k=k, dim=1).indices
                    vertex_ids = torch.arange(0, len(x), device=x.device)
                    repeated_tensor = vertex_ids.repeat_interleave(k)
                    min_indices += offset_graph
                    repeated_tensor += offset_surf
                    offset_surf += len(x)
                    offset_graph += len(x.T)
                    neighbors.append(torch.stack((repeated_tensor, min_indices.flatten())))
            # Slicing requires tuple
            neighbors = torch.cat([x for x in neighbors], dim=1).long()
            neigh_verts, neigh_graphs = neighbors[0, :], neighbors[1, :]

            # extract relevant chem features
            all_chem_feats = chem_feats[neigh_graphs]
            verts_normals = torch.cat([surf.x[:, -3:] for surf in surface.to_data_list()], dim=0)
            all_normals = verts_normals[neigh_verts]
            edge_vecs = graph.node_pos[neigh_graphs] - surface.verts[neigh_verts]
            edge_dists = torch.linalg.norm(edge_vecs, axis=-1)
            normed_edge_vecs = edge_vecs / edge_dists[:, None]
            nbr_angular = torch.einsum('vj,vj->v', normed_edge_vecs, all_normals)

            # Compute expanded message feature, and concatenate
            encoded_dists = _rbf(edge_dists, D_min=0, D_max=8, D_count=self.num_gdf)
            encoded_angles = _rbf(nbr_angular, D_min=-1, D_max=1, D_count=self.num_gdf)
            expanded_message_features = torch.cat([all_chem_feats, encoded_dists, encoded_angles], axis=-1)

            # Now use MLP to modulate those messages, with self filtering, and aggregate them over vertices.
            embedded_messages = self.surf_chem_mlp(expanded_message_features)
            nbr_filter, nbr_core = embedded_messages.chunk(2, dim=-1)
            nbr_filter = self.sigmoid(nbr_filter)
            nbr_core = self.softplus(nbr_core)
            h_chem_geom = nbr_filter * nbr_core
            h_chem_geom = scatter(h_chem_geom, neigh_verts, dim=0, reduce="sum")
            h_geom = self.feat_mlp(torch.cat((h_chem_geom, h_geom), dim=-1))
        graph.x = h_chem
        surface.x = h_geom
        return surface, graph


class HMRChemGeomFeatEncoder(SurfaceGraphCommunication):
    def __init__(self, hparams, **kwargs):

        use_bp = hparams.use_bp
        use_gvp = hparams.use_gvp if "use_gvp" in hparams else False
        use_normals = hparams.use_normals if "use_gvp" in hparams else False
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
                                         dim_out=h_dim,
                                         use_normals=use_normals)
                bp_gs_block = init_block("gvp",
                                         dim_in=h_dim,
                                         dim_out=h_dim,
                                         use_normals=use_normals)
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
                         neigh_thresh=hparams.neigh_thresh, sigma=hparams.sigma,
                         use_knn=hparams.use_knn, **kwargs)


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
