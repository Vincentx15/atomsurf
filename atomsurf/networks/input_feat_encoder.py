import torch
import torch.nn as nn
from torch_scatter import scatter

from atomsurf.network_utils.communication.surface_graph_comm import SurfaceGraphCommunication
from atomsurf.network_utils.communication.utils_blocks import init_block
from atomsurf.network_utils.communication.passing_utils import _rbf


class HMRInputEncoder(nn.Module):
    def __init__(self, dropout=0.1, graph_feat_dim=30, h_dim=128, num_gdf=16, surface_feat_dim=22,
                 use_neigh=True, **kwargs):
        super().__init__()
        self.num_gdf = num_gdf
        self.use_neigh = use_neigh

        h_dim = h_dim
        dropout = dropout
        chem_feat_dim = graph_feat_dim
        geom_feat_dim = surface_feat_dim

        # chem feat
        # chem_feat_dim = 2 + num_gdf * 2
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * h_dim)
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
                nn.Linear(h_dim + 2 * self.num_gdf, h_dim),  # chem_feat_dim
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
        # First, let us get geom and chem features
        chem_feats, geom_feats = graph.x, surface.x
        h_geom = self.geom_mlp(geom_feats)

        # HMR introduced this weird self filtering
        h_chem = self.chem_mlp(chem_feats)
        nbr_filter, nbr_core = h_chem.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h_chem = nbr_filter * nbr_core

        # If we additionally use neighboring info, we need to compute it and propagate a message that
        # uses dists and angles
        if self.use_neigh:
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
            all_chem_feats = h_chem[neigh_graphs]  # chem_feats
            verts_normals = surface.vnormals
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
            del neighbors, neigh_verts, neigh_graphs, all_chem_feats, verts_normals, all_normals, edge_vecs, edge_dists, normed_edge_vecs, nbr_angular, encoded_dists, encoded_angles, expanded_message_features, embedded_messages, nbr_filter, nbr_core, h_chem_geom
            torch.cuda.empty_cache()
        graph.x = h_chem
        surface.x = h_geom
        del h_chem, h_geom
        torch.cuda.empty_cache()
        return surface, graph


class InputEncoder(SurfaceGraphCommunication):
    def __init__(self, surface_feat_dim=22, graph_feat_dim=30, h_dim=128, dropout=0.1,
                 use_gvp=False, use_normals=False, gvp_use_angles=False,
                 n_layers=3, vector_gate=False,
                 use_gat=True, use_v2=False,
                 add_self_loops=False, fill_value="mean",
                 neigh_thresh=8, sigma=2.5, use_knn=False, num_gdf=16,
                 **kwargs):

        # chem feat
        chem_mlp = HMR2LayerMLP([graph_feat_dim, h_dim, h_dim], dropout)

        # geom feats
        geom_mlp = HMR2LayerMLP([surface_feat_dim, h_dim, h_dim], dropout)

        # preprocess blocks
        s_pre_block = geom_mlp
        g_pre_block = chem_mlp

        # message passing blocks
        # * this version does not use self-loops, because we will be summing surface-level features with graph-level features (not good apriori)
        if use_gvp:
            bp_sg_block = init_block("gvp",
                                     dim_in=h_dim,
                                     dim_out=h_dim,
                                     gvp_use_angles=gvp_use_angles,
                                     use_normals=use_normals, n_layers=n_layers, vector_gate=vector_gate)
            bp_gs_block = init_block("gvp",
                                     dim_in=h_dim,
                                     dim_out=h_dim,
                                     gvp_use_angles=gvp_use_angles,
                                     use_normals=use_normals, n_layers=n_layers, vector_gate=vector_gate)
        else:
            bp_sg_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=h_dim, dim_out=h_dim,
                                     add_self_loops=add_self_loops, fill_value=fill_value)
            bp_gs_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=h_dim, dim_out=h_dim,
                                     add_self_loops=add_self_loops, fill_value=fill_value)

        # post-process blocks
        # * skip connection is a bad design, summing surface-level features with graph-level features, the skip is done in two different spaces
        # * we will use concatenation instead

        # merge SG/GS features
        merge_sg = HMR2LayerMLP([h_dim * 2, h_dim * 2, h_dim], dropout)
        merge_gs = HMR2LayerMLP([h_dim * 2, h_dim * 2, h_dim], dropout)

        s_post_block = CatMergeBlock(merge_sg)
        g_post_block = CatMergeBlock(merge_gs)

        super().__init__(bp_sg_block=bp_sg_block, bp_gs_block=bp_gs_block,
                         s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                         s_post_block=s_post_block, g_post_block=g_post_block,
                         neigh_thresh=neigh_thresh, sigma=sigma, num_gdf=num_gdf,
                         use_knn=use_knn, **kwargs)


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


class BiHMRInputEncoder(nn.Module):
    def __init__(self, graph_feat_dim=30, surface_feat_dim=22, h_dim=128, dropout=0.1, num_gdf=16,
                 use_neigh=True, **kwargs):
        super().__init__()

        self.num_gdf = num_gdf
        chem_feat_dim = graph_feat_dim
        geom_feat_dim = surface_feat_dim

        # chem feat
        # chem_feat_dim = 2 + num_gdf * 2
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * h_dim)
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * h_dim),
        )

        self.surf_chem_mlp = nn.Sequential(
            nn.Linear(h_dim + 2 * self.num_gdf, h_dim),  # chem_feat_dim
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * h_dim),
        )
        self.graph_geom_mlp = nn.Sequential(
            nn.Linear(h_dim + 2 * self.num_gdf, h_dim),  # chem_feat_dim
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2 * h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * h_dim),
        )
        # chem + geom feats
        self.feat_mlp_chem_geom = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            # nn.Linear(h_dim + h_dim // 2, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
        )
        self.feat_mlp_geom_chem = nn.Sequential(
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
        # First, let us get geom and chem features
        chem_feats, geom_feats = graph.x, surface.x

        # HMR introduced this weird self filtering
        h_geom = self.geom_mlp(geom_feats)
        nbr_filter, nbr_core = h_geom.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h_geom = nbr_filter * nbr_core

        h_chem = self.chem_mlp(chem_feats)
        nbr_filter, nbr_core = h_chem.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h_chem = nbr_filter * nbr_core

        # If we additionally use neighboring info, we need to compute it and propagate a message that
        # uses dists and angles
        verts_list = [surf.verts for surf in surface.to_data_list()]
        nodepos_list = [gr.node_pos for gr in graph.to_data_list()]
        with torch.no_grad():
            all_dists = [torch.cdist(vert, nodepos) for vert, nodepos in zip(verts_list, nodepos_list)]
            neighbors_surf = []
            neighbors_graph = []
            offset_surf, offset_graph = 0, 0
            for x in all_dists:
                k = 16
                # these are the 16 closest point in the graph for each vertex
                min_indices_vert = torch.topk(-x, k=k, dim=1).indices
                min_indices_graph = torch.topk(-x.T, k=k, dim=1).indices
                vertex_ids = torch.arange(0, len(x), device=x.device)
                graph_ids = torch.arange(0, len(x.T), device=x.device)
                repeated_tensor_vert = vertex_ids.repeat_interleave(k)
                repeated_tensor_graph = graph_ids.repeat_interleave(k)
                min_indices_vert += offset_graph
                repeated_tensor_vert += offset_surf
                min_indices_graph += offset_surf
                repeated_tensor_graph += offset_graph
                offset_surf += len(x)
                offset_graph += len(x.T)
                neighbors_surf.append(torch.stack((repeated_tensor_vert, min_indices_vert.flatten())))
                neighbors_graph.append(torch.stack((repeated_tensor_graph, min_indices_graph.flatten())))

        ## calculate surf exchange first
        # Slicing requires tuple   

        neighbors_surf = torch.cat([x for x in neighbors_surf], dim=1).long()
        neigh_verts, neigh_graphs = neighbors_surf[0, :], neighbors_surf[1, :]

        # extract relevant chem features
        all_chem_feats = h_chem[neigh_graphs]  # chem_feats
        verts_normals = surface.vnormals
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
        h_geom_mix = self.feat_mlp_chem_geom(torch.cat((h_chem_geom, h_geom), dim=-1))
        del neighbors_surf, neigh_verts, neigh_graphs, all_chem_feats, all_normals, edge_vecs, edge_dists, normed_edge_vecs, nbr_angular, encoded_dists, encoded_angles, expanded_message_features, embedded_messages, nbr_filter, nbr_core, h_chem_geom  # verts_normals,
        torch.cuda.empty_cache()

        ## calculate graph exchang here
        neighbors_graph = torch.cat([x for x in neighbors_graph], dim=1).long()
        neigh_graphs, neigh_verts = neighbors_graph[0, :], neighbors_graph[1, :]
        all_geom_feats = h_geom[neigh_verts]
        # verts_normals = torch.cat([surf.x[:, -3:] for surf in surface.to_data_list()], dim=0)
        all_normals = verts_normals[neigh_verts]
        edge_vecs = surface.verts[neigh_verts] - graph.node_pos[neigh_graphs]
        edge_dists = torch.linalg.norm(edge_vecs, axis=-1)
        normed_edge_vecs = edge_vecs / edge_dists[:, None]
        nbr_angular = torch.einsum('vj,vj->v', normed_edge_vecs, all_normals)
        encoded_dists = _rbf(edge_dists, D_min=0, D_max=8, D_count=self.num_gdf)
        encoded_angles = _rbf(nbr_angular, D_min=-1, D_max=1, D_count=self.num_gdf)
        expanded_message_features = torch.cat([all_geom_feats, encoded_dists, encoded_angles], axis=-1)
        embedded_messages = self.graph_geom_mlp(expanded_message_features)
        nbr_filter, nbr_core = embedded_messages.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h_geom_chem = nbr_filter * nbr_core
        h_geom_chem = scatter(h_geom_chem, neigh_graphs, dim=0, reduce="sum")
        h_chem_mix = self.feat_mlp_geom_chem(torch.cat((h_geom_chem, h_chem), dim=-1))

        del neighbors_graph, neigh_verts, neigh_graphs, all_geom_feats, all_normals, edge_vecs, edge_dists, normed_edge_vecs, nbr_angular, encoded_dists, encoded_angles, expanded_message_features, embedded_messages, nbr_filter, nbr_core, h_geom_chem, verts_normals
        torch.cuda.empty_cache()
        graph.x = h_chem_mix
        surface.x = h_geom_mix
        del h_chem_mix, h_geom_mix
        torch.cuda.empty_cache()
        return surface, graph
