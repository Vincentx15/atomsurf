import time

import torch
from torch_geometric.data import Data, Batch


def _rbf(D, D_min=0., D_max=8., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class BPGraphBatch:
    def __init__(self, surfaces, graphs, neighbors, num_gdf=16, mode='sg', neigh_th=8):
        # Now concatenate our graphs into one big one with surface vertices first and graph nodes then
        self.all_surf_size = len(surfaces.verts)
        all_pos = torch.cat((surfaces.verts, graphs.node_pos))
        verts_normals = surfaces.vnormals
        normals = torch.cat((verts_normals, torch.zeros_like(graphs.node_pos)))

        # If mode = 'sg', neighbors holds edges from surface to graph, and vice-versae.
        # When batching, one needs to take it into account
        if mode == 'sg':
            neigh_verts, neigh_graphs = neighbors[0, :], neighbors[1, :]
            edge_vecs = graphs.node_pos[neigh_graphs] - surfaces.verts[neigh_verts]
        elif mode == 'gs':
            neigh_graphs, neigh_verts = neighbors[0, :], neighbors[1, :]
            edge_vecs = surfaces.verts[neigh_verts] - graphs.node_pos[neigh_graphs]
        else:
            raise ValueError('mode is not gs or sg')
        neigh_graphs += self.all_surf_size

        # Compute edge features
        all_normals = verts_normals[neigh_verts]
        edge_dists = torch.linalg.norm(edge_vecs, axis=-1) + 0.00001
        normed_edge_vecs = edge_vecs / edge_dists[:, None]
        nbr_angular = (normed_edge_vecs * all_normals).sum(dim=-1)
        encoded_dists = _rbf(edge_dists, D_min=0, D_max=8, D_count=num_gdf)
        encoded_angles = _rbf(nbr_angular, D_min=-1, D_max=1, D_count=num_gdf)
        weights = torch.exp(-edge_dists / (neigh_th / 2))

        neighbors = torch.stack((neigh_verts, neigh_graphs)) if mode == 'sg' \
            else torch.stack((neigh_graphs, neigh_verts))
        self.bp_graph = Data(
            all_pos=all_pos,
            normals=normals,
            num_nodes=len(all_pos),
            encoded_dists=encoded_dists,
            encoded_angles=encoded_angles,
            edge_index=neighbors,
            edge_weight=weights,
            edge_v=edge_vecs)

    def get_surfs(self, x):
        return x[:self.all_surf_size]

    def get_graphs(self, x):
        return x[self.all_surf_size:]

    def aggregate(self, surf_x, graph_x):
        return torch.cat((surf_x, graph_x))


def compute_bipartite_graphs(surfaces, graphs, neigh_th=8, k=16, use_knn=False, num_gdf=16):
    """
    Code to compute the graph tying surface vertices to graph nodes
    :param surfaces: A batched Surface object
    :param graphs: A batched graph object
    :param neigh_th: the distance threshold to use
    :param k: the number of neighbors to use
    :param use_knn: Whether to use knn or distance threshold
    :return:
    """

    with torch.no_grad():
        # First, split the node/vertices pos for cdist computations
        surface_sizes = list(surfaces.n_verts.detach().cpu().numpy())
        graph_sizes = list(graphs.node_len.detach().cpu().numpy())
        verts_list = torch.split(surfaces.verts, surface_sizes)
        nodepos_list = torch.split(graphs.node_pos, graph_sizes)
        all_dists = [torch.cdist(vert, nodepos) for vert, nodepos in zip(verts_list, nodepos_list)]

        # First we compute edges from the meshes and from the graphs
        neighbors_v = []
        neighbors_g = []
        offset_surf, offset_graph = 0, 0
        for x in all_dists:
            # Find the closest ids
            min_indices_v = torch.topk(-x, k=k, dim=1).indices
            min_indices_g = torch.topk(-x.T, k=k, dim=1).indices
            vertex_ids = torch.arange(0, len(x), device=x.device)
            graph_ids = torch.arange(0, len(x.T), device=x.device)
            repeated_tensor_vert = vertex_ids.repeat_interleave(k)
            repeated_tensor_graph = graph_ids.repeat_interleave(k)

            # We offset by hand to avoid slow batching.
            min_indices_v += offset_graph
            repeated_tensor_vert += offset_surf
            min_indices_g += offset_surf
            repeated_tensor_graph += offset_graph
            offset_surf += len(x)
            offset_graph += len(x.T)
            neighbors_v.append(torch.stack((repeated_tensor_vert, min_indices_v.flatten())))
            neighbors_g.append(torch.stack((repeated_tensor_graph, min_indices_g.flatten())))
        neighbors_v = torch.cat([x for x in neighbors_v], dim=1).long()
        neighbors_g = torch.cat([x for x in neighbors_g], dim=1).long()

        # From this set of edges between all graphs and all surfs, we create a unique large graph
        # with concatenated vertices and nodes S1,S2,S3...G1,G2,... and edges with features
        bipartite_surfgraph = BPGraphBatch(surfaces, graphs, neighbors_v, mode='sg',
                                           num_gdf=num_gdf, neigh_th=neigh_th)
        bipartite_graphsurf = BPGraphBatch(surfaces, graphs, neighbors_g, mode='gs',
                                           num_gdf=num_gdf, neigh_th=neigh_th)
        return bipartite_graphsurf, bipartite_surfgraph
