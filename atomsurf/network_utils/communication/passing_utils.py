import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops


# todo check if this valid (copy pasted)
class NoParamAggregate(MessagePassing):
    # this class perform aggregation without any parameters, using only edge weights
    def __init__(self, aggr='add', add_self_loops=True, fill_value="mean"):
        super().__init__(aggr=aggr)
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value

    def forward(self, x, edge_index, edge_weights):
        # todo self loop added here, was not supported in the original code
        if self.add_self_loops:
            edge_index, edge_weights = add_self_loops(edge_index, edge_weights, fill_value=self.fill_value,
                                                      num_nodes=x.size(0))

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_weights=edge_weights)
        return out

    def message(self, x_j, edge_weights):
        # x_j has shape [E, out_channels]
        return edge_weights[..., None] * x_j


def compute_rbf_graph(surface, graph, sigma):
    vertices = surface.vertices
    with torch.no_grad():
        all_dists = [torch.cdist(vert, mini_graph.pos) for vert, mini_graph in zip(vertices, graph.to_data_list())]
        rbf_weights = [torch.exp(-x / sigma) for x in all_dists]
    return rbf_weights


# FROM GVP
def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


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


def get_gvp_feats(pos, edge_index, neigh_th=8):
    E_vectors = pos[edge_index[0]] - pos[edge_index[1]]
    dists = E_vectors.norm(dim=-1)
    rbf = _rbf(dists, D_max=neigh_th)
    u_vec = _normalize(E_vectors)[:, None, :]
    weights = torch.exp(-dists / (neigh_th / 2))
    return weights, rbf, u_vec


def get_gvp_graph(pos, edge_index, normals=None, neigh_th=8):
    # Keep dists for compatibility with mixed GVP/bipartite approaches
    weights, rbf, u_vec = get_gvp_feats(pos, edge_index, neigh_th=neigh_th)
    return Data(all_pos=pos, edge_index=edge_index, edge_s=rbf, edge_v=u_vec, edge_weight=weights, normals=normals)


class BPGraphBatch:
    def __init__(self, graph_list, surface_sizes, graph_sizes):
        batch = Batch.from_data_list(graph_list)
        self.surface_sizes = surface_sizes
        self.graph_sizes = graph_sizes

        # To go from concatenated format to split one
        extracter = torch.cat([torch.cat((torch.ones(surface_size), torch.zeros(graph_size)))
                               for surface_size, graph_size in zip(surface_sizes, graph_sizes)])
        self.surf_extracter = extracter == 1
        self.graph_extracter = extracter == 0
        self.batch = batch  # TODO not sure how to subclass the batch here

    def get_surfs(self, x):
        return x[self.surf_extracter]

    def get_graphs(self, x):
        return x[self.graph_extracter]

    def aggregate(self, surf_x, graph_x):
        split_surfs = torch.split(surf_x, self.surface_sizes)
        split_graphs = torch.split(graph_x, self.graph_sizes)
        alternated = [elt for surfgraph in zip(split_surfs, split_graphs) for elt in surfgraph]
        return torch.cat(alternated)


def compute_bipartite_graphs(surfaces, graphs, neigh_th=8, k=16, use_knn=False, gvp_feats=False):
    """
    Code to compute the graph tying surface vertices to graph nodes
    :param surfaces: A batched Surface object
    :param graphs: A batched graph object
    :param neigh_th: the distance threshold to use
    :param k: the number of neighbors to use
    :param use_knn: Whether to use knn or distance threshold
    :param gvp_feats: Whether to add gvp features or not
    :return:
    """

    surface_sizes = list(surfaces.n_verts.detach().cpu().numpy())
    graph_sizes = list(graphs.node_len.detach().cpu().numpy())
    verts_list = torch.split(surfaces.verts, surface_sizes)
    # TODO: UNSAFE, if normals are useful, find a better way (probably by including it as a surface attribute)
    vertsnormals_list = torch.split(surfaces.x[:, -3:], surface_sizes)
    nodepos_list = torch.split(graphs.node_pos, graph_sizes)
    nodenormals_list = [torch.zeros_like(graph_node_pos) for graph_node_pos in nodepos_list]
    sigma = neigh_th / 2
    with torch.no_grad():
        all_dists = [torch.cdist(vert, nodepos) for vert, nodepos in zip(verts_list, nodepos_list)]
        if use_knn:
            neighbors = []
            for x in all_dists:
                min_indices = torch.topk(-x, k=k, dim=1).indices
                vertex_ids = torch.arange(0, len(x), device=x.device)
                repeated_tensor = vertex_ids.repeat_interleave(k)
                neighbors.append((repeated_tensor, min_indices.flatten()))  # TODO use a better rneighbor
        else:
            neighbors = [torch.where(x < neigh_th) for x in all_dists]

        # Slicing requires tuple
        neighbors = [torch.stack(x).long() for x in neighbors]

        # Plot the number of neighbors per vertex
        # neigh_counts = neighbors[0][1].unique(return_counts=True)[1]
        # import matplotlib.pyplot as plt
        # plt.hist(neigh_counts.cpu())
        # plt.show()

        # This is the torch_cluster radius version, which does not return distances
        # Unbatched version is actually close/slower.
        # Batched version is close on CPU and approx 3 times faster on GPU.

        # from torch_cluster import radius
        # neighbors_2 = []
        # for vert, nodepos in zip(verts_list, nodepos_list):
        #     # Counterintuitive, radius finds points in y close to x..
        #     # result is 2,N with pair corresponding to, point in node, point in vert
        #     edge_index = radius(nodepos, vert, neigh_th)
        #     neighbors_2.append(edge_index)

        # # results of batch version is hard to work with, a flat list of indices with batch increments..
        # neighbors_3 = radius(x=graphs.node_pos,
        #                      y=surfaces.verts,
        #                      batch_x=graphs.batch,
        #                      batch_y=surfaces.batch,
        #                      r=neigh_th)

        # Neighbor holds the 16 closest point in the graph for each vertex,
        # it needs to be offset by the #vertex (bipartite)
        for i, neighbor in enumerate(neighbors):
            neighbor[1] += len(verts_list[i])
        reverse_neighbors = [torch.flip(neigh, dims=(0,)) for neigh in neighbors]
        all_pos = [torch.cat((vert, nodepos)) for vert, nodepos in zip(verts_list, nodepos_list)]
        all_normals = [torch.cat((vernorms, nodenorms)) for vernorms, nodenorms in
                       zip(vertsnormals_list, nodenormals_list)]

        if gvp_feats:
            bipartite_surfgraph = [get_gvp_graph(pos=pos, edge_index=neighbor, neigh_th=neigh_th, normals=normals)
                                   for pos, normals, neighbor in zip(all_pos, all_normals, neighbors)]
            bipartite_graphsurf = [get_gvp_graph(pos=pos, edge_index=rneighbor, neigh_th=neigh_th, normals=normals)
                                   for pos, normals, rneighbor in zip(all_pos, all_normals, reverse_neighbors)]
        else:
            dists = [all_dist[neigh[0, :], (neigh[1, :] - all_dist.shape[0])] for all_dist, neigh in
                     zip(all_dists, neighbors)]
            weights = [torch.exp(-x / sigma) for x in dists]
            bipartite_surfgraph = [Data(all_pos=pos, edge_index=neighbor, edge_weight=weight)
                                   for pos, neighbor, weight in zip(all_pos, neighbors, weights)]
            bipartite_graphsurf = [Data(all_pos=pos, edge_index=rneighbor, edge_weight=weight)
                                   for pos, rneighbor, weight in zip(all_pos, reverse_neighbors, weights)]

        # addition
        # we want to go from S1,G1,S2,G2.. to S1,S2,.. G1,...
        # MAYBE CONSTRUCT as S1,S2.. directly ? just have to update neighbors and not use batch.from_data_list...
        # It's not the pyg way, but might be faster..
        # TODO remove pyg batching, current bottleneck:
        # Pre:      0.0007
        # Batch:    0.0013
        bipartite_graphsurf = BPGraphBatch(bipartite_graphsurf, surface_sizes, graph_sizes)
        bipartite_surfgraph = BPGraphBatch(bipartite_surfgraph, surface_sizes, graph_sizes)
        return bipartite_graphsurf, bipartite_surfgraph


# todo check this and verify is behaves similar to the original function
def compute_bipartite_graphs2(surface, graph, neigh_dist_th):
    # this function remove the multiple loops, so it is more efficient
    # make it more efficient by not computing `cdist`
    # todo: there is no self loop in the original code, should we add it? (we don't need it, does not make sense to add graph to surface features)
    vertices = surface.verts
    sigma = neigh_dist_th / 2
    with torch.no_grad():
        graph_list = graph.to_data_list()
        all_dists = [torch.cdist(vert, mini_graph.pos) for vert, mini_graph in zip(vertices, graph_list)]
        neighbors = [torch.nonzero(dist < neigh_dist_th) for dist in all_dists]
        exp_dists = [torch.exp(-dist / sigma) for dist in all_dists]

        combined_positions = [torch.cat((vert, mini_graph.pos)) for vert, mini_graph in zip(vertices, graph_list)]

        # prepare edge indices and weights
        bipartite_graphs = []
        for pos, neighbor_indices, exp_dist in zip(combined_positions, neighbors, exp_dists):
            # adjust indices for the second part of the graph
            adjusted_neighbor_indices = neighbor_indices.clone()
            adjusted_neighbor_indices[:, 1] += len(vertices)

            # reverse indices for the reverse graph
            reverse_neighbor_indices = torch.flip(adjusted_neighbor_indices, dims=[1])

            edge_weights = exp_dist[neighbor_indices[:, 0], neighbor_indices[:, 1] - len(vertices)]

            bipartite_graphs.append(Data(all_pos=pos, edge_index=neighbor_indices.t(), edge_weight=edge_weights))
            bipartite_graphs.append(
                Data(all_pos=pos, edge_index=reverse_neighbor_indices.t(), edge_weight=edge_weights))

        return bipartite_graphs
