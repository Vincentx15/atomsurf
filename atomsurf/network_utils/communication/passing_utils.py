import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
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


# todo check if this valid (copy pasted)
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
    return dists, rbf, u_vec


def get_gvp_graph(pos, edge_index, neigh_th=8):
    # Keep dists for compatibility with mixed GVP/bipartite approaches
    dists, rbf, u_vec = get_gvp_feats(pos, edge_index, neigh_th=neigh_th)
    return Data(all_pos=pos, edge_index=edge_index, edge_s=rbf, edge_v=u_vec, edge_weight=dists)


# todo check if this valid (copy pasted)
# TODO benchmark against torch_cluster radius graph
def compute_bipartite_graphs(surfaces, graphs, neigh_th, gvp_feats=False):
    verts_list = [surface.verts for surface in surfaces.to_data_list()]
    nodepos_list = [graph.node_pos for graph in graphs.to_data_list()]
    sigma = neigh_th / 2  # todo is this the right way to do it?
    with torch.no_grad():
        all_dists = [torch.cdist(vert, nodepos) for vert, nodepos in zip(verts_list, nodepos_list)]
        neighbors = [torch.where(x < neigh_th) for x in all_dists]
        # Slicing requires tuple
        dists = [all_dist[neigh] for all_dist, neigh in zip(all_dists, neighbors)]
        dists = [torch.exp(-x / sigma) for x in dists]
        neighbors = [torch.stack(x).long() for x in neighbors]
        for i, neighbor in enumerate(neighbors):
            neighbor[1] += len(verts_list[i])
        reverse_neighbors = [torch.flip(neigh, dims=(0,)) for neigh in neighbors]
        all_pos = [torch.cat((vert, nodepos)) for vert, nodepos in zip(verts_list, nodepos_list)]

        if gvp_feats:
            bipartite_surfgraph = [get_gvp_graph(pos=pos, edge_index=neighbor, neigh_th=neigh_th) for pos, neighbor in
                                   zip(all_pos, neighbors)]
            bipartite_graphsurf = [get_gvp_graph(pos=pos, edge_index=rneighbor, neigh_th=neigh_th) for pos, rneighbor in
                                   zip(all_pos, reverse_neighbors)]
        else:
            bipartite_surfgraph = [Data(all_pos=pos, edge_index=neighbor, edge_weight=dist) for pos, neighbor, dist in
                                   zip(all_pos, neighbors, dists)]
            bipartite_graphsurf = [Data(all_pos=pos, edge_index=rneighbor, edge_weight=dist) for pos, rneighbor, dist in
                                   zip(all_pos, reverse_neighbors, dists)]
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
