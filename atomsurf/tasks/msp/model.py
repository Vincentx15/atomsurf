import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Batch

from atomsurf.networks.protein_encoder import ProteinEncoder
from atomsurf.network_utils import GCNx2Block


def create_pyg_graph_object(coords, features, sigma=3):
    """
    Define a simple fully connected graph from a set of coords
    :param coords:
    :param features:
    :param sigma:
    :return:
    """
    num_nodes = len(coords)
    device = coords.device

    # Calculate pairwise distances using torch.cdist
    with torch.no_grad():
        pairwise_distances = torch.cdist(coords, coords)
        rbf_weights = torch.exp(-pairwise_distances / sigma)

        # Create edge index using torch.triu_indices and remove self-loops
        row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)
        edge_index = torch.stack([row, col], dim=0)

        # Create bidirectional edges by concatenating (row, col) and (col, row)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)

        # Extract edge weights from pairwise_distances using the created edge_index
        edge_weight = rbf_weights[row, col]
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0).to(device)

    return Data(x=features, edge_index=edge_index, edge_attr=edge_weight)


class MSPNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        self.sigma = 2.5
        in_features = hparams_head.encoded_dims + 1  # add left/right flag
        self.gcn = GCNx2Block(dim_in=in_features, hidden_dims=in_features, dim_out=in_features,
                              dropout=hparams_head.dropout, use_weighted_edge_distance=True)
        self.top_net = nn.Sequential(*[
            nn.Linear(2 * in_features, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])

    def forward(self, batch):
        # First organize in lists to avoid copy pasting
        all_surfaces = batch.surface_lo, batch.surface_ro, batch.surface_lm, batch.surface_rm
        all_graphs = batch.graph_lo, batch.graph_ro, batch.graph_lm, batch.graph_rm
        all_ids_transposed = list(zip(*batch.all_ids))

        # forward pass on each, and extract relevant features and pos
        extracted_pos_feats = list()
        for i, (surfaces, graphs, ids) in enumerate(zip(all_surfaces, all_graphs, all_ids_transposed)):
            surfaces, graphs = self.encoder(graph=graphs, surface=surfaces)

            # update idx, first create a global mapping to directly extract from batch
            ids_lens = [len(i) for i in ids]
            global_ids = [i + ptr for i, ptr in zip(ids, graphs.ptr)]
            global_ids = torch.cat(global_ids, dim=0)
            selected_pos = graphs.node_pos[global_ids]
            selected_feats = graphs.x[global_ids]

            # Add a left/right flag to make the difference between left and right
            flag = (i % 1) * torch.ones(size=(len(selected_feats), 1), device=selected_feats.device)
            selected_feats = torch.cat((selected_feats, flag), dim=-1)

            # Then split those back in a list of extracted pos/vectors for each individual graph
            split_selected_pos = torch.split(selected_pos, ids_lens)
            split_selected_feats = torch.split(selected_feats, ids_lens)
            extracted_pos_feats.append((split_selected_pos, split_selected_feats))

        # Build orig and mut graphs from extracted_pos_feats
        # We need to loop to create individual interface graphs
        # extracted_pos_feats (4 x graphs | pos=[(N1,3),(N2,3...)], feats=[(N1,n_feats+1),(N2,n_feats+1),...])
        orig_graphs, mut_graphs = [], []
        for i in range(len(all_ids_transposed[0])):
            coords_orig = torch.cat((extracted_pos_feats[0][0][i], extracted_pos_feats[1][0][i]), dim=-2)
            coords_mut = torch.cat((extracted_pos_feats[2][0][i], extracted_pos_feats[3][0][i]), dim=-2)
            projected_orig = torch.cat((extracted_pos_feats[0][1][i], extracted_pos_feats[1][1][i]), dim=-2)
            projected_mut = torch.cat((extracted_pos_feats[2][1][i], extracted_pos_feats[3][1][i]), dim=-2)
            orig_graph = create_pyg_graph_object(coords_orig, projected_orig)
            mut_graph = create_pyg_graph_object(coords_mut, projected_mut)
            orig_graphs.append(orig_graph)
            mut_graphs.append(mut_graph)
        orig_graphs = Batch.from_data_list(orig_graphs)
        mut_graphs = Batch.from_data_list(mut_graphs)

        # Finally run GCN, maxpool, and feed to topnet
        orig_graphs = self.gcn(orig_graphs)
        mut_graphs = self.gcn(mut_graphs)
        orig_emb = torch_geometric.nn.global_max_pool(orig_graphs.x, orig_graphs.batch)
        mut_emb = torch_geometric.nn.global_max_pool(mut_graphs.x, mut_graphs.batch)
        x = torch.cat((orig_emb, mut_emb), dim=-1)
        x = self.top_net(x)
        return x
