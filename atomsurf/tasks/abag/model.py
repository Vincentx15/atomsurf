import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from atomsurf.networks.protein_encoder import ProteinEncoder


def build_block_sparse_mask(list_ab, list_ag):
    """
    Build a block-sparse mask for attention.
    - list1, list2: List of node indices defining valid attention blocks.
    - total_length: Total number of nodes in the sequence (both batches combined).
    """
    device = list_ab.device
    list_ab = [0] + torch.cumsum(list_ab, dim=0).tolist()
    list_ag = [0] + torch.cumsum(list_ag, dim=0).tolist()

    edge_indices = []

    # Iterate over the blocks and set mask to 1 where attention is allowed
    for start1, end1, start2, end2 in zip(list_ab[:-1], list_ab[1:], list_ag[:-1], list_ag[1:]):
        nodes1 = torch.arange(start1, end1, device=device)
        nodes2 = torch.arange(start2, end2, device=device)

        # Create a complete graph between the nodes in this block
        grid1, grid2 = torch.meshgrid(nodes1, nodes2)
        edges = torch.vstack([grid1.flatten(), grid2.flatten()])

        # Append to the list of edge indices
        edge_indices.append(edges)

    # Concatenate all the edge indices from each block
    full_edge_index = torch.cat(edge_indices, dim=1)
    # mask = torch.zeros(sum(list_ab), sum(list_ag), dtype=torch.bool)
    # mask[start1:end1, start2:end2] = 1
    # mask[start2:end2, start1:end1] = 1
    # mask = torch_geometric.utils.sparse.dense_to_sparse(mask)
    return full_edge_index


class AbAgNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        self.sigma = 2.5
        in_features = hparams_head.encoded_dims  # 12

        # Copied from GEP
        self.relu = nn.ReLU()
        self.gat = GATConv(in_features, in_features, dropout=0.5)
        self.gat2 = GATConv(in_features, in_features, dropout=0.5)
        self.agbn2 = nn.BatchNorm1d(in_features * 2)
        self.dropout1 = nn.Dropout(0.5)
        self.agfc = nn.Linear(in_features * 2, 1, 1)
        self.bn2 = nn.BatchNorm1d(in_features * 2)
        self.dropout2 = nn.Dropout(0.15)
        self.fc = nn.Linear(in_features * 2, 1, 1)

    def forward(self, batch):
        # forward pass
        _, graph_ab = self.encoder(graph=batch.graph_ab, surface=batch.surface_ab)
        _, graph_ag = self.encoder(graph=batch.graph_ag, surface=batch.surface_ag)
        device = batch.graph_ab.x.device

        # Extract only cdr prediction on Ab side
        cdrs = batch.cdr
        offset = 0
        global_cdr = []
        for cdr, num_node in zip(cdrs, batch.graph_ab.node_len):
            offset_cdr = torch.tensor(cdr, device=device) + offset
            offset += num_node
            global_cdr.append(offset_cdr)
        global_cdr = torch.cat(global_cdr)
        selected_ab = graph_ab.x[global_cdr]

        # Build graph to run gat on. Just a diagonal of ones
        cdr_length = torch.tensor([len(cdr) for cdr in cdrs], device=device)
        edge_index = build_block_sparse_mask(cdr_length, batch.graph_ag.node_len)

        # Now we have the ab => ag blocks. IE edge_index is of shape (cdr1+cdr2+..., ag1+ag2+...)
        # Let's put it in one graph to be able to use bipartite approaches,
        # like in BPGraph (cdr1+cdr2+...+ ag1+ag2+..., cdr1+cdr2+...+ag1+ag2+...)
        edge_index[1] += len(global_cdr)
        # Then add reverse direction
        full_edge_index = torch.cat((edge_index, edge_index[[1, 0], :]), dim=1)

        # Finally run a gat over it and split it back
        x = torch.cat((selected_ab, batch.graph_ag.x), dim=0)
        x = self.gat(x, full_edge_index)
        x = self.relu(x)
        x = self.gat2(x, full_edge_index)
        x_ab = x[:len(global_cdr), :]
        x_ag = x[len(global_cdr):, :]

        # And apply our two top nets on Ab...
        x_ab = torch.cat((x_ab, selected_ab), dim=1)  # Residual connection for ab
        x_ab = self.bn2(x_ab)
        x_ab = self.relu(x_ab)
        x_ab = self.dropout2(x_ab)
        x_ab = self.fc(x_ab)

        # .. and on Ag
        x_ag = torch.cat((x_ag, batch.graph_ag.x), dim=1)  # Residual connection for ag
        x_ag = self.agbn2(x_ag)
        x_ag = self.relu(x_ag)
        x_ag = self.dropout2(x_ag)
        x_ag = self.agfc(x_ag)
        return x_ab.flatten(), x_ag.flatten()
