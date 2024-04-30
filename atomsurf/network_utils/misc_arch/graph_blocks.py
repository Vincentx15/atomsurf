# 3p
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GCNx2Block(torch.nn.Module):
    def __init__(self, dim_in, hidden_dims, dim_out, dropout=0.0, use_bn=False, use_weighted_edge_distance=False):
        super().__init__()
        self.conv1 = GCNConv(dim_in, hidden_dims)
        self.conv2 = GCNConv(hidden_dims, dim_out)
        self.use_bn = use_bn
        self.use_weighted_edge_distance = use_weighted_edge_distance
        if use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_dims)
            self.bn2 = nn.BatchNorm1d(dim_out)
        self.dropout = dropout

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        edge_weight = graph.edge_attr if self.use_weighted_edge_distance else None
        x = self.conv1(x, edge_index, edge_weight)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if self.use_bn:
            x = self.bn2(x)
        graph.x = x
        return graph
