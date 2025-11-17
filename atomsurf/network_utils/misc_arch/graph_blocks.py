# 3p
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from .gvp_gnn import GVPConv

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

class GVPBlock(torch.nn.Module):
    def __init__(self,in_dims, out_dims, edge_dims,n_layers=3, module_list=None, aggr="mean",activations=(F.relu, torch.sigmoid), vector_gate=False):
        super().__init__()
        in_dims = in_dims, 1
        out_dims = out_dims, 1
        edge_dims = edge_dims, 1
        self.gvp_conv=GVPConv(in_dims, out_dims, edge_dims,n_layers, module_list=module_list, aggr=aggr,activations=activations, vector_gate=vector_gate)

    def forward(self,graph):
        x, edge_index = graph.x, graph.edge_index
        coord= graph.node_pos.reshape(-1,1,3)
        edge_attr_scalar = graph.edge_attr[:,None]
        edge_attr_vector = graph.node_pos[edge_index[0]]-graph.node_pos[edge_index[1]]
        edge_attr_vector = edge_attr_vector/ torch.linalg.norm(edge_attr_vector,dim=1)[:,None]
        edge_attr_vector = edge_attr_vector[:,None,:]
        input = (x, coord)
        x_o_s, x_o_v = self.gvp_conv(x=input, edge_index=edge_index, edge_attr=(edge_attr_scalar, edge_attr_vector))
        graph.x = x_o_s
        return graph
