import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from atomsurf.network_utils.misc_arch.gvp_gnn import GVPConv


def init_block(name, use_normals=True, use_gat=False, use_v2=False, add_self_loops=False,
               fill_value="mean", aggr='add', dim_in=128, dim_out=64):
    if name == "identity":
        return IdentityLayer()
    if name == "linear":
        return torch.nn.Linear(dim_in, dim_out)
    elif name == "no_param_aggregate":
        return NoParamAggregate(aggr=aggr, add_self_loops=add_self_loops, fill_value=fill_value)
    elif name == "cat_post_process":
        return CatPostProcessBlock(dim_in, dim_out)
    elif name == "skip_connection":
        return SkipConnectionBlock()
    elif name == "return_processed":
        return ReturnProcessedBlock()
    elif name == "gvp":
        return GVPWrapper(dim_in, dim_out, use_normals=use_normals)
    elif name == "gcn":
        if not use_gat:
            conv_layer = GCNConv
        else:
            conv_layer = GATv2Conv if use_v2 else GATConv
        edge_dim = 1 if use_v2 else None
        return conv_layer(dim_in, dim_out, add_self_loops=add_self_loops, fill_value=fill_value, edge_dim=edge_dim)


class GVPWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, use_normals=True):
        super().__init__()
        self.use_normals = use_normals

        # Complete initial node features with zero vectors, we could include normals here as initial features
        in_dims = dim_in, 1

        # Edge features: 16 RBF encoding of distance + unit norm direction
        edge_dims = 16, 1

        # We need some vectors to construct interesting representations. Output dims are also used in the network.
        # We later drop the final vectors (we could take the norms)
        out_dims = dim_out, 3
        self.gvp = GVPConv(in_dims, out_dims, edge_dims)

    def forward(self, x, graph):
        if graph.normals is not None and self.use_normals:
            x_v = graph.normals[:, None, :]
        else:
            x_v = torch.zeros((len(x), 1, 3), device=x.device)
        x = (x, x_v)

        e_s = graph.edge_s
        e_v = graph.edge_v

        x_o_s, x_o_v = self.gvp(x=x, edge_index=graph.edge_index, edge_attr=(e_s, e_v))
        return x_o_s


class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        # If only one input is given and no keyword arguments, return the input directly
        if len(args) == 1 and not kwargs:
            return args[0]
        # Return both args and kwargs if either or both are provided
        return args, kwargs


class SkipConnectionBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in, x_out):
        return x_in + x_out


class CatPostProcessBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x_in, x_out):
        return self.lin(torch.cat((x_in, x_out), dim=-1))


class ReturnProcessedBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in, x_out):
        return x_out


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
