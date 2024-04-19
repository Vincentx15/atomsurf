import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


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
    def __init__(self, use_skip, use_cat, cat_block):
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
            edge_index, edge_weights = add_self_loops(edge_index, edge_weights, fill_value=self.fill_value, num_nodes=x.size(0))

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_weights=edge_weights)
        return out

    def message(self, x_j, edge_weights):
        # x_j has shape [E, out_channels]
        return edge_weights[..., None] * x_j
