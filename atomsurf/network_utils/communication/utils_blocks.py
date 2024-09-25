import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from atomsurf.network_utils.misc_arch.gvp_gnn import GVPConv
from torch_scatter import scatter


def init_block(name, use_normals=True, use_gat=False, use_v2=False, add_self_loops=False, gvp_use_angles=False,
               fill_value="mean", aggr='add', dim_in=128, dim_out=64, n_layers=3, vector_gate=False, num_gdf=16):
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
        return GVPWrapper(dim_in, dim_out, use_normals=use_normals, n_layers=n_layers, gvp_use_angles=gvp_use_angles,
                          vector_gate=vector_gate)
    elif name == "hmr":
        return HMRWrapper(dim_in, dim_out, num_gdf=num_gdf)
    elif name == "gcn":
        return GraphconvWrapper(dim_in, dim_out, use_gat=use_gat, use_v2=use_v2, add_self_loops=add_self_loops,
                                fill_value=fill_value)


class GraphconvWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, use_gat=False, use_v2=False, add_self_loops=False, fill_value="mean"):
        super().__init__()
        if not use_gat:
            conv_layer = GCNConv
        else:
            conv_layer = GATv2Conv if use_v2 else GATConv
        edge_dim = 1 if use_v2 else None
        self.conv = conv_layer(dim_in, dim_out, add_self_loops=add_self_loops, fill_value=fill_value, edge_dim=edge_dim)

    def forward(self, x, bpgraph):
        return self.conv(x, bpgraph.edge_index, bpgraph.edge_weight)


class HMRWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, num_gdf=16, dropout=0.25):
        super().__init__()
        self.embed_mlp = nn.Sequential(
            nn.Linear(dim_in + 2 * num_gdf, dim_out),  # chem_feat_dim
            nn.Dropout(dropout),
            nn.BatchNorm1d(dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, 2 * dim_out),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2 * dim_out),
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x, bpgraph):
        neigh_in, neigh_out = bpgraph.edge_index
        all_feats = x[neigh_in]
        expanded_message_features = torch.cat([all_feats, bpgraph.encoded_dists, bpgraph.encoded_angles], axis=-1)
        embedded_messages = self.embed_mlp(expanded_message_features)
        nbr_filter, nbr_core = embedded_messages.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        h_chem_geom = nbr_filter * nbr_core
        h_chem_geom = scatter(h_chem_geom, neigh_out, dim=0, reduce="sum", dim_size=len(x))
        return h_chem_geom


class GVPWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, n_layers, vector_gate, gvp_use_angles=False, use_normals=True):
        super().__init__()
        self.use_normals = use_normals
        self.gvp_use_angles = gvp_use_angles

        # Complete initial node features with zero vectors, we could include normals here as initial features
        in_dims = dim_in, 1

        # Edge features:
        # - scalars : 16 RBF encoding of distance + optional 16 angles
        # - vectors: unit norm direction
        scalar_dim = 32 if self.gvp_use_angles else 16
        edge_dims = scalar_dim, 1

        # We need some vectors to construct interesting representations. Output dims are also used in the network.
        # We later drop the final vectors (we could take the norms)
        out_dims = dim_out, 3
        self.gvp = GVPConv(in_dims, out_dims, edge_dims, n_layers=n_layers, vector_gate=vector_gate)

    def forward(self, x, bpgraph):
        if bpgraph.normals is not None and self.use_normals:
            x_v = bpgraph.normals[:, None, :]
        else:
            x_v = torch.zeros((len(x), 1, 3), device=x.device)
        x = (x, x_v)

        if self.gvp_use_angles:
            e_s = torch.cat([bpgraph.encoded_dists, bpgraph.encoded_angles], axis=-1)
        else:
            e_s = bpgraph.encoded_dists
        e_v = bpgraph.edge_v[:, None, :]

        x_o_s, x_o_v = self.gvp(x=x, edge_index=bpgraph.edge_index, edge_attr=(e_s, e_v))
        return x_o_s


class IdentityMP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, bpgraph):
        return x


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


class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        # If only one input is given and no keyword arguments, return the input directly
        if len(args) == 1 and not kwargs:
            return args[0]
        # Return both args and kwargs if either or both are provided
        return args, kwargs


class LinearWrapper(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)


class HMR2LayerMLP(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dim_mid),
            nn.SiLU(),
            nn.Linear(dim_mid, dim_out),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dim_out),
        )

    def forward(self, x):
        return self.net(x)


class HMR2LayerMLPChunk(nn.Module):
    def __init__(self, dim_in, hdim, dropout):
        super().__init__()
        self.net = HMR2LayerMLP(dim_in=dim_in, dim_mid=hdim, dim_out=2 * hdim, dropout=dropout)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, x):
        doubled = self.net(x)
        nbr_filter, nbr_core = doubled.chunk(2, dim=-1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus(nbr_core)
        return nbr_filter * nbr_core


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


class CatMergeBlock(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x_in, x_out):
        return self.net(torch.cat((x_in, x_out), dim=-1))


class ReturnProcessedBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_in, x_out):
        return x_out
