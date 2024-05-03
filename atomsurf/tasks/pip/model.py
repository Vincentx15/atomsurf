import torch
import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder



def torch_rbf(points_1, points_2, feats_1, sigma=2.5, eps=0.01, concat=False):
    """
    Get the signal on the points1 onto the points 2
    :param points_1: n,3
    :param points_2:m,3
    :param feats_1:n,k
    :param sigma:
    :return: m,k
    """
    if not points_1.dtype == points_2.dtype == feats_1.dtype:
        raise ValueError(
            f"can't RBF with different dtypes. point1 :{points_1.dtype}, "
            f"point2 :{points_2.dtype}, feat1 :{feats_1.dtype}")
    # Get all to all dists, make it a weight and message passing.
    #     TODO : Maybe include sigma as a learnable parameter
    with torch.no_grad():
        all_dists = torch.cdist(points_2, points_1)
        rbf_weights = torch.exp(-all_dists / sigma)
        # TODO : Probably some speedup is possible with sparse
        # rbf_weights_selection = rbf_weights > (np.exp(-2 * sigma))
        # rbf_weights = rbf_weights_selection * rbf_weights

        # Then normalize by line, the sum are a confidence score
        line_norms = torch.sum(rbf_weights, dim=1) + eps
        rbf_weights = torch.div(rbf_weights, line_norms[:, None])
    feats_2 = torch.mm(rbf_weights, feats_1)
    if concat:
        return torch.cat((feats_2, torch.tanh(line_norms[:, None])), dim=1)
    return feats_2, line_norms

class PIPNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head,use_graph_only):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        self.sigma=2.5
        self.use_graph_only=use_graph_only
        if self.use_graph_only:
            in_features=12*2
            self.top_net = nn.Sequential(*[
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(in_features, 1)
            ])
        else:
            in_features = 12*2+13*2
            self.top_net = nn.Sequential(*[
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(in_features, 1)
            ])

    def project_processed_surface(self, pos, processed, lig_coords):
        # find nearest neighbors between doing last layers
        with torch.no_grad():
            dists = torch.cdist(pos, lig_coords.float())
            min_indices = torch.topk(-dists, k=10, dim=0).indices.unique()
        return processed[min_indices]

    def forward(self, batch):
        # forward pass
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        surface_2, graph_2 = self.encoder(graph=batch.graph_2, surface=batch.surface_2)
        if self.use_graph_only:
            dists = torch.cdist(torch.cat(batch.locs_left), graph_1.node_pos)
            min_indices = torch.argmin(dists, dim=1)
            processed_left = graph_1.x[min_indices]
            dists = torch.cdist(torch.cat(batch.locs_right), graph_2.node_pos)
            min_indices = torch.argmin(dists, dim=1)
            processed_right = graph_2.x[min_indices]
            x=torch.cat([processed_left,processed_right],dim=1)
            x=self.top_net(x)
            return x
        else:
            dists = torch.cdist(torch.cat(batch.locs_left), graph_1.node_pos)
            min_indices = torch.argmin(dists, dim=1)
            processed_left = graph_1.x[min_indices]
            dists = torch.cdist(torch.cat(batch.locs_right), graph_2.node_pos)
            min_indices = torch.argmin(dists, dim=1)
            processed_right = graph_2.x[min_indices]
            feats_left = torch_rbf(points_1=surface_1.verts, feats_1=surface_1.x,
                                                 points_2=torch.cat(batch.locs_left), concat=True, sigma=self.sigma)
            feats_right = torch_rbf(points_1=surface_2.verts, feats_1=surface_2.x,
                                                 points_2=torch.cat(batch.locs_right), concat=True, sigma=self.sigma)
            x=self.top_net([processed_left,feats_left,processed_right,feats_right],dim=1)
            return x
