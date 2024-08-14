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


class PSRNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head, use_graph_only):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        self.sigma = 2.5
        self.use_graph_only = use_graph_only
        if self.use_graph_only:
            in_features = 128  # 12
            self.top_net = nn.Sequential(*[
                nn.Linear(in_features, in_features//2),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(in_features//2, 1)
            ])
        else:
            in_features = 128 + 129  # 12
            self.top_net = nn.Sequential(*[
                nn.Linear(in_features, in_features//2),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Linear(in_features//2, 1)
            ])

    def project_processed_surface(self, pos, processed, lig_coords):
        # find nearest neighbors between doing last layers
        with torch.no_grad():
            dists = torch.cdist(pos, lig_coords.float())
            min_indices = torch.topk(-dists, k=10, dim=0).indices.unique()
        return processed[min_indices]

    def forward(self, batch):
        # forward pass
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)

        if self.use_graph_only:
            
            sum_result = torch.zeros(graph.batch.max() + 1,graph.x.shape[1]).to(graph.x.device)
            count_result = torch.zeros(graph.batch.max() + 1,graph.x.shape[1]).to(graph.x.device)
            sum_result.scatter_add_(0, graph.batch.unsqueeze(-1).expand_as(graph.x), graph.x)
            count_result.scatter_add_(0, graph.batch.unsqueeze(-1).expand_as(graph.x), torch.ones_like(graph.x))
            # Calculate the mean by dividing sum by count
            mean_result = sum_result / count_result

            mean_result = self.top_net(mean_result)
            return mean_result
        else:
            feats_rbf = torch_rbf(points_1=surface.verts, feats_1=surface.x,
                                   points_2=graph.node_pose, concat=True, sigma=self.sigma)
            x = torch.cat([graph.x,feats_rbf],dim=1)
            sum_result = torch.zeros(graph.batch.max() + 1,graph.x.shape[1]).to(graph.x.device)
            count_result = torch.zeros(graph.batch.max() + 1,graph.x.shape[1]).to(graph.x.device)
            sum_result.scatter_add_(0, graph.batch.unsqueeze(-1).expand_as(graph.x), x)
            count_result.scatter_add_(0, graph.batch.unsqueeze(-1).expand_as(graph.x), torch.ones_like(x))
            # Calculate the mean by dividing sum by count
            mean_result = sum_result / count_result  
            mean_result = self.top_net(mean_result)
            
            return mean_result
