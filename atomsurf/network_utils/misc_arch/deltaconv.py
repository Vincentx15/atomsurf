# 3p
import torch
from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
from torch_geometric.nn import global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from deltaconv.models import DeltaNetBase
from deltaconv.nn import MLP


class DeltaConv(torch.nn.Module):
    def __init__(self, dim_in=3, dim_out=128, dropout=0.5, conv_channels=[64, 128, 256], mlp_depth=2,
                 embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1):
        super().__init__()

        self.deltaconv_net = DeltaNetSegmentation(dim_in, dim_out, conv_channels, mlp_depth,
                                                  embedding_size, num_neighbors, grad_regularizer, grad_kernel_width, dropout)

    def forward(self, surface):
        # input data
        x = surface.x
        norm = surface.norm
        pos = surface.verts

        # create a new batch
        batch_list = [Data(x=x[i], pos=pos[i], norm=norm[i]) for i in range(x.size(0))]
        pyg_batch = Batch.from_data_list(batch_list)

        # forward pass
        features = self.deltaconv_net(pyg_batch)

        # reshape features using batch information
        features = to_dense_batch(features, pyg_batch.batch)[0]

        # output data
        surface.y = features
        return surface


class DeltaNetSegmentation(torch.nn.Module):
    def __init__(self, in_channels, num_classes, conv_channels=[64, 128, 256], mlp_depth=2,
                 embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1, dropout=0.5):
        """Segmentation of Point Clouds with DeltaConv.
        The architecture is based on the architecture used by DGCNN (https://dl.acm.org/doi/10.1145/3326362.

        Args:
            in_channels (int): the number of channels provided as input.
            num_classes (int): the number of classes to segment.
            conv_channels (list[int]): the number of output channels of each convolution.
            mlp_depth (int): the depth of the MLPs of each convolution.
            embedding_size (int): the embedding size before the segmentation head is applied.
            num_neighbors (int): the number of neighbors to use in estimating the gradient.
            grad_regularizer (float): the regularizer value used in the least-squares fitting procedure.
                In the paper, this value is referred to as \lambda.
                Larger grad_regularizer gives a smoother, but less accurate gradient.
                Lower grad_regularizer gives a more accurate, but more variable gradient.
                The grad_regularizer value should be >0 (e.g., 1e-4) to prevent exploding values.
            grad_kernel_width (float): the width of the gaussian kernel used to weight the
                least-squares problem to approximate the gradient.
                Larger kernel width means that more points are included, which is a 'smoother' gradient.
                Lower kernel width gives a more accurate, but possibly noisier gradient.
        """
        super().__init__()

        self.deltanet_base = DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width)

        # Global embedding
        self.lin_global = MLP([sum(conv_channels), embedding_size])

        # For ShapeNet segmentation, most authors add an embedding of the category to aid with segmentation.
        self.segmentation_head = Seq(
            MLP([embedding_size + sum(conv_channels), 256]), Dropout(dropout), MLP([256, 256]), Dropout(dropout),
            Linear(256, 128), LeakyReLU(negative_slope=0.2), Linear(128, num_classes))

    def forward(self, data):
        conv_out = self.deltanet_base(data)

        x = torch.cat(conv_out, dim=1)
        x = self.lin_global(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)[batch]

        x = torch.cat([x_max] + conv_out, dim=1)

        return self.segmentation_head(x)
