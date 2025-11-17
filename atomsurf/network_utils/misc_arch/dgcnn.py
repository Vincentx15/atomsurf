# 3p
import torch
import torch.nn as nn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx_unchanged=None):
    device = x.device
    batch_size, num_dims, num_points = x.size()

    if idx_unchanged is None:
        idx_unchanged = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_unchanged + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  ->
    # (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx_unchanged  # (batch_size, 2*num_dims, num_points, k)


class LayerNormWrapper(nn.Module):
    def __init__(self, dim, two_d=True):
        super(LayerNormWrapper, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.two_d = two_d

    def forward(self, x):
        if self.two_d:
            x = x.permute(0, 3, 2, 1)
            x = self.ln(x)
            x = x.permute(0, 3, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.ln(x)
            x = x.permute(0, 2, 1)
        return x


def get_batch_norm_layers(emb_dims, bn=True, use_in=False, use_gn=False, use_ln=False):
    if not bn:
        return nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity()
    else:
        if use_in:
            return (nn.InstanceNorm2d(64), nn.InstanceNorm2d(64), nn.InstanceNorm2d(64), nn.InstanceNorm2d(64),
                    nn.InstanceNorm2d(64), nn.InstanceNorm1d(emb_dims), nn.InstanceNorm1d(512), nn.InstanceNorm1d(256))
        elif use_gn:
            nb_gb = 8 if emb_dims <= 64 else 32
            return (nn.GroupNorm(8, 64), nn.GroupNorm(8, 64), nn.GroupNorm(8, 64), nn.GroupNorm(8, 64),
                    nn.GroupNorm(8, 64), nn.GroupNorm(nb_gb, emb_dims), nn.GroupNorm(32, 512), nn.GroupNorm(32, 256))
        elif use_ln:
            return (LayerNormWrapper(64), LayerNormWrapper(64), LayerNormWrapper(64), LayerNormWrapper(64),
                    LayerNormWrapper(64), LayerNormWrapper(emb_dims, two_d=False), LayerNormWrapper(512, two_d=False), LayerNormWrapper(256, two_d=False))
        else:
            return (nn.BatchNorm2d(64), nn.BatchNorm2d(64), nn.BatchNorm2d(64), nn.BatchNorm2d(64),
                    nn.BatchNorm2d(64), nn.BatchNorm1d(emb_dims), nn.BatchNorm1d(512), nn.BatchNorm1d(256))


class DGCNN(nn.Module):
    def __init__(self, dim_in=3, dim_out=128, k=20, emb_dims=1024, p_dropout=0.5, fixed_graph=False, bn=True, use_in=False, use_gn=False, use_ln=False):
        super().__init__()
        self.k = k
        self.fixed_graph = fixed_graph

        self.bn1, self.bn2, self.bn3, self.bn4, self.bn5, self.bn6, self.bn7, self.bn8 = get_batch_norm_layers(emb_dims, bn, use_in, use_gn, use_ln)

        self.conv1 = nn.Sequential(nn.Conv2d(dim_in * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=p_dropout)
        self.conv9 = nn.Conv1d(256, dim_out, kernel_size=1, bias=False)

    def forward(self, surface):
        # input data
        x = surface.x
        x = x.transpose(2, 1).contiguous()

        num_points = x.size(2)

        x, true_idx = get_graph_feature(x, k=self.k)      # (batch_size, dim_in, num_points) -> (batch_size, dim_in*2, num_points, k)
        x = self.conv1(x)                                 # (batch_size, dim_in*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                                 # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]              # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x, _ = get_graph_feature(x1, k=self.k, idx_unchanged=(true_idx if self.fixed_graph else None))  # (batch_size, 64, num_points)
        #                                                                                               # -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x, _ = get_graph_feature(x2, k=self.k, idx_unchanged=(true_idx if self.fixed_graph else None))     # (batch_size, 64, num_points)
        #                                                                                                  # -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, emb_dims, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, emb_dims + 64*3, num_points)

        x = self.conv7(x)                       # (batch_size, emb_dims + 64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, dim_out, num_points)

        # output data
        x = x.transpose(2, 1).contiguous()
        surface.x = x
        return surface


class DGCNNLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k=20, fixed_graph=False, emb_dims=1024, p_dropout=0.5):
        super().__init__()
        super().__init__()
        self.k = k
        self.fixed_graph = fixed_graph

    def forward(self, surface):
        # input data
        x = surface.x
        x = x.transpose(2, 1).contiguous()

        true_idx = surface.true_idx if hasattr(surface, 'true_idx') else None
        idx_unchanged = true_idx if self.fixed_graph else None

        x, true_idx = get_graph_feature(x, k=self.k, idx_unchanged=idx_unchanged)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]

        # output data
        x = x.transpose(2, 1).contiguous()
        surface.x = x
        surface.true_idx = true_idx
        return surface
