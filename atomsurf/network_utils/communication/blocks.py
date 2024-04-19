# 3p
# import torch
# import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
# project
from .surface_graph_comm import SurfaceGraphCommunication
from .utils_blocks import NoParamAggregate, IdentityLayer, CatPostProcessBlock, SkipConnectionBlock, ReturnProcessedBlock


class ParallelCommunicationV1(SurfaceGraphCommunication):
    def __init__(self, use_bp=True,
                 aggr_type='add', add_self_loops=False, fill_value="mean",
                 dim_in_s=128, dim_out_s=64,
                 dim_in_g=128, dim_out_g=64,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = IdentityLayer()
        g_pre_block = IdentityLayer()

        # message passing blocks
        if use_bp:
            bp_gs_block = NoParamAggregate(aggr=aggr_type, add_self_loops=add_self_loops, fill_value=fill_value)
            bp_sg_block = NoParamAggregate(aggr=aggr_type, add_self_loops=add_self_loops, fill_value=fill_value)
        else:
            bp_gs_block, bp_sg_block = None, None

        # post-process blocks
        concatenate_block = CatPostProcessBlock(dim_in=dim_in_s, dim_out=dim_out_s)
        s_post_block = concatenate_block
        g_post_block = concatenate_block

        super().__init__(use_bp, bp_gs_block, bp_sg_block, s_pre_block, g_pre_block, s_post_block, g_post_block,
                         neigh_thresh, sigma, **kwargs)


class ParallelCommunicationV2(SurfaceGraphCommunication):
    def __init__(self, use_bp=True,
                 aggr_type='add', add_self_loops=False, fill_value="mean",
                 dim_in_s=128, dim_out_s=64,
                 dim_in_g=128, dim_out_g=64,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = IdentityLayer()
        g_pre_block = IdentityLayer()

        # message passing blocks
        # * self_loops a priori are not beneficial here, because we will be summing surface-level features with graph-level features
        if use_bp:
            bp_gs_block = NoParamAggregate(aggr=aggr_type, add_self_loops=add_self_loops, fill_value=fill_value)
            bp_sg_block = NoParamAggregate(aggr=aggr_type, add_self_loops=add_self_loops, fill_value=fill_value)
        else:
            bp_gs_block, bp_sg_block = None, None

        # post-process blocks
        # * this is different from ParallelCommunicationV1
        # * in V1, the same linear layer is used for both s and g
        s_post_block = CatPostProcessBlock(dim_in=dim_in_s, dim_out=dim_out_s)
        g_post_block = CatPostProcessBlock(dim_in=dim_in_g, dim_out=dim_out_g)

        super().__init__(use_bp, bp_gs_block, bp_sg_block, s_pre_block, g_pre_block, s_post_block, g_post_block,
                         neigh_thresh, sigma, **kwargs)


class SequentialCommunicationV1(SurfaceGraphCommunication):
    def __init__(self, use_bp=True, surface_to_graph=True,
                 use_gat=False, bp_self_loops=True, bp_fill_value="mean",
                 bp_s_dim_in=64, bp_s_dim_out=64, bp_g_dim_in=64, bp_g_dim_out=64,
                 post_use_skip=False,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = IdentityLayer()
        g_pre_block = IdentityLayer()

        # message passing blocks
        if surface_to_graph:
            if use_bp:
                conv_layer = GCNConv if not use_gat else GATConv
                bp_sg_block = conv_layer(bp_s_dim_in, bp_s_dim_out, add_self_loops=bp_self_loops, fill_value=bp_fill_value)
            else:
                bp_sg_block = None
            bp_gs_block = IdentityLayer()
        else:
            if use_bp:
                conv_layer = GCNConv if not use_gat else GATConv
                bp_gs_block = conv_layer(bp_g_dim_in, bp_g_dim_out, add_self_loops=bp_self_loops, fill_value=bp_fill_value)
            else:
                bp_gs_block = None
            bp_sg_block = IdentityLayer()

        # post-process blocks
        if surface_to_graph:
            s_post_block = IdentityLayer()
            g_post_block = SkipConnectionBlock() if post_use_skip else ReturnProcessedBlock()
        else:
            s_post_block = SkipConnectionBlock() if post_use_skip else ReturnProcessedBlock()
            g_post_block = IdentityLayer()

        super().__init__(use_bp, bp_gs_block, bp_sg_block, s_pre_block, g_pre_block, s_post_block, g_post_block,
                         neigh_thresh, sigma, **kwargs)


class SequentialCommunicationV2(SurfaceGraphCommunication):
    def __init__(self, use_bp=True, surface_to_graph=True,
                 use_gat=False, bp_self_loops=False, bp_fill_value="mean",
                 bp_s_dim_in=64, bp_s_dim_out=64, bp_g_dim_in=64, bp_g_dim_out=64,
                 dim_in_s=128, dim_out_s=64,
                 dim_in_g=128, dim_out_g=64,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = IdentityLayer()
        g_pre_block = IdentityLayer()

        # message passing blocks
        # * this version does not use self-loops, because we will be summing surface-level features with graph-level features (not good apriori)
        if surface_to_graph:
            if use_bp:
                conv_layer = GCNConv if not use_gat else GATConv
                bp_sg_block = conv_layer(bp_s_dim_in, bp_s_dim_out, add_self_loops=bp_self_loops, fill_value=bp_fill_value)
            else:
                bp_sg_block = None
            bp_gs_block = IdentityLayer()
        else:
            if use_bp:
                conv_layer = GCNConv if not use_gat else GATConv
                bp_gs_block = conv_layer(bp_g_dim_in, bp_g_dim_out, add_self_loops=bp_self_loops, fill_value=bp_fill_value)
            else:
                bp_gs_block = None
            bp_sg_block = IdentityLayer()

        # post-process blocks
        # * skip connection is a bad design, summing surface-level features with graph-level features, the skip is done in two different spaces
        # * we will use concatenation instead
        if surface_to_graph:
            s_post_block = IdentityLayer()
            g_post_block = CatPostProcessBlock(dim_in=dim_in_g, dim_out=dim_out_g)
        else:
            s_post_block = CatPostProcessBlock(dim_in=dim_in_s, dim_out=dim_out_s)
            g_post_block = IdentityLayer()

        super().__init__(use_bp, bp_gs_block, bp_sg_block, s_pre_block, g_pre_block, s_post_block, g_post_block,
                         neigh_thresh, sigma, **kwargs)


class GATCommunicationV1(SurfaceGraphCommunication):
    def __init__(self, use_bp=True,
                 use_gat=False, use_v2=False, bp_self_loops=True, bp_fill_value="mean",
                 bp_s_dim_in=64, bp_s_dim_out=64, bp_g_dim_in=64, bp_g_dim_out=64,
                 post_use_skip=False,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = IdentityLayer()
        g_pre_block = IdentityLayer()

        # message passing blocks
        if use_bp:
            if not use_gat:
                conv_layer = GCNConv
            else:
                conv_layer = GATv2Conv if use_v2 else GATConv
            edge_dim = 1 if use_v2 else None

            bp_gs_block = conv_layer(bp_g_dim_in, bp_g_dim_out, add_self_loops=bp_self_loops, fill_value=bp_fill_value, edge_dim=edge_dim)
            bp_sg_block = conv_layer(bp_s_dim_in, bp_s_dim_out, add_self_loops=bp_self_loops, fill_value=bp_fill_value, edge_dim=edge_dim)
        else:
            bp_gs_block, bp_sg_block = None, None

        # post-process blocks
        s_post_block = SkipConnectionBlock() if post_use_skip else ReturnProcessedBlock()
        g_post_block = SkipConnectionBlock() if post_use_skip else ReturnProcessedBlock()

        super().__init__(use_bp, bp_gs_block, bp_sg_block, s_pre_block, g_pre_block, s_post_block, g_post_block,
                         neigh_thresh, sigma)


class GATCommunicationV2(SurfaceGraphCommunication):
    def __init__(self, use_bp=True,
                 use_gat=False, use_v2=False, bp_self_loops=False, bp_fill_value="mean",
                 bp_s_dim_in=64, bp_s_dim_out=64, bp_g_dim_in=64, bp_g_dim_out=64,
                 dim_in_s=128, dim_out_s=64,
                 dim_in_g=128, dim_out_g=64,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = IdentityLayer()
        g_pre_block = IdentityLayer()

        # message passing blocks
        # * this version does not use self-loops, because we will be summing surface-level features with graph-level features (not good apriori)
        if use_bp:
            if not use_gat:
                conv_layer = GCNConv
            else:
                conv_layer = GATv2Conv if use_v2 else GATConv
            edge_dim = 1 if use_v2 else None

            bp_gs_block = conv_layer(bp_g_dim_in, bp_g_dim_out, add_self_loops=bp_self_loops, fill_value=bp_fill_value, edge_dim=edge_dim)
            bp_sg_block = conv_layer(bp_s_dim_in, bp_s_dim_out, add_self_loops=bp_self_loops, fill_value=bp_fill_value, edge_dim=edge_dim)
        else:
            bp_gs_block, bp_sg_block = None, None

        # post-process blocks
        # * skip connection is a bad design, summing surface-level features with graph-level features, the skip is done in two different spaces
        # * we will use concatenation instead
        s_post_block = CatPostProcessBlock(dim_in=dim_in_s, dim_out=dim_out_s)
        g_post_block = CatPostProcessBlock(dim_in=dim_in_g, dim_out=dim_out_g)

        super().__init__(use_bp, bp_gs_block, bp_sg_block, s_pre_block, g_pre_block, s_post_block, g_post_block,
                         neigh_thresh, sigma)
