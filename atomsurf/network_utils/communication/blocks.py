# project
from .surface_graph_comm import SurfaceGraphCommunication, SequentialSurfaceGraphCommunication
from .utils_blocks import init_block


class ConcurrentCommunication(SurfaceGraphCommunication):
    def __init__(self,
                 # preprocess blocks
                 pre_s_block="identity", pre_g_block="identity", pre_s_dim_in=128, pre_s_dim_out=64, pre_g_dim_in=128,
                 pre_g_dim_out=64,
                 # message passing blocks
                 use_gat=False, use_v2=False, bp_self_loops=False, bp_fill_value="mean",  # GCN args
                 use_gvp=False, use_normals=True, n_layers=3, vector_gate=False, gvp_use_angles=False,  # GVP args
                 bp_s_dim_in=64, bp_s_dim_out=64, bp_g_dim_in=64, bp_g_dim_out=64,
                 # postprocess blocks
                 post_s_block="identity", post_g_block="identity", post_s_dim_in=128, post_s_dim_out=64,
                 post_g_dim_in=128, post_g_dim_out=64,
                 # misc
                 neigh_thresh=8, sigma=2.5, use_hmr=False, num_gdf=16,
                 **kwargs):

        # preprocess blocks
        s_pre_block = init_block(pre_s_block, dim_in=pre_s_dim_in, dim_out=pre_s_dim_out)
        g_pre_block = init_block(pre_g_block, dim_in=pre_g_dim_in, dim_out=pre_g_dim_out)

        # message passing blocks
        # * this version does not use self-loops, because we will be summing surface-level features with graph-level features (not good apriori)
        if use_gvp:
            bp_sg_block = init_block("gvp",
                                     dim_in=bp_s_dim_in, dim_out=bp_s_dim_out, use_normals=use_normals,
                                     gvp_use_angles=gvp_use_angles, n_layers=n_layers, vector_gate=vector_gate)
            bp_gs_block = init_block("gvp",
                                     dim_in=bp_g_dim_in, dim_out=bp_g_dim_out, use_normals=use_normals,
                                     gvp_use_angles=gvp_use_angles, n_layers=n_layers, vector_gate=vector_gate)
        elif use_hmr:
            bp_sg_block = init_block("hmr",
                                     dim_in=bp_s_dim_in, dim_out=bp_s_dim_out, num_gdf=num_gdf)
            bp_gs_block = init_block("hmr",
                                     dim_in=bp_g_dim_in, dim_out=bp_g_dim_out, num_gdf=num_gdf)
        else:
            bp_sg_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=bp_s_dim_in, dim_out=bp_s_dim_out,
                                     add_self_loops=bp_self_loops, fill_value=bp_fill_value)
            bp_gs_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=bp_g_dim_in, dim_out=bp_g_dim_out,
                                     add_self_loops=bp_self_loops, fill_value=bp_fill_value)
        # post-process blocks
        # * skip connection is a bad design, summing surface-level features with graph-level features, the skip is done in two different spaces
        # * we will use concatenation instead
        s_post_block = init_block(post_s_block, dim_in=post_s_dim_in, dim_out=post_s_dim_out)
        g_post_block = init_block(post_g_block, dim_in=post_g_dim_in, dim_out=post_g_dim_out)

        super().__init__(bp_sg_block=bp_sg_block, bp_gs_block=bp_gs_block,
                         s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                         s_post_block=s_post_block, g_post_block=g_post_block,
                         neigh_thresh=neigh_thresh, sigma=sigma, **kwargs)


class SequentialCommunication(SequentialSurfaceGraphCommunication):
    def __init__(self, use_bp=True,
                 # preprocess blocks
                 pre_s_block="identity", pre_g_block="identity", pre_s_dim_in=128, pre_s_dim_out=64, pre_g_dim_in=128,
                 pre_g_dim_out=64,
                 # message passing blocks
                 use_gat=False, use_v2=False, bp_self_loops=False, bp_fill_value="mean",
                 bp_s_dim_in=64, bp_s_dim_out=64, bp_g_dim_in=64, bp_g_dim_out=64,
                 # postprocess blocks
                 post_s_block="identity", post_g_block="identity", post_s_dim_in=128, post_s_dim_out=64,
                 post_g_dim_in=128, post_g_dim_out=64,
                 # misc
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = init_block(pre_s_block, dim_in=pre_s_dim_in, dim_out=pre_s_dim_out)
        g_pre_block = init_block(pre_g_block, dim_in=pre_g_dim_in, dim_out=pre_g_dim_out)

        # message passing blocks
        # * this version does not use self-loops, because we will be summing surface-level features with graph-level features (not good apriori)

        if use_bp:
            bp_sg_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=bp_s_dim_in, dim_out=bp_s_dim_out,
                                     add_self_loops=bp_self_loops, fill_value=bp_fill_value)
            bp_gs_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=bp_g_dim_in, dim_out=bp_g_dim_out,
                                     add_self_loops=bp_self_loops, fill_value=bp_fill_value)
        else:
            bp_sg_block, bp_gs_block = None, None

        # post-process blocks
        # * skip connection is a bad design, summing surface-level features with graph-level features, the skip is done in two different spaces
        # * we will use concatenation instead
        s_post_block = init_block(post_s_block, dim_in=post_s_dim_in, dim_out=post_s_dim_out)
        g_post_block = init_block(post_g_block, dim_in=post_g_dim_in, dim_out=post_g_dim_out)

        super().__init__(use_bp, bp_sg_block=bp_sg_block, bp_gs_block=bp_gs_block,
                         s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                         s_post_block=s_post_block, g_post_block=g_post_block,
                         neigh_thresh=neigh_thresh, sigma=sigma, **kwargs)


# Backward compatibility - old blocks
class ParallelCommunicationV1(SurfaceGraphCommunication):
    def __init__(self, use_bp=True,
                 aggr_type='add', add_self_loops=False, fill_value="mean",
                 post_s_dim_in=128, post_s_dim_out=64,
                 post_g_dim_in=128, post_g_dim_out=64,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = init_block("identity")
        g_pre_block = init_block("identity")

        # message passing blocks
        if use_bp:
            bp_gs_block = init_block("no_param_aggregate", aggr=aggr_type, self_loops=add_self_loops,
                                     fill_value=fill_value)
            bp_sg_block = init_block("no_param_aggregate", aggr=aggr_type, self_loops=add_self_loops,
                                     fill_value=fill_value)
        else:
            bp_gs_block, bp_sg_block = None, None

        # post-process blocks
        concatenate_block = init_block("cat_post_process", dim_in=post_s_dim_in, dim_out=post_s_dim_out)
        s_post_block = concatenate_block
        g_post_block = concatenate_block

        super().__init__(use_bp, bp_sg_block=bp_sg_block, bp_gs_block=bp_gs_block,
                         s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                         s_post_block=s_post_block, g_post_block=g_post_block,
                         neigh_thresh=neigh_thresh, sigma=sigma, **kwargs)


class SequentialCommunicationV1(SequentialSurfaceGraphCommunication):
    def __init__(self, use_bp=True,
                 use_gat=False, use_v2=False, bp_self_loops=True, bp_fill_value="mean",
                 bp_s_dim_in=64, bp_s_dim_out=64, bp_g_dim_in=64, bp_g_dim_out=64,
                 post_use_skip=False,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = init_block("identity")
        g_pre_block = init_block("identity")

        # message passing blocks
        if use_bp:
            bp_gs_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=bp_g_dim_in, dim_out=bp_g_dim_out,
                                     add_self_loops=bp_self_loops, fill_value=bp_fill_value)
            bp_sg_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=bp_s_dim_in, dim_out=bp_s_dim_out,
                                     add_self_loops=bp_self_loops, fill_value=bp_fill_value)
        else:
            bp_gs_block, bp_sg_block = None, None

        # post-process blocks
        s_post_block = init_block("skip_connection") if post_use_skip else init_block("return_processed")
        g_post_block = init_block("skip_connection") if post_use_skip else init_block("return_processed")

        super().__init__(use_bp, bp_sg_block=bp_sg_block, bp_gs_block=bp_gs_block,
                         s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                         s_post_block=s_post_block, g_post_block=g_post_block,
                         neigh_thresh=neigh_thresh, sigma=sigma, **kwargs)


class GATCommunicationV1(SurfaceGraphCommunication):
    def __init__(self, use_bp=True,
                 use_gat=False, use_v2=False, bp_self_loops=True, bp_fill_value="mean",
                 bp_s_dim_in=64, bp_s_dim_out=64, bp_g_dim_in=64, bp_g_dim_out=64,
                 post_use_skip=False,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):

        # preprocess blocks
        s_pre_block = init_block("identity")
        g_pre_block = init_block("identity")

        # message passing blocks
        if use_bp:
            bp_gs_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=bp_g_dim_in, dim_out=bp_g_dim_out,
                                     add_self_loops=bp_self_loops, fill_value=bp_fill_value)
            bp_sg_block = init_block("gcn",
                                     use_gat=use_gat, use_v2=use_v2,
                                     dim_in=bp_s_dim_in, dim_out=bp_s_dim_out,
                                     add_self_loops=bp_self_loops, fill_value=bp_fill_value)
        else:
            bp_gs_block, bp_sg_block = None, None

        # post-process blocks
        s_post_block = init_block("skip_connection") if post_use_skip else init_block("return_processed")
        g_post_block = init_block("skip_connection") if post_use_skip else init_block("return_processed")

        super().__init__(use_bp, bp_sg_block=bp_sg_block, bp_gs_block=bp_gs_block,
                         s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                         s_post_block=s_post_block, g_post_block=g_post_block,
                         neigh_thresh=neigh_thresh, sigma=sigma, **kwargs)
