import torch
import torch.nn as nn
# project
from .passing_utils import compute_bipartite_graphs
from .utils_blocks import IdentityLayer


class SurfaceGraphCommunication(nn.Module):
    def __init__(self, use_bp,
                 s_pre_block=None, g_pre_block=None,
                 bp_sg_block=None, bp_gs_block=None,
                 s_post_block=None, g_post_block=None,
                 neigh_thresh=8, sigma=2.5, use_gvp=False,
                 use_knn=False, use_hmr=False,
                 **kwargs):
        super().__init__()

        self.use_bp = use_bp
        self.use_gvp = use_gvp
        self.use_hmr = use_hmr

        self.s_pre_block = s_pre_block
        self.g_pre_block = g_pre_block

        self.bp_sg_block = bp_sg_block
        self.bp_gs_block = bp_gs_block

        self.s_post_block = s_post_block
        self.g_post_block = g_post_block

        self.neigh_thresh = neigh_thresh
        self.sigma = sigma
        self.use_knn = use_knn

        # init variables
        self.bp_gs, self.bp_sg = None, None

    def forward(self, surface=None, graph=None):
        if surface is None or graph is None:
            return surface, graph

        # prepare the communication graph
        self.compute_graph(surface, graph)

        # get input features and apply preprocessing
        surface.x = self.s_pre_block(surface.x)
        graph.x = self.g_pre_block(graph.x)

        # == apply the message passing ==
        # concatenate the features for the graph structure
        bp_gs_batch_container = self.bp_gs
        bp_sg_batch_container = self.bp_sg
        bp_sg_batch = bp_sg_batch_container.bp_graph
        bp_gs_batch = bp_gs_batch_container.bp_graph
        x_batch = bp_gs_batch_container.aggregate(surface.x, graph.x)

        # apply the message passing
        if self.use_gvp or self.use_hmr:
            xg_out = self.bp_sg_block(x_batch, bp_sg_batch)
            xs_out = self.bp_gs_block(x_batch, bp_gs_batch)
        else:
            xs_out = self.bp_gs_block(x_batch, bp_gs_batch.edge_index, bp_gs_batch.edge_weight)
            xg_out = self.bp_sg_block(x_batch, bp_sg_batch.edge_index, bp_sg_batch.edge_weight)
        # ====

        # Split back embeddings into surface and graph
        xs_out = bp_sg_batch_container.get_surfs(xs_out)
        xg_out = bp_sg_batch_container.get_graphs(xg_out)

        # apply post-processing
        xs = self.s_post_block(surface.x, xs_out)
        xg = self.g_post_block(graph.x, xg_out)

        # update the features and return
        surface.x = xs
        graph.x = xg
        return surface, graph

    def compute_graph(self, surface, graph):
        if "bp_gs" not in surface or "bp_sg" not in surface:
            self.bp_gs, self.bp_sg = compute_bipartite_graphs(surface, graph,
                                                              neigh_th=self.neigh_thresh,
                                                              use_knn=self.use_knn)
            surface["bp_gs"], surface[
                "bp_sg"] = self.bp_gs, self.bp_sg  # Previously included a clone which I think was unnecessary
        else:
            self.bp_gs, self.bp_sg = surface.bp_gs, surface.bp_sg


class SequentialSurfaceGraphCommunication(SurfaceGraphCommunication):
    def __init__(self, use_bp,
                 s_pre_block=None, g_pre_block=None,
                 bp_sg_block=None, bp_gs_block=None,
                 s_post_block=None, g_post_block=None,
                 neigh_thresh=8, sigma=2.5,
                 **kwargs):
        super().__init__(use_bp=use_bp,
                         s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                         bp_gs_block=bp_gs_block, bp_sg_block=bp_sg_block,
                         s_post_block=s_post_block, g_post_block=g_post_block,
                         neigh_thresh=neigh_thresh, sigma=sigma, **kwargs)

    def forward(self, surface=None, graph=None, first_pass=None):
        assert first_pass is not None, "first_pass must be specified"

        # get the processing blocks
        s_pre_block, g_pre_block = self.s_pre_block, self.g_pre_block

        if first_pass:
            # transfer features from the surface to the graph
            # preprocessing graph features is not necessary
            self.g_pre_block = IdentityLayer()
            surface_new, graph_new = super().forward(surface, graph)
            self.g_pre_block = g_pre_block
            return surface, graph_new
        else:
            # transfer features from the graph to the surface
            # preprocessing surface features is not necessary
            self.s_pre_block = IdentityLayer()
            surface_new, graph_new = super().forward(surface, graph)
            self.s_pre_block = s_pre_block
            return surface_new, graph
