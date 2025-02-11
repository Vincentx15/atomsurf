import os
from torch_geometric.data import Data

from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader
from atomsurf.network_utils.communication.utils_blocks import HMR2LayerMLP, CatMergeBlock, GVPWrapper
from atomsurf.network_utils.communication.utils_blocks import CatPostProcessBlock, LinearWrapper
from atomsurf.network_utils.communication.surface_graph_comm import SurfaceGraphCommunication
from atomsurf.network_utils import ProNet
from diffusion_net import DiffusionNetBlock
from atomsurf.networks import ProteinEncoderBlock, ProteinEncoder


class DefaultLoader():
    def __init__(self, graph_dir, surface_dir, embeddings_dir):
        graph_data_dir, graph_data_name = os.path.split(graph_dir)
        surface_data_dir, surface_data_name = os.path.split(surface_dir)
        cfg_surface = Data(
            data_dir=surface_data_dir,
            data_name=surface_data_name,
            use_surfaces=True,
            feat_keys='all',
            gdf_expand=True,
            oh_keys='all')
        cfg_graph = Data(
            data_dir=graph_data_dir,
            data_name=graph_data_name,
            use_esm=True,
            use_graphs=True,
            esm_dir=embeddings_dir,
            feat_keys='all',
            oh_keys=['amino_types'])
        self.surf_loader = SurfaceLoader(cfg_surface)
        self.graph_loader = GraphLoader(cfg_graph)

    def __call__(self, pdb):
        surface = self.surf_loader.load("1ycr")
        graph = self.graph_loader.load("1ycr")
        return surface, graph


def get_default_input(in_dim_surface, in_dim_graph, model_dim=128, dropout=0.1):
    """
    Input goes through pre-encoders, exchange information and are then combined together
    :param in_dim_surface:
    :param in_dim_graph:
    :param model_dim:
    :param dropout:
    :return:
    """
    s_pre_block = HMR2LayerMLP(in_dim_surface, model_dim, model_dim, dropout)
    g_pre_block = HMR2LayerMLP(in_dim_graph, model_dim, model_dim, dropout)
    sg = GVPWrapper(model_dim, model_dim, n_layers=3, vector_gate=False, gvp_use_angles=True, use_normals=True)
    gs = GVPWrapper(model_dim, model_dim, n_layers=3, vector_gate=False, gvp_use_angles=True, use_normals=True)
    s_post_block = CatMergeBlock(net=HMR2LayerMLP(model_dim * 2, model_dim, model_dim, dropout))
    g_post_block = CatMergeBlock(net=HMR2LayerMLP(model_dim * 2, model_dim, model_dim, dropout))
    input_block = SurfaceGraphCommunication(s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                                            bp_sg_block=sg, bp_gs_block=gs,
                                            s_post_block=s_post_block, g_post_block=g_post_block)
    return input_block


def get_middle_block(model_dim=128, dropout=0.1):
    """
    Input goes through a surface and a graph encoder,
    then each feature is compacted, exchanged with a message passing and aggregated back
    :param model_dim:
    :param dropout:
    :return:
    """
    # Encoders
    half_dim = model_dim // 2
    diff_net = DiffusionNetBlock(C_width=model_dim, dropout=dropout, use_bn=False, use_layernorm=True,
                                 init_time=10, init_std=10)
    pronet = ProNet(hidden_channels=model_dim, mid_emb=half_dim, dropout=0.1)

    # Mid-encoder
    s_pre_block = LinearWrapper(model_dim, half_dim)
    g_pre_block = LinearWrapper(model_dim, half_dim)
    sg = GVPWrapper(half_dim, half_dim, n_layers=3, vector_gate=False, gvp_use_angles=True, use_normals=True)
    gs = GVPWrapper(half_dim, half_dim, n_layers=3, vector_gate=False, gvp_use_angles=True, use_normals=True)
    s_post_block = CatPostProcessBlock(model_dim, model_dim)
    g_post_block = CatPostProcessBlock(model_dim, model_dim)
    mp_block = SurfaceGraphCommunication(s_pre_block=s_pre_block, g_pre_block=g_pre_block,
                                         bp_sg_block=sg, bp_gs_block=gs,
                                         s_post_block=s_post_block, g_post_block=g_post_block)
    middle_block = ProteinEncoderBlock(surface_encoder=diff_net, graph_encoder=pronet, message_passing=mp_block)
    return middle_block


def get_default_model(in_dim_surface, in_dim_graph, model_dim=128, dropout=0.1, n_block=4):
    """
    The default atomsurf construct leverages an input encoding block along with four middle blocks
    :param in_dim_surface:
    :param in_dim_graph:
    :param model_dim:
    :param dropout:
    :param n_block:
    :return:
    """
    block_list = [get_default_input(in_dim_surface, in_dim_graph, model_dim=model_dim, dropout=dropout)]
    for _ in range(n_block):
        block_list.append(get_middle_block(model_dim=model_dim, dropout=dropout))
    return ProteinEncoder.from_blocks_list(block_list=block_list)
