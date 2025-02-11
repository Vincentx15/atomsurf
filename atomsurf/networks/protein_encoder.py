import hydra
from torch import nn as nn

from atomsurf.network_utils.communication.surface_graph_comm import SequentialSurfaceGraphCommunication


class ProteinEncoder(nn.Module):
    """
    Just piping protein encoder blocks
    """

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        block_list = []
        if cfg is not None:
            for x in cfg.blocks:
                block = hydra.utils.instantiate(x)
                block_list.append(block)
        self.blocks = nn.ModuleList(block_list)

    def forward(self, surface=None, graph=None):
        for block in self.blocks:
            surface, graph = block(surface, graph)
        return surface, graph

    @classmethod
    def from_blocks_list(cls, block_list):
        encoder = cls()
        encoder.blocks = nn.ModuleList(block_list)
        return encoder


class ProteinEncoderBlock(nn.Module):
    def __init__(self, surface_encoder=None, graph_encoder=None, message_passing=None):
        super().__init__()
        self.surface_encoder = surface_encoder
        self.graph_encoder = graph_encoder
        self.message_passing = message_passing


    def forward(self, surface=None, graph=None):
        if surface is not None and self.surface_encoder !='None':
            surface = self.surface_encoder(surface)

        if graph is not None and self.graph_encoder !='None':
            graph = self.graph_encoder(graph)
        if self.message_passing !='None':
            surface, graph = self.message_passing(surface, graph)

        return surface, graph

class SequentialProteinEncoderBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.surface_encoder = hydra.utils.instantiate(hparams.surface_encoder.instanciate,
                                                       **hparams.surface_encoder.kwargs)
        self.graph_encoder = hydra.utils.instantiate(hparams.graph_encoder.instanciate, **hparams.graph_encoder.kwargs)
        self.message_passing = hydra.utils.instantiate(hparams.communication_block.instanciate,
                                                       **hparams.communication_block.kwargs)

        # check if message_passing is an instance of SequentialSurfaceGraphCommunication
        if not isinstance(self.message_passing, SequentialSurfaceGraphCommunication):
            raise ValueError("message_passing must be an instance of SequentialSurfaceGraphCommunication")

    def forward(self, surface=None, graph=None):
        # We always start with surface starting with graph is possible,
        # by adding a layer before that has an identity surface encoder
        if surface is not None:
            surface = self.surface_encoder(surface)

        surface, graph = self.message_passing(surface, graph, first_pass=True)

        if graph is not None:
            graph = self.graph_encoder(graph)

        surface, graph = self.message_passing(surface, graph, first_pass=False)

        return surface, graph
