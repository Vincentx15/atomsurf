import hydra
import torch.nn as nn
# project
from atomsurf.network_utils.communication.surface_graph_comm import SequentialSurfaceGraphCommunication


class ProteinEncoderBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.surface_encoder = hydra.utils.instantiate(hparams.surface_encoder)
        self.graph_encoder = hydra.utils.instantiate(hparams.graph_encoder)
        self.message_passing = hydra.utils.instantiate(hparams.message_passing)

    def forward(self, surface=None, graph=None):
        if surface is not None:
            surface = self.surface_encoder(surface)

        if graph is not None:
            graph = self.graph_encoder(graph)

        surface, graph = self.message_passing(surface, graph)

        return surface, graph


class SequentialProteinEncoderBlock(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.surface_encoder = hydra.utils.instantiate(hparams.surface_encoder)
        self.graph_encoder = hydra.utils.instantiate(hparams.graph_encoder)
        self.message_passing = hydra.utils.instantiate(hparams.message_passing)

        # check if message_passing is an instance of SequentialSurfaceGraphCommunication
        if not isinstance(self.message_passing, SequentialSurfaceGraphCommunication):
            raise ValueError("message_passing must be an instance of SequentialSurfaceGraphCommunication")

    def forward(self, surface=None, graph=None):
        if self.input_encoder is not None:
            surface, graph = self.input_encoder(surface, graph)

        # we always start with surface
        # starting with graph is possible, by adding a layer before that has an identity surface encoder
        if surface is not None:
            surface = self.surface_encoder(surface)

        surface, graph = self.message_passing(surface, graph, first_pass=True)

        if graph is not None:
            graph = self.graph_encoder(graph)

        surface, graph = self.message_passing(surface, graph, first_pass=False)

        return surface, graph
