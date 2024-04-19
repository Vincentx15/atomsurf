import torch.nn as nn


class ProteinEncoder(nn.Module):
    def __init__(self, input_encoder, surface_encoder, graph_encoder, message_passing):
        super().__init__()

        self.input_encoder = input_encoder
        self.surface_encoder = surface_encoder
        self.graph_encoder = graph_encoder
        self.message_passing = message_passing

    def forward(self, surface=None, graph=None):
        if self.input_encoder is not None:
            surface, graph = self.input_encoder(surface, graph)

        if surface is not None:
            surface = self.surface_encoder(surface)

        if graph is not None:
            graph = self.graph_encoder(graph)

        surface, graph = self.message_passing(surface, graph)

        return surface, graph
