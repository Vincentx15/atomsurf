import torch.nn as nn


class SurfaceEncoderBlock(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, surface):
        surface = self.encoder(surface)

        return surface


class GraphEncoderBlock(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, graph):
        graph = self.encoder(graph)

        return graph
