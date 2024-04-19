import torch.nn as nn


class GraphEncoder(nn.Module):
    def __init__(self, encoder, input_encoder):
        super().__init__()

        self.encoder = encoder

    def forward(self, graph):

        graph = self.encoder(graph)

        return graph
