import torch.nn as nn


class SurfaceEncoder(nn.Module):
    def __init__(self, encoder, input_encoder):
        super().__init__()

        self.encoder = encoder

    def forward(self, surface):

        surface = self.encoder(surface)

        return surface
