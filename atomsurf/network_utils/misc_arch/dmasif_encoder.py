import sys
import torch
import torch.nn as nn
from .dmasif_utils.benchmark_models  import dMaSIFConv_seg
from .dmasif_utils.geometry_processing import (
        curvatures,
)
from easydict import EasyDict
import yaml

class dMasifWrapper(nn.Module):
    def __init__(self, dim_in, dim_out,argdir):
        super().__init__()
        with open(argdir, 'r') as f: #"/work/lpdi/users/ymiao/code/MFE/configs/config.yml"
            dmasifcfg = EasyDict(yaml.safe_load(f))
        self.args = dmasifcfg.model.dmasif
        self.curvature_scales=self.args.curvature_scales
        self.orientation_scores=nn.Sequential(
            nn.Linear(dim_in+2*(len(self.curvature_scales)), dim_out),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim_out, 1),
        )
        self.conv = dMaSIFConv_seg(self.args,
            in_channels=dim_in+2*(len(self.curvature_scales)),
            out_channels=dim_out,
            n_layers=self.args.n_layers,
            radius=self.args.radius,
        )
    def features(self, surface):
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""
        # Estimate the curvatures using the triangles or the estimated normals:
        P_curvatures = curvatures(
            surface.verts,
            triangles= None,
            normals= surface.vnormals,
            scales=self.curvature_scales,
            batch=surface.batch,
        )
        surface.x = torch.cat([surface.x,P_curvatures],dim=1)
        return surface

    def forward(self, surface):
        surface= self.features(surface)
        self.conv.load_mesh(xyz=surface.verts,normals=surface.vnormals,weights=self.orientation_scores(surface.x),batch=surface.batch)
        surface.x = self.conv(surface.x)
        return surface