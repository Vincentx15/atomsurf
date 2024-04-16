import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

import atomsurf.utils.diffusion_net_utils as diff_utils
import atomsurf.protein.create_operators as operators


class SurfaceObject(Data):
    def __init__(self, features=None, verts=None, faces=None,
                 mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None,
                 chem_feats=None, geom_feats=None, nbr_vids=None, **kwargs):
        super().__init__(**kwargs)

        self.verts = verts
        self.faces = faces
        self.mass = mass
        self.L = L
        self.evals = evals
        self.evecs = evecs
        self.gradX = gradX
        self.gradY = gradY
        self.k_eig = len(evals)

        self.features = features
        self.chem_feats = chem_feats
        self.geom_feats = geom_feats
        self.nbr_vids = nbr_vids

    def from_numpy(self, device='cpu', dtype=torch.float32):
        for attr_name in ['verts', 'faces', 'mass', 'evals', 'evecs']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, torch.from_numpy(attr_value).to(device=device, dtype=dtype))

        for attr_name in ['L', 'gradX', 'gradY']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.sparse_np_to_torch(attr_value).to(device=device, dtype=dtype))
        return self

    def numpy(self, dtype_np=np.float32):
        for attr_name in ['verts', 'faces', 'mass', 'evals', 'evecs']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.toNP(attr_value, dtype_np))

        for attr_name in ['L', 'gradX', 'gradY']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.sparse_torch_to_np(attr_value, dtype_np))
        return self

    def save(self, npz_path):
        self.numpy()
        np.savez(npz_path,
                 verts=self.verts,
                 faces=self.faces,
                 mass=self.mass,
                 L_data=self.L.data,
                 L_indices=self.L.indices,
                 L_indptr=self.L.indptr,
                 L_shape=self.L.shape,
                 evals=self.evals,
                 evecs=self.evecs,
                 gradX_data=self.gradX.data,
                 gradX_indices=self.gradX.indices,
                 gradX_indptr=self.gradX.indptr,
                 gradX_shape=self.gradX.shape,
                 gradY_data=self.gradY.data,
                 gradY_indices=self.gradY.indices,
                 gradY_indptr=self.gradY.indptr,
                 gradY_shape=self.gradY.shape,
                 )

    def save_torch(self, torch_path):
        self.from_numpy()
        torch.save(self, open(torch_path, 'wb'))

    @staticmethod
    def load(npz_path):
        npz_file = np.load(npz_path, allow_pickle=True)
        verts = npz_file['verts']
        faces = npz_file['faces']
        mass, L, evals, evecs, gradX, gradY = operators.load_operators(npz_file)

        return SurfaceObject(verts=verts, faces=faces,
                             mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

    @classmethod
    def batch_from_data_list(cls, data_list):
        # filter out None
        data_list = [data for data in data_list if data is not None]
        if len(data_list) == 0:
            return None
        data_list = [data.from_numpy() for data in data_list]

        # create batch
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = cls()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
            item = batch[key][0]
            if isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            elif torch.is_tensor(item):
                batch[key] = batch[key]
            elif isinstance(item, SparseTensor):
                batch[key] = batch[key]
        return batch.contiguous()


if __name__ == "__main__":
    pass
    surface_file = "../../data/example_files/example_operator.npz"
    surface = SurfaceObject.load(surface_file)
    surface_file_np = "../../data/example_files/example_surface.npz"
    surface.save(surface_file_np)
    surface = SurfaceObject.load(surface_file_np)

    surface_file_torch = "../../data/example_files/example_surface.pt"
    surface.save_torch(surface_file_torch)
    surface = torch.load(surface_file_torch)

    a=1
