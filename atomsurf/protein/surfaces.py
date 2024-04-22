import os
import sys
import time

import igl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

import atomsurf.utils.helpers as diff_utils
from atomsurf.protein.features import Features


def compute_HKS(evecs, evals, num_t, t_min=0.1, t_max=1000, scale=1000):
    evals = evals.flatten()
    assert evals[1] > 0
    assert np.min(evals) > -1E-6
    assert np.array_equal(evals, sorted(evals))

    t_list = np.geomspace(t_min, t_max, num_t)
    phase = np.exp(-np.outer(t_list, evals[1:]))
    wphi = phase[:, None, :] * evecs[None, :, 1:]
    HKS = np.einsum('tnk,nk->nt', wphi, evecs[:, 1:]) * scale
    heat_trace = np.sum(phase, axis=1)
    HKS /= heat_trace

    return HKS


def get_geom_feats(verts, faces, evecs, evals, num_signatures=16):
    # Following masif site
    _, _, k1, k2 = igl.principal_curvature(verts, faces)

    gauss_curvs = k1 * k2
    mean_curvs = 0.5 * (k1 + k2)
    gauss_curvs = gauss_curvs.reshape(-1, 1)
    mean_curvs = mean_curvs.reshape(-1, 1)
    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)
    si = si.reshape(-1, 1)

    # HKS:
    hks = compute_HKS(evecs, evals, num_signatures)
    vnormals = igl.per_vertex_normals(verts, faces)
    geom_feats = np.concatenate([gauss_curvs, mean_curvs, si, hks, vnormals], axis=-1)
    return geom_feats


class SurfaceObject(Data):
    def __init__(self, features=None, verts=None, faces=None,
                 mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None,
                 **kwargs):
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

        if features is None:
            self.features = Features(num_nodes=len(self.verts))
        else:
            self.features = features

    def from_numpy(self, device='cpu', dtype=torch.float32):
        for attr_name in ['verts', 'faces', 'mass', 'evals', 'evecs']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.safe_to_torch(attr_value).to(device=device, dtype=dtype))

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

    @classmethod
    def load(cls, npz_path):
        from atomsurf.protein.create_operators import load_operators
        npz_file = np.load(npz_path, allow_pickle=True)
        verts = npz_file['verts']
        faces = npz_file['faces']
        mass, L, evals, evecs, gradX, gradY = load_operators(npz_file)

        return cls(verts=verts, faces=faces,
                   mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

    def add_geom_feats(self):
        self.numpy()
        geom_feats = get_geom_feats(verts=self.verts, faces=self.faces, evecs=self.evecs, evals=self.evals)
        self.features.add_named_features('geom_feats', geom_feats)

    @classmethod
    def from_verts_faces(cls, verts, faces, use_hmr_decomp=False):
        from atomsurf.protein.create_operators import compute_operators
        frames, massvec, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, use_hmr_decomp=use_hmr_decomp)
        surface = cls(verts=verts, faces=faces, mass=massvec, L=L, evals=evals,
                      evecs=evecs, gradX=gradX, gradY=gradY)
        return surface

    @classmethod
    def from_pdb_path(cls, pdb_path, out_ply_path=None):
        from atomsurf.protein.create_surface import get_surface
        verts, faces = get_surface(pdb_path, out_ply_path=out_ply_path)
        return cls.from_verts_faces(verts, faces)

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
    surface.add_geom_feats()

    # Save as np
    surface_file_np = "../../data/example_files/example_surface.npz"
    surface.save(surface_file_np)

    # Save as torch, a bit heavier
    surface_file_torch = "../../data/example_files/example_surface.pt"
    surface.save_torch(surface_file_torch)

    t0 = time.time()
    for _ in range(100):
        surface = SurfaceObject.load(surface_file_np)
    print('np', time.time() - t0)

    t0 = time.time()
    for _ in range(100):
        surface = torch.load(surface_file_torch)
    print('torch', time.time() - t0)
    # torch is MUCH faster : 4.34 vs 0.7...
    a = 1
