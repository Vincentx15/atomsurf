import os
import sys
import time

import igl
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

import atomsurf.utils.torch_utils as diff_utils
from atomsurf.protein.features import Features, FeaturesHolder


def compute_HKS(evecs, evals, num_t, t_min=0.1, t_max=1000, scale=1000):
    evals = evals.flatten()
    assert evals[1] > 0
    assert np.min(evals) > -1E-6
    assert np.array_equal(evals, sorted(evals))

    t_list = np.geomspace(t_min, t_max, num_t, dtype=np.float32)
    phase = np.exp(-np.outer(t_list, evals[1:]))
    wphi = phase[:, None, :] * evecs[None, :, 1:]
    hks = np.einsum('tnk,nk->nt', wphi, evecs[:, 1:]) * scale
    heat_trace = np.sum(phase, axis=1)
    hks /= heat_trace

    return hks


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


class SurfaceObject(Data, FeaturesHolder):
    def __init__(self, features=None, verts=None, faces=None,
                 mass=None, L=None, evals=None, evecs=None, gradX=None, gradY=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.verts = verts
        self.n_verts = len(self.verts) if verts is not None else 0
        self.faces = faces

        self.mass = mass
        self.L = L
        self.evals = evals
        self.evecs = evecs
        self.gradX = gradX
        self.gradY = gradY
        self.k_eig = len(evals) if evals is not None else 0

        if features is None:
            self.features = Features(num_nodes=self.n_verts)
        else:
            self.features = features

    def from_numpy(self, device='cpu'):
        for attr_name in ['verts', 'faces', 'evals', 'evecs']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.safe_to_torch(attr_value).to(device=device))

        for attr_name in ['L', 'mass', 'gradX', 'gradY']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.sparse_np_to_torch(attr_value).to(device=device))
        return self

    def numpy(self):
        for attr_name in ['verts', 'faces', 'evals', 'evecs']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.toNP(attr_value))

        for attr_name in ['L', 'mass', 'gradX', 'gradY']:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.sparse_torch_to_np(attr_value))
        return self

    def save(self, npz_path):
        self.numpy()
        np.savez(npz_path,
                 verts=self.verts,
                 faces=self.faces,
                 mass_data=self.mass.data,
                 mass_indices=self.mass.indices,
                 mass_indptr=self.mass.indptr,
                 mass_shape=self.mass.shape,
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
    def from_verts_faces(cls, verts, faces, use_fem_decomp=False):
        from atomsurf.protein.create_operators import compute_operators
        verts = diff_utils.toNP(verts)
        faces = diff_utils.toNP(faces).astype(int)
        frames, massvec, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, use_fem_decomp=use_fem_decomp)
        surface = cls(verts=verts, faces=faces, mass=massvec, L=L, evals=evals,
                      evecs=evecs, gradX=gradX, gradY=gradY)
        return surface

    @classmethod
    def from_pdb_path(cls, pdb_path, out_ply_path=None, max_vert_number=50000):
        from atomsurf.protein.create_surface import get_surface
        verts, faces = get_surface(pdb_path, out_ply_path=out_ply_path, max_vert_number=max_vert_number)
        return cls.from_verts_faces(verts, faces)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["mass", "L", "gradX", "gradY"]:
            return (0, 1)
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    @staticmethod
    def batch_from_data_list(data_list):
        return SurfaceBatch.batch_from_data_list(data_list=data_list)


class SurfaceBatch(Batch):
    """
    This class is useful for PyG Batching

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def batch_from_data_list(cls, data_list):
        for surface in data_list:
            for key in {'L', 'mass', 'gradX', 'gradY'}:
                tensor_coo = getattr(surface, key)
                tensor_sparse = SparseTensor.from_torch_sparse_coo_tensor(tensor_coo)
                surface[key] = tensor_sparse
        batch = Batch.from_data_list(data_list, follow_batch=('L', 'mass', 'gradX', 'gradY'))
        batch = batch.contiguous()
        surface_batch = cls()
        surface_batch.__dict__.update(batch.__dict__)
        return surface_batch

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["mass", "L", "gradX", "gradY"]:
            return (0, 1)
        else:
            return Data.__cat_dim__(None, key, value, *args, **kwargs)

    def to_lists(self):
        surfaces = self.to_data_list()
        x_in = [mini_surface.x for mini_surface in surfaces]
        mass = [mini_surface.mass for mini_surface in surfaces]
        L = [mini_surface.L for mini_surface in surfaces]
        evals = [mini_surface.evals for mini_surface in surfaces]
        evecs = [mini_surface.evecs for mini_surface in surfaces]
        gradX = [mini_surface.gradX for mini_surface in surfaces]
        gradY = [mini_surface.gradY for mini_surface in surfaces]
        return x_in, mass, L, evals, evecs, gradX, gradY


if __name__ == "__main__":
    pass
    surface_file = "../../data/example_files/example_operator.npz"
    surface = SurfaceObject.load(surface_file)
    surface = surface.from_numpy()
    surface = surface.numpy()
    surface.add_geom_feats()

    # Save as np
    surface_file_np = "../../data/example_files/example_surface.npz"
    surface.save(surface_file_np)
    # Save as torch, a bit heavier
    surface_file_torch = "../../data/example_files/example_surface.pt"
    surface.save_torch(surface_file_torch)
    # t0 = time.time()
    # for _ in range(100):
    #     surface = SurfaceObject.load(surface_file_np)
    # print('np', time.time() - t0)
    #
    # t0 = time.time()
    # for _ in range(100):
    #     surface = torch.load(surface_file_torch)
    # print('torch', time.time() - t0)
    # torch is MUCH faster : 4.34 vs 0.7...

    verts, faces = surface.verts, surface.faces
    surface_hmr = SurfaceObject.from_verts_faces(verts, faces, use_fem_decomp=True)
    a = 1
