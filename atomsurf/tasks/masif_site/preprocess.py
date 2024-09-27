import os
import sys
import time

import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.utils.data_utils import PreprocessDataset
from atomsurf.utils.python_utils import do_all

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


class PreProcessMSDataset(PreprocessDataset):
    def __init__(self,
                 recompute_s=False,
                 recompute_g=False,
                 data_dir=None,
                 face_reduction_rate=0.5,
                 max_vert_number=100000,
                 use_pymesh=True):
        if data_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')

        super().__init__(data_dir=data_dir, recompute_s=recompute_s, recompute_g=recompute_g,
                         max_vert_number=max_vert_number, face_reduction_rate=face_reduction_rate,
                         use_pymesh=use_pymesh)
        # Set up input/output dirs
        self.pdb_dir = os.path.join(data_dir, '01-benchmark_pdbs')
        self.ply_dir = os.path.join(data_dir, '01-benchmark_surfaces')

        # Set up systems list
        train_list = os.path.join(data_dir, 'train_list.txt')
        test_list = os.path.join(data_dir, 'test_list.txt')
        train_names = set([name.strip() for name in open(train_list, 'r').readlines()])
        test_names = set([name.strip() for name in open(test_list, 'r').readlines()])
        self.all_pdbs = sorted(list(train_names.union(test_names)))

    def __getitem__(self, idx):
        pdb_name = self.all_pdbs[idx]
        ply_path = os.path.join(self.ply_dir, pdb_name + '.ply')
        surface_dump = os.path.join(self.out_surf_dir, f'{pdb_name}.pt')
        try:
            if self.recompute_s or not os.path.exists(surface_dump):
                # Made a version without pymesh to load the initial data
                if self.use_pymesh:
                    import pymesh
                    mesh = pymesh.load_mesh(ply_path)
                    verts = mesh.vertices.astype(np.float32)
                    faces = mesh.faces.astype(np.int32)
                    iface_labels = mesh.get_attribute("vertex_iface").astype(np.int32)
                else:
                    from plyfile import PlyData
                    with open(ply_path, 'rb') as f:
                        plydata = PlyData.read(f)
                        vx = plydata['vertex']['x']
                        vy = plydata['vertex']['y']
                        vz = plydata['vertex']['z']
                        verts = np.stack((vx, vy, vz), axis=1).astype(np.float32)
                        faces = np.stack(plydata['face']['vertex_indices'], axis=0).astype(np.int32)
                        iface_labels = plydata['vertex']['iface'].astype(np.int32)
                iface_labels = torch.from_numpy(iface_labels)
                # If coarsened, we need to adapt the iface_labels on the new verts
                surface = SurfaceObject.from_verts_faces(verts=verts, faces=faces,
                                                         use_pymesh=self.use_pymesh,
                                                         face_reduction_rate=self.face_reduction_rate)
                if len(surface.verts) < len(iface_labels):
                    old_verts = torch.from_numpy(verts)
                    new_verts = torch.from_numpy(surface.verts)
                    with torch.no_grad():
                        k = 4
                        dists = torch.cdist(old_verts, new_verts)  # (old,new)
                        min_indices = torch.topk(-dists, k=k, dim=0).indices  # (k, new)
                        neigh_labels = torch.sum(iface_labels[min_indices], dim=0) > k / 2
                        # old_fraction = iface_labels.sum() / len(iface_labels)
                        # new_fraction = neigh_labels.sum() / len(neigh_labels)
                        iface_labels = neigh_labels.int()
                surface.iface_labels = iface_labels
                surface.add_geom_feats()
                surface.save_torch(surface_dump)
        except Exception as e:
            print('*******failed******', pdb_name, e)
            return 0
        success = self.name_to_graphs(name=pdb_name)
        return success


if __name__ == '__main__':
    pass
    recompute_g = False
    recompute_s = True
    for use_pymesh in (False, True):
        for face_red in [0.1, 0.2, 0.5, 0.9, 1.0]:
            dataset = PreProcessMSDataset(recompute_s=recompute_s, recompute_g=recompute_g,
                                          face_reduction_rate=face_red, use_pymesh=use_pymesh)
            # do_all(dataset, num_workers=0)
            do_all(dataset, num_workers=20)
