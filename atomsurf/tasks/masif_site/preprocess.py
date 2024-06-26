import os
import sys
import time

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.graphs import parse_pdb_path
from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.protein.atom_graph import AtomGraphBuilder
from atomsurf.protein.residue_graph import ResidueGraphBuilder

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


class PreProcessPDBDataset(Dataset):
    def __init__(self,
                 recompute_surfaces=False,
                 recompute_graphs=False,
                 data_dir=None,
                 face_reduction_rate=0.5,
                 use_pymesh=True):
        if data_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            masif_site_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')
        else:
            masif_site_data_dir = data_dir

        self.face_reduction_rate = face_reduction_rate
        self.use_pymesh = use_pymesh

        # Set up input/output dirs
        self.pdb_dir = os.path.join(masif_site_data_dir, '01-benchmark_pdbs')
        self.ply_dir = os.path.join(masif_site_data_dir, '01-benchmark_surfaces')
        # self.out_surf_dir_full = os.path.join(masif_site_data_dir, f'surfaces_{face_reduction_rate}')
        self.out_surf_dir_full = os.path.join(masif_site_data_dir, f'surfaces_{face_reduction_rate}_{use_pymesh}')
        # self.ply_dir = os.path.join(masif_site_data_dir, 'ply_dir')
        self.out_rgraph_dir = os.path.join(masif_site_data_dir, 'rgraph')
        self.out_agraph_dir = os.path.join(masif_site_data_dir, 'agraph')
        # os.makedirs(self.ply_dir, exist_ok=True)
        os.makedirs(self.out_surf_dir_full, exist_ok=True)
        os.makedirs(self.out_rgraph_dir, exist_ok=True)
        os.makedirs(self.out_agraph_dir, exist_ok=True)

        # Set up systems list
        train_list = os.path.join(masif_site_data_dir, 'train_list.txt')
        test_list = os.path.join(masif_site_data_dir, 'test_list.txt')
        train_names = set([name.strip() for name in open(train_list, 'r').readlines()])
        test_names = set([name.strip() for name in open(test_list, 'r').readlines()])
        self.all_sys = list(train_names.union(test_names))
        self.recompute_surfaces = recompute_surfaces
        self.recompute_graphs = recompute_graphs

    def __len__(self):
        return len(self.all_sys)

    def __getitem__(self, idx):
        pdb_name = self.all_sys[idx]
        pdb_path = os.path.join(self.pdb_dir, pdb_name + '.pdb')
        ply_path = os.path.join(self.ply_dir, pdb_name + '.ply')
        # out_ply_path = os.path.join(self.ply_dir, pdb_name + '.ply')
        surface_full_dump = os.path.join(self.out_surf_dir_full, f'{pdb_name}.pt')
        agraph_dump = os.path.join(self.out_agraph_dir, f'{pdb_name}.pt')
        rgraph_dump = os.path.join(self.out_rgraph_dir, f'{pdb_name}.pt')
        try:
            # From there get surface and both graphs
            if self.recompute_surfaces or not os.path.exists(surface_full_dump):
                # Made a version without pymesh to load the initial data
                if self.use_pymesh:
                    import pymesh
                    mesh = pymesh.load_mesh(ply_path)
                    verts = mesh.vertices.astype(np.float32)
                    faces = mesh.faces.astype(np.int32)
                    iface_labels = mesh.get_attribute("vertex_iface").astype(np.int32)
                else:
                    from plyfile import PlyData, PlyElement
                    with open(ply_path, 'rb') as f:
                        plydata = PlyData.read(f)
                        vx = plydata['vertex']['x']
                        vy = plydata['vertex']['y']
                        vz = plydata['vertex']['z']
                        verts = np.stack((vx, vy, vz), axis=1).astype(np.float32)
                        faces = np.stack(plydata['face']['vertex_indices'], axis=0).astype(np.int32)
                        iface_labels = plydata['vertex']['iface'].astype(np.int32)
                surface = SurfaceObject.from_verts_faces(verts=verts, faces=faces,
                                                         face_reduction_rate=self.face_reduction_rate)
                iface_labels = torch.from_numpy(iface_labels)
                # If coarsened, we need to adapt the iface_labels on the new verts
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
                surface.save_torch(surface_full_dump)

            if self.recompute_graphs or not os.path.exists(agraph_dump) or not os.path.exists(rgraph_dump):
                arrays = parse_pdb_path(pdb_path)

                # create atomgraph
                if self.recompute_graphs or not os.path.exists(agraph_dump):
                    agraph = AtomGraphBuilder().arrays_to_agraph(arrays)
                    torch.save(agraph, open(agraph_dump, 'wb'))

                # create residuegraph
                if self.recompute_graphs or not os.path.exists(rgraph_dump):
                    rgraph_builder = ResidueGraphBuilder(add_pronet=True, add_esm=False)
                    rgraph = rgraph_builder.arrays_to_resgraph(arrays)
                    torch.save(rgraph, open(rgraph_dump, 'wb'))
            success = 1
        except Exception as e:
            print(pdb_name, e)
            success = 0
        return success


def do_all(dataset, num_workers=4, prefetch_factor=100):
    prefetch_factor = prefetch_factor if num_workers > 0 else 2
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=1,
                            prefetch_factor=prefetch_factor,
                            collate_fn=lambda x: x[0])
    total_success = 0
    t0 = time.time()
    for i, success in enumerate(dataloader):
        # if i > 5: break
        total_success += int(success)
        if not i % 5:
            print(f'Processed {i + 1}/{len(dataloader)}, in {time.time() - t0:.3f}s, with {total_success} successes')
            # => Processed 3351/3362, with 3328 successes ~1% failed systems, mostly missing ply files


if __name__ == '__main__':
    pass
    recompute_graphs = False
    recompute_surfaces = True
    for use_pymesh in (True, False):
        for face_red in [0.1, 0.2, 0.5, 0.9, 1.0]:
            dataset = PreProcessPDBDataset(recompute_surfaces=recompute_surfaces, recompute_graphs=recompute_graphs,
                                           face_reduction_rate=face_red, use_pymesh=use_pymesh)
            # do_all(dataset, num_workers=0)
            do_all(dataset, num_workers=40)
