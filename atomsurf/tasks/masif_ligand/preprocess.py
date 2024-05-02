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


class PreprocessPatchDataset(Dataset):
    def __init__(self, recompute=False):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
        self.patch_dir = os.path.join(masif_ligand_data_dir, 'dataset_MasifLigand')
        self.out_surf_dir_hmr = os.path.join(masif_ligand_data_dir, 'surf_hmr')
        self.out_surf_dir_ours = os.path.join(masif_ligand_data_dir, 'surf_ours_2')
        self.patches = list(os.listdir(self.patch_dir))
        self.recompute = recompute
        os.makedirs(self.out_surf_dir_ours, exist_ok=True)
        os.makedirs(self.out_surf_dir_hmr, exist_ok=True)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        path_torch_name = patch.replace('.npz', '.pt')
        surface_ours_dump = os.path.join(self.out_surf_dir_ours, path_torch_name)
        surface_hmr_dump = os.path.join(self.out_surf_dir_hmr, path_torch_name)

        try:
            patch_in = os.path.join(self.patch_dir, patch)
            data = np.load(patch_in, allow_pickle=True)
            verts = data['pkt_verts']
            faces = data['pkt_faces'].astype(int)
            # eigen_vals = data['eigen_vals']
            # eigen_vecs = data['eigen_vecs']
            # mass = data['mass'].item().todense()

            # Compare different ways to produce surface, our, using HMR preprocs, HMR eigenvecs cached...
            # Ours from verts faces (pdb further)
            if self.recompute or not os.path.exists(surface_ours_dump):
                surface_ours = SurfaceObject.from_verts_faces(verts=verts, faces=faces)
                surface_ours.add_geom_feats()
                surface_ours.save_torch(surface_ours_dump)

            # Using HMR preproc
            if self.recompute or not os.path.exists(surface_hmr_dump):
                surface_ours_hmr = SurfaceObject.from_verts_faces(verts=verts, faces=faces, use_fem_decomp=True)
                surface_ours_hmr.add_geom_feats()
                surface_ours_hmr.save_torch(surface_hmr_dump)

            # These are close, but not identical. Usually there is a max difference of about 0.003
            # for a_hmr, a_ours in zip(surface_ours.evecs.T, surface_ours_hmr.evecs[:, :40].T):
            #     max_diff = torch.max(a_hmr - a_ours)
            #     max_diff_opp = torch.max(a_hmr + a_ours)
            #     a = 1

            # SANITY CHECK, our recomputation is the same as cached vectors => OK, evecs are the same or opposite
            # Using HMR cached, we still need our processing to get grads operators
            # surface_hmr = SurfaceObject(verts=verts, faces=faces, mass=mass, L=surface_ours_hmr.L,
            #                             evals=eigen_vals, evecs=eigen_vecs,
            #                             gradX=surface_ours_hmr.gradX, gradY=surface_ours_hmr.gradY)
            # surface_hmr.save_torch(surface_hmr_dump)
            # for a_hmr, a_ours in zip(surface_hmr.evecs.T, surface_ours_hmr.evecs[:, :40].T):
            #     max_diff = torch.max(a_hmr - a_ours)
            #     max_diff_opp = torch.max(a_hmr + a_ours)
            success = 1
        except Exception as e:
            print(e)
            success = 0
        return success


class PreProcessPDBDataset(Dataset):

    def __init__(self, recompute=True, data_dir=None, max_vert_number=100000):

        script_dir = os.path.dirname(os.path.realpath(__file__))
        if data_dir is None:
            masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
        else:
            masif_ligand_data_dir = data_dir
        self.pdb_dir = os.path.join(masif_ligand_data_dir, 'raw_data_MasifLigand', 'pdb')
        self.out_surf_dir_full = os.path.join(masif_ligand_data_dir, 'surf_full')
        self.out_rgraph_dir = os.path.join(masif_ligand_data_dir, 'rgraph')
        self.out_agraph_dir = os.path.join(masif_ligand_data_dir, 'agraph')

        os.makedirs(self.out_surf_dir_full, exist_ok=True)
        os.makedirs(self.out_rgraph_dir, exist_ok=True)
        os.makedirs(self.out_agraph_dir, exist_ok=True)

        self.all_pdbs = os.listdir(self.pdb_dir)  # TODO filter
        self.recompute = recompute
        self.max_vert_number = max_vert_number

    def __len__(self):
        return len(self.all_pdbs)

    def __getitem__(self, idx):
        pdb = self.all_pdbs[idx]
        try:
            pdb_path = os.path.join(self.pdb_dir, pdb)
            name = pdb.rstrip('.pdb')
            surface_full_dump = os.path.join(self.out_surf_dir_full, f'{name}.pt')
            agraph_dump = os.path.join(self.out_agraph_dir, f'{name}.pt')
            rgraph_dump = os.path.join(self.out_rgraph_dir, f'{name}.pt')

            if self.recompute or not os.path.exists(surface_full_dump):
                surface = SurfaceObject.from_pdb_path(pdb_path, out_ply_path=None, max_vert_number=self.max_vert_number)
                surface.add_geom_feats()
                surface.save_torch(surface_full_dump)

            if self.recompute or not os.path.exists(surface_full_dump) or not os.path.exists(surface_full_dump):
                arrays = parse_pdb_path(pdb_path)

                # create atomgraph
                if self.recompute or not os.path.exists(agraph_dump):
                    agraph_builder = AtomGraphBuilder()
                    agraph = agraph_builder.arrays_to_agraph(arrays)
                    torch.save(agraph, open(agraph_dump, 'wb'))

                # create residuegraph
                if self.recompute or not os.path.exists(rgraph_dump):
                    rgraph_builder = ResidueGraphBuilder(add_pronet=True, add_esm=False)
                    rgraph = rgraph_builder.arrays_to_resgraph(arrays)
                    torch.save(rgraph, open(rgraph_dump, 'wb'))
            success = 1
        except Exception as e:
            print(pdb, e)
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
    recompute = False
    dataset = PreprocessPatchDataset(recompute=recompute)
    do_all(dataset, num_workers=4)

    # dataset = PreProcessPDBDataset(recompute=recompute, max_vert_number=100000)
    # do_all(dataset, num_workers=4)
