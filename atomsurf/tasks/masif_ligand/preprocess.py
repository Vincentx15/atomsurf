import os
import sys
import time

import numpy as np
import torch

from torch.utils.data import Dataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.protein.create_esm import get_esm_embedding_batch
from atomsurf.utils.data_utils import PreprocessDataset
from atomsurf.utils.python_utils import do_all

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


class PreprocessPatchDataset(Dataset):
    def __init__(self, data_dir=None, recompute=False, face_reduction_rate=1., use_pymesh=True):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if data_dir is None:
            masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
        else:
            masif_ligand_data_dir = data_dir
        self.patch_dir = os.path.join(masif_ligand_data_dir, 'dataset_MasifLigand')
        self.out_surf_dir_ours = os.path.join(masif_ligand_data_dir, f'surf_{face_reduction_rate}_{use_pymesh}')

        self.face_reduction_rate = face_reduction_rate
        self.use_pymesh = use_pymesh

        self.patches = list(os.listdir(self.patch_dir))
        self.recompute = recompute
        os.makedirs(self.out_surf_dir_ours, exist_ok=True)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        path_torch_name = patch.replace('.npz', '.pt')
        surface_ours_dump = os.path.join(self.out_surf_dir_ours, path_torch_name)
        try:
            patch_in = os.path.join(self.patch_dir, patch)
            data = np.load(patch_in, allow_pickle=True)
            verts = data['pkt_verts']
            faces = data['pkt_faces'].astype(int)

            if self.recompute or not os.path.exists(surface_ours_dump):
                surface_ours = SurfaceObject.from_verts_faces(verts=verts, faces=faces,
                                                              face_reduction_rate=self.face_reduction_rate,
                                                              use_pymesh=self.use_pymesh)
                surface_ours.add_geom_feats()
                surface_ours.save_torch(surface_ours_dump)
            success = 1
        except Exception as e:
            print(e)
            success = 0
        return success


class PreProcessPDBDataset(PreprocessDataset):

    def __init__(self, data_dir=None, compute_s=True, recompute_s=False, recompute_g=False,
                 max_vert_number=100000, face_reduction_rate=1.0, use_pymesh=True):
        if data_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')

        super().__init__(data_dir=data_dir, recompute_s=recompute_s, recompute_g=recompute_g,
                         max_vert_number=max_vert_number, face_reduction_rate=face_reduction_rate)
        # Compared to super(), we redefine the original PDB location and the
        # out_surf dir (since those are "_full", as opposed to patches)
        self.pdb_dir = os.path.join(data_dir, 'raw_data_MasifLigand', 'pdb')
        surface_dirname = f'surfaces_full_{face_reduction_rate}{f"_{use_pymesh}" if use_pymesh is not None else ""}'
        self.out_surf_dir = os.path.join(data_dir, surface_dirname)
        os.makedirs(self.out_surf_dir, exist_ok=True)
        self.compute_s = compute_s

        self.all_pdbs = self.get_all_pdbs()

    def __getitem__(self, idx):
        pdb = self.all_pdbs[idx]
        name = pdb[0:-4]
        if self.compute_s:
            success = self.name_to_surf_graphs(name)
        else:
            success = self.name_to_graphs(name)
        return success


if __name__ == '__main__':
    pass
    recompute = False
    recompute_s = False
    recompute_g = False
    use_pymesh = False
    dataset = PreprocessPatchDataset(recompute=recompute,
                                     face_reduction_rate=1.0,
                                     use_pymesh=use_pymesh)
    do_all(dataset, num_workers=20)
    dataset = PreProcessPDBDataset(recompute_g=recompute_g,
                                   recompute_s=recompute_s,
                                   compute_s=False,
                                   face_reduction_rate=1.0,
                                   use_pymesh=use_pymesh)
    do_all(dataset, num_workers=20)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
    pdb_dir = os.path.join(masif_ligand_data_dir, 'raw_data_MasifLigand', 'pdb')
    out_esm_dir = os.path.join(masif_ligand_data_dir, 'esm_embs')
    get_esm_embedding_batch(in_pdbs_dir=pdb_dir, dump_dir=out_esm_dir)
