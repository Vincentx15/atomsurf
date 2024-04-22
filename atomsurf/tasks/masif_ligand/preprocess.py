import os
import sys

import numpy as np
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.main_data import create_protein
from atomsurf.protein.surfaces import SurfaceObject

if __name__ == '__main__':
    pass
    # In our current pipeline, we used to do MEAN POOLING over small patch.
    # Maybe we could use the lig_coords directly TODO: look how pooling happens in masif
    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
    pdb_dir = os.path.join(masif_ligand_data_dir, 'raw_data_MasifLigand', 'pdb')
    patch_dir = os.path.join(masif_ligand_data_dir, 'dataset_MasifLigand')

    out_surf_dir_hmr = os.path.join(masif_ligand_data_dir, 'surf_hmr')
    out_surf_dir_ours = os.path.join(masif_ligand_data_dir, 'surf_ours')

    out_surf_dir_full = os.path.join(masif_ligand_data_dir, 'surf_full')
    out_rgraph_dir = os.path.join(masif_ligand_data_dir, 'rgraph')
    out_agraph_dir = os.path.join(masif_ligand_data_dir, 'agraph')

    os.makedirs(out_surf_dir_ours, exist_ok=True)
    os.makedirs(out_surf_dir_hmr, exist_ok=True)
    os.makedirs(out_surf_dir_full, exist_ok=True)
    os.makedirs(out_rgraph_dir, exist_ok=True)
    os.makedirs(out_agraph_dir, exist_ok=True)

    for i, patch in enumerate(os.listdir(patch_dir)):
        break
        print(i)
        path_torch_name = patch.replace('.npz', '.pt')
        surface_ours_dump = os.path.join(out_surf_dir_ours, path_torch_name)
        surface_hmr_dump = os.path.join(out_surf_dir_hmr, path_torch_name)
        try:
            patch_in = os.path.join(patch_dir, patch)
            data = np.load(patch_in, allow_pickle=True)
            label = data['label']
            verts = data['pkt_verts']
            faces = data['pkt_faces'].astype(int)
            eigen_vals = data['eigen_vals']
            eigen_vecs = data['eigen_vecs']
            mass = data['mass'].item().todense()

            # Compare different ways to produce surface, our, using HMR preprocs, HMR eigenvecs cached...
            # Ours from verts faces (pdb further)
            surface_ours = SurfaceObject.from_verts_faces(verts=verts, faces=faces)
            surface_ours.add_geom_feats()
            surface_ours.save_torch(surface_ours_dump)

            # Using HMR preproc
            surface_ours_hmr = SurfaceObject.from_verts_faces(verts=verts, faces=faces, use_hmr_decomp=True)
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
        except Exception as e:
            print(e)

    for i, pdb in enumerate(os.listdir(pdb_dir)):
        print(i)
        pdb_path = os.path.join(pdb_dir, pdb)
        name = pdb.rstrip('.pdb')
        surface_full_dump = os.path.join(out_surf_dir_full, f'{name}.pt')
        agraph_dump = os.path.join(out_agraph_dir, f'{name}.pt')
        rgraph_dump = os.path.join(out_rgraph_dir, f'{name}.pt')
        try:
            create_protein(pdb_path=pdb_path,
                           dump_ply=None,
                           dump_surf=surface_full_dump,
                           dump_agraph=agraph_dump,
                           dump_rgraph=rgraph_dump)
        except Exception as e:
            print(e)
