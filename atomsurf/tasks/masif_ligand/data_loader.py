import os
import sys

import os
import igl
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

import scipy.spatial as ss
from sklearn.neighbors import BallTree
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import numpy as np
import pickle
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))



ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
type_idx = {type_: ix for ix, type_ in enumerate(ligands)}


def get_systems_from_ligands(split_list_path, ligands_path, out_path=None, recompute=False):
    if out_path is not None:
        if os.path.exists(out_path) and not recompute:
            all_pockets = pickle.load(open(out_path, "rb"))
            return all_pockets
    all_pockets = {}
    split_list = open(split_list_path).read().splitlines()
    for pdb_chains in split_list:
        pdb = pdb_chains.split('_')[0]
        ligand_coords = np.load(os.path.join(ligands_path, f"{pdb}_ligand_coords.npy"), allow_pickle=True,
                                encoding='bytes')
        ligand_types = np.load(os.path.join(ligands_path, f"{pdb}_ligand_types.npy"))
        ligand_types = [lig.decode() for lig in ligand_types]
        for ix, (lig_type, lig_coord) in enumerate(zip(ligand_types, ligand_coords)):
            pocket = f'{pdb_chains}_patch_{ix}_{lig_type}.npz'
            all_pockets[pocket] = np.reshape(lig_coord, (-1, 3)), type_idx[lig_type]
            a = 1
    if out_path is not None:
        pickle.dump(all_pockets, open(out_path, "wb"))
    return all_pockets



if __name__ == '__main__':
    pass
    masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
    train_splits_dir = os.path.join(masif_ligand_data_dir, 'raw_data_MasifLigand', 'splits')
    ligands_path = os.path.join(masif_ligand_data_dir, 'raw_data_MasifLigand', 'ligand')
    for split in ['train', 'val', 'test']:
        train_splits_path = os.path.join(train_splits_dir, f'{split}-list.txt')
        out_path = os.path.join(train_splits_dir, f'.p')
        train_systems = get_systems_from_ligands(train_splits_path,
                                                 ligands_path=ligands_path,
                                                 out_path=out_path,
                                                 recompute=False)
