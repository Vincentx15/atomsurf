import os
import sys

from Bio.PDB import PDBParser
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
import shutil
from subprocess import Popen, PIPE
import time
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.graphs import parse_pdb_path, extract_chains
from atomsurf.protein.atom_graph import AtomGraphBuilder
from atomsurf.protein.residue_graph import ResidueGraphBuilder
from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.protein.create_esm import get_esm_embedding_batch
from atomsurf.utils.python_utils import do_all
from atomsurf.utils.data_utils import PreprocessDataset

chothia_cdr_def = {
    "L1": (24, 34), "L2": (50, 56), "L3": (89, 97),
    "H1": (26, 32), "H2": (52, 56), "H3": (95, 102)}


def parse_pdb_path_mini(pdb_path):
    """
    Only get minimal information, no need to run pdb2pqr here
    :param pdb_path:
    :return:
    """
    pdb_path = Path(pdb_path)
    parser = PDBParser()
    structure = parser.get_structure("toto", pdb_path)
    atom_amino_id = []  # size: (n_atom,)
    atom_pos = []  # size: (n_atom,3)
    real_resid = []
    res_id = 0
    for residue in structure.get_residues():
        # HETATM
        if residue.id[0] != " ":
            continue
        for atom in residue.get_atoms():
            # Skip H
            if atom.get_name().startswith("H"):
                continue
            atom_pos.append(atom.get_coord())
            atom_amino_id.append(res_id)
            real_resid.append((residue.full_id[2] + str(residue.id[1])))
        res_id += 1
    atom_amino_id = np.asarray(atom_amino_id, dtype=np.int32)
    atom_pos = np.asarray(atom_pos, dtype=np.float32)
    real_resid = np.asarray(real_resid)
    return atom_amino_id, atom_pos, real_resid


def get_pdb(pdb_code="", out_dir="", recompute=False):
    out_path = os.path.join(out_dir, f"{pdb_code}.pdb")
    if recompute or not os.path.exists(out_path):
        os.system(f"wget -qnc -O {out_path} https://files.rcsb.org/view/{pdb_code}.pdb ")


class ExtractAALabelDataset(Dataset):
    def __init__(self, csv_file, datadir, recompute=False):
        self.pdblist = pd.read_csv(csv_file)
        self.datadir = datadir
        self.recompute = recompute

    def __len__(self):
        return len(self.pdblist)

    def __getitem__(self, idx):
        row = self.pdblist.loc[idx]
        try:

            # Parse chain information to get a system id
            pdb, chain_H, chain_L, chain_ag = row['pdb'], row['Hchain'], row['Lchain'], row['antigen_chain']
            chains_HL, chains_ag = [chain_H, chain_L], [chain_ag]  # List of chains to extract
            system_id = f"{pdb}_{chain_H}{chain_L}_{''.join(chain_ag)}"

            input_pdb = os.path.join(self.datadir, f'{pdb}.pdb')
            output_pdbab = os.path.join(self.datadir, f'{system_id}_ab.pdb')
            output_pdbag = os.path.join(self.datadir, f'{system_id}_ag.pdb')
            out_json = os.path.join(self.datadir, f'{system_id}.json')
            if not self.recompute and (
                    os.path.exists(output_pdbab) and os.path.exists(output_pdbag) and os.path.exists(out_json)):
                return 1

            # Extract relevant chains and dump extracted pdbs
            extract_chains(input_pdb, output_pdbab, chains_HL)
            extract_chains(input_pdb, output_pdbag, chains_ag)

            # Open those and compute distances
            arrays1 = parse_pdb_path_mini(output_pdbab)
            arrays2 = parse_pdb_path_mini(output_pdbag)
            atom_amino_id1, atom_pos1, real_resid1 = arrays1
            atom_amino_id2, atom_pos2, real_resid2 = arrays2
            dists = cdist(atom_pos1, atom_pos2)
            idx1, idx2 = np.where(dists < 4.5)
            contact_mask1 = np.zeros_like(real_resid1, dtype=bool)
            contact_mask2 = np.zeros_like(real_resid2, dtype=bool)
            contact_mask1[idx1] = True
            contact_mask2[idx2] = True

            # GEP additionally filters on the CDR to define residues in contact
            chain = {'H': chain_H, 'L': chain_L, 'ag': chain_ag}
            cdr_region = []
            for cdrname, pos in chothia_cdr_def.items():
                cdr_region += ([chain[cdrname[0]] + str(i) for i in range(pos[0] - 2, pos[1] + 2 + 1)])
            cdr_mask = np.zeros_like(real_resid1, dtype=bool)
            for i in range(0, len(real_resid1)):
                if real_resid1[i] in cdr_region:
                    cdr_mask[i] = True
            cdr_res = np.unique(atom_amino_id1[cdr_mask])
            cdr_contact_res = np.unique(atom_amino_id1[cdr_mask & contact_mask1])
            ag_contact_res = np.unique(atom_amino_id2[contact_mask2])

            # Finally dump those results in a json
            label_dict = {}
            label_dict['cdr'] = cdr_res.tolist()
            label_dict['cdr_contact'] = cdr_contact_res.tolist()
            label_dict['ag_contact'] = ag_contact_res.tolist()
            contact_pair = [(i, j) for i, j in zip(atom_amino_id1[idx1], atom_amino_id2[idx2])]
            contact_pair = list(set(contact_pair))
            label_dict['cdr_ag_pair'] = np.array(contact_pair).tolist()
            with open(out_json, "w") as outfile:
                json.dump(label_dict, outfile)
            return 1
        except Exception as e:
            print('failed', row['pdb'], e)
            return 0


class PreProcessABAGDataset(PreprocessDataset):

    def __init__(self, csv_file, data_dir=None, recompute_s=False, recompute_g=False,
                 max_vert_number=100000, face_reduction_rate=1.0):
        if data_dir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'abag')

        super().__init__(data_dir=data_dir, recompute_s=recompute_s, recompute_g=recompute_g,
                         max_vert_number=max_vert_number, face_reduction_rate=face_reduction_rate)
        self.all_pdbs = pd.read_csv(csv_file)

    def __getitem__(self, idx):
        row = self.all_pdbs.loc[idx]
        pdb, chain_H, chain_L, chain_ag = row['pdb'], row['Hchain'], row['Lchain'], row['antigen_chain']
        system_id = f"{pdb}_{chain_H}{chain_L}_{''.join(chain_ag)}"
        name_ab = f'{system_id}_ab'
        name_ag = f'{system_id}_ag'
        success1 = self.name_to_surf_graphs(name_ab)
        success2 = self.name_to_surf_graphs(name_ag)
        return success1 and success2


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    abag_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'abag')
    file_paths = {'train': 'train.csv', 'val': 'val_pecan_aligned.csv', 'test': 'test70.csv'}
    pdb_dir = os.path.join(abag_data_dir, 'pdb')
    os.makedirs(pdb_dir, exist_ok=True)


    def get_all_pdbs():
        dfs = []
        for fp in file_paths.values():
            dfs.append(pd.read_csv(os.path.join(abag_data_dir, fp))['pdb'])
        pdbs = pd.concat(dfs).unique()
        for pdb in tqdm(pdbs):
            get_pdb(pdb_code=pdb, out_dir=pdb_dir)


    # get_all_pdbs()
    for mode, fp in file_paths.items():
        print("Preprocessing on", mode)
        csv = os.path.join(abag_data_dir, fp)
        extract = ExtractAALabelDataset(csv_file=csv, datadir=pdb_dir)
        do_all(extract, num_workers=20)
        process_pdb = PreProcessABAGDataset(csv_file=os.path.join(abag_data_dir, fp))
        do_all(process_pdb, num_workers=20)
