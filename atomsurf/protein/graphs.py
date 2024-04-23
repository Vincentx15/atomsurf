import os
import sys

from Bio.PDB import PDBParser, MMCIFParser
import numpy as np

import scipy.spatial as ss
import torch
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

# atom type label for one-hot-encoding
atom_type_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'P': 6, 'Cl': 7, 'Se': 8,
                  'Br': 9, 'I': 10, 'UNK': 11}
# residue type label for one-hot-encoding
res_type_dict = {
    'ALA': 0, 'GLY': 1, 'SER': 2, 'THR': 3, 'LEU': 4, 'ILE': 5, 'VAL': 6, 'ASN': 7, 'GLN': 8, 'ARG': 9, 'HIS': 10,
    'TRP': 11, 'PHE': 12, 'TYR': 13, 'GLU': 14, 'ASP': 15, 'LYS': 16, 'PRO': 17, 'CYS': 18, 'MET': 19, 'UNK': 20, }

protein_letters_1to3 = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "X": "Unk"
}

protein_letters_3to1 = {value.upper(): key for key, value in protein_letters_1to3.items()}

res_type_idx_to_1 = {
    idx: protein_letters_3to1[res_type] for res_type, idx in res_type_dict.items()
}

# Kyte Doolittle scale for hydrophobicity
hydrophob_dict = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7,
    'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5,
    'LYS': -3.9, 'ARG': -4.5, 'UNK': 0.0,
}

res_type_to_hphob = {
    idx: hydrophob_dict[res_type] for res_type, idx in res_type_dict.items()
}


def parse_pdb_path(pdb_path, verbose=False):
    parser = MMCIFParser(QUIET=not verbose) if pdb_path.endswith('.cif') else PDBParser(QUIET=not verbose)
    structure = parser.get_structure("toto", pdb_path)

    amino_types = []  # size: (n_amino,)
    atom_amino_id = []  # size: (n_atom,)
    atom_names = []  # size: (n_atom,)
    atom_types = []  # size: (n_atom,)
    atom_pos = []  # size: (n_atom,3)
    res_id = 0
    # Iterate over all residues in a model
    for residue in structure.get_residues():
        # HETATM
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname()
        # resname = protein_letters_3to1[resname.title()]
        if resname.upper() not in res_type_dict:
            resname = 'UNK'
        resname = res_type_dict[resname.upper()]
        amino_types.append(resname)
        for atom in residue.get_atoms():
            # Skip H
            element = atom.element
            if atom.get_name().startswith("H"):
                continue
            if not element in atom_type_dict:
                element = 'UNK'
            atom_types.append(atom_type_dict[element])
            atom_names.append(atom.get_name())
            atom_pos.append(atom.get_coord())
            atom_amino_id.append(res_id)
        res_id += 1

    amino_types = np.asarray(amino_types)
    atom_amino_id = np.asarray(atom_amino_id)
    atom_names = np.asarray(atom_names)
    atom_types = np.asarray(atom_types)
    atom_pos = np.asarray(atom_pos)
    return amino_types, atom_amino_id, atom_names, atom_types, atom_pos


def atom_coords_to_edges(node_pos, edge_dist_cutoff=4.5):
    r"""
    Turn nodes position into neighbors graph.
    """
    # import time
    # t0 = time.time()
    kd_tree = ss.KDTree(node_pos)
    edge_tuples = list(kd_tree.query_pairs(edge_dist_cutoff))
    edges = torch.LongTensor(edge_tuples).t().contiguous()
    edges = to_undirected(edges)
    # print(f"time to pre_dist : {time.time() - t0}")

    # t0 = time.time()
    node_a = node_pos[edges[0, :]]
    node_b = node_pos[edges[1, :]]
    with torch.no_grad():
        my_edge_weights_torch = 1 / (np.linalg.norm(node_a - node_b, axis=1) + 1e-5)
    return edges, my_edge_weights_torch


if __name__ == "__main__":
    pass
