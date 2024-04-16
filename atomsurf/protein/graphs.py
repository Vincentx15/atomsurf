import os
import sys

from Bio.PDB import PDBParser
import numpy as np

import scipy.spatial as ss
import torch
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


def parse_pdb_path(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure("toto", pdb_path)

    amino_types = []  # size: (n_amino,)
    atom_amino_id = []  # size: (n_atom,)
    atom_names = []  # size: (n_atom,)
    atom_pos = []  # size: (n_atom,3)
    res_id = 0
    # Iterate over all residues in a model
    for residue in structure.get_residues():
        resname = residue.get_resname()
        # resname = protein_letters_3to1[resname.title()]
        if resname.upper() not in res_type_dict:
            resname = 'UNK'
        resname = res_type_dict[resname.upper()]
        amino_types.append(resname)
        for atom in residue.get_atoms():
            # skip h
            if atom.get_name().startswith("H"):
                continue
            atom_amino_id.append(res_id)
            atom_names.append(atom.get_name())
            atom_pos.append(atom.get_coord())
        res_id += 1

    amino_types = np.asarray(amino_types)
    atom_amino_id = np.asarray(atom_amino_id)
    atom_names = np.asarray(atom_names)
    atom_pos = np.asarray(atom_pos)
    return amino_types, atom_amino_id, atom_names, atom_pos


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


class PDBGraph(Data):
    def __init__(self, node_pos, edge_index=None, named_one_hot_features=None, named_features=None, flat_features=None,
                 **kwargs):
        super(PDBGraph, self).__init__(edge_index=edge_index, **kwargs)
        self.node_pos = node_pos
        self.named_one_hot_features = named_one_hot_features
        self.named_features = named_features
        self.flat_features = flat_features

    def add_flat_features(self, value):
        assert len(value) == len(self.node_pos)
        if self.flat_features is None:
            self.flat_features = value
        else:
            self.flat_features = torch.cat((self.flat_features, value), 0)

    def add_named_features(self, key, value):
        assert len(value) == len(self.node_pos)
        if self.named_features is None:
            self.named_features = {key: value}
        else:
            self.named_features[key] = value

    def add_named_oh_features(self, key, value):
        assert len(value) == len(self.node_pos)
        if self.named_one_hot_features is None:
            self.named_one_hot_features = {key: value}
        else:
            self.named_one_hot_features[key] = value

    @staticmethod
    def load(npz_path):
        npz_file = np.load(npz_path, allow_pickle=True)
        node_pos = npz_file['node_pos']
        edge_index = npz_file['edge_index']
        named_one_hot_features = npz_file['named_one_hot_features']
        flat_features = npz_file['flat_features']
        return PDBGraph(node_pos, edge_index, named_one_hot_features, flat_features)

    def save(self, npz_path):
        pass

if __name__ == "__main__":
    pass
