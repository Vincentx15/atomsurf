import os
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from atomsurf.protein.graphs import parse_pdb_path, atom_coords_to_edges
from atomsurf.protein.features import Features


class AtomGraph(Data):
    def __init__(self, node_pos, res_map, edge_index=None, features=None, **kwargs):
        super(AtomGraph, self).__init__(edge_index=edge_index, **kwargs)
        self.node_pos = self.safe_to_torch(node_pos)
        self.res_map = self.safe_to_torch(res_map)
        self.num_atoms = len(node_pos)
        self.num_res = res_map.max() + 1
        if features is None:
            self.features = Features(num_nodes=self.num_atoms, res_map=res_map)
        else:
            self.features = features


class AtomGraphBuilder:
    def __init__(self):
        pass

    def pdb_to_atomgraph(self, pdb_path):
        amino_types, atom_amino_id, atom_names, atom_pos = parse_pdb_path(pdb_path)
        edge_index, edge_dists = atom_coords_to_edges(atom_pos)
        atom_graph = Data(node_pos=atom_pos,
                              res_map=atom_amino_id,
                              edge_index=edge_index,
                              edge_attr=edge_dists)
        atom_graph.add_named_oh_features('amino_types', amino_types)
        atom_graph.add_named_oh_features('atom_name', atom_names)
        return atom_graph


if __name__ == "__main__":
    pass
    pdb = "../../data/example_files/4kt3.pdb"
    atom_graph_builder = AtomGraphBuilder()
    atom_graph = atom_graph_builder.pdb_to_atomgraph(pdb)
