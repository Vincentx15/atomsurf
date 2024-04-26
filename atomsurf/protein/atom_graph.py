import os
import sys

import numpy as np
from pathlib import Path
import shutil
from subprocess import Popen, PIPE
import torch
from torch_geometric.data import Data

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from atomsurf.protein.graphs import parse_pdb_path, atom_coords_to_edges
from atomsurf.protein.features import Features, FeaturesHolder
from atomsurf.utils.helpers import safe_to_torch


class AtomGraph(Data, FeaturesHolder):
    def __init__(self, node_pos, res_map, edge_index=None, features=None, **kwargs):
        super(AtomGraph, self).__init__(edge_index=edge_index, **kwargs)
        self.node_pos = safe_to_torch(node_pos)
        self.res_map = safe_to_torch(res_map)
        self.num_atoms = len(node_pos)
        self.num_res = res_map.max() + 1
        if features is None:
            self.features = Features(num_nodes=self.num_atoms, res_map=res_map)
        else:
            self.features = features

    @staticmethod
    def batch_from_data_list(data_list):
        # filter out None
        data_list = [data for data in data_list if data is not None]
        if len(data_list) == 0:
            return None
        return data_list


class AtomGraphBuilder:
    def __init__(self):
        pass

    def arrays_to_agraph(self, arrays):
        amino_types, atom_chain_id, atom_amino_id, atom_names, atom_types, atom_pos, atom_charge, atom_radius, res_sse = arrays
        edge_index, edge_dists = atom_coords_to_edges(atom_pos)
        atom_graph = AtomGraph(node_pos=atom_pos,
                               res_map=atom_amino_id,
                               edge_index=edge_index,
                               edge_attr=edge_dists)
        atom_graph.features.add_named_oh_features('amino_types', amino_types, nclasses=21)
        atom_graph.features.add_named_oh_features('atom_types', atom_types, nclasses=12)
        atom_graph.features.add_named_features('charge', atom_charge)
        atom_graph.features.add_named_features('radius', atom_radius)
        return atom_graph

    def pdb_to_atomgraph(self, pdb_path):
        arrays = parse_pdb_path(pdb_path)
        return self.arrays_to_agraph(arrays)


if __name__ == "__main__":
    pass
    pdb = "../../data/example_files/4kt3.pdb"
    atomgraph_path = "../../data/example_files/4kt3_atomgraph.pt"
    atom_graph_builder = AtomGraphBuilder()
    atom_graph = atom_graph_builder.pdb_to_atomgraph(pdb)
    torch.save(atom_graph, atomgraph_path)
    atom_graph = torch.load(atomgraph_path)
    a = 1
