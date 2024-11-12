import os
import sys

import numpy as np
from pathlib import Path
import shutil
from subprocess import Popen, PIPE
import torch
from torch_geometric.data import Data, Batch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from atomsurf.protein.graphs import parse_pdb_path, atom_coords_to_edges, res_type_to_hphob
from atomsurf.protein.features import Features, FeaturesHolder
from atomsurf.utils.torch_utils import safe_to_torch


class AtomGraph(Data, FeaturesHolder):
    def __init__(self, node_pos, res_map, edge_index=None, edge_attr=None, features=None, node_names=None, **kwargs):
        edge_index = safe_to_torch(edge_index)
        edge_attr = safe_to_torch(edge_attr)
        super(AtomGraph, self).__init__(edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        self.node_pos = safe_to_torch(node_pos)
        self.res_map = safe_to_torch(res_map)
        self.num_atoms = len(node_pos) if node_pos is not None else 0
        self.num_res = res_map.max() + 1 if res_map is not None else 0
        self.node_names = node_names
        # Useful for bipartite computations
        self.node_len = self.num_atoms
        if features is None:
            self.features = Features(num_nodes=self.num_atoms, res_map=res_map)
        else:
            self.features = features

    def collapse_feats_residues(self, feats, reduce='mean'):
        from torch_scatter import scatter
        res_feats = scatter(feats, index=self.res_map.long(), reduce=reduce, dim=0)
        return res_feats


class AGraphBatch(Batch):
    """
    This class is useful for PyG Batching

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def batch_from_data_list(cls, data_list):
        batch = Batch.from_data_list(data_list)
        batch = batch.contiguous()
        agraph_batch = cls()
        agraph_batch.__dict__.update(batch.__dict__)
        return agraph_batch


class AtomGraphBuilder:
    def __init__(self):
        pass

    def arrays_to_agraph(self, arrays):
        amino_types, _, atom_amino_id, _, atom_types, atom_pos, atom_charge, atom_radius, res_sse, _, atom_ids = arrays
        edge_index, edge_dists = atom_coords_to_edges(atom_pos)
        atom_graph = AtomGraph(node_pos=atom_pos,
                               res_map=atom_amino_id,
                               edge_index=edge_index,
                               edge_attr=edge_dists,
                               node_names=atom_ids)
        # Add res_level features to be expanded
        atom_graph.features.add_named_oh_features('amino_types', amino_types, nclasses=21)
        atom_graph.features.add_named_oh_features('sse', res_sse, nclasses=8)
        hphob = np.asarray([res_type_to_hphob[amino_type] for amino_type in amino_types], dtype=np.float32)
        atom_graph.features.add_named_features('hphobs', hphob)

        # Add atom_level features
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
    atom_graph.expand_features()
    grouped = atom_graph.collapse_feats_residues(atom_graph.x)
    a = 1
