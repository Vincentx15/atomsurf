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
from atomsurf.protein.features import Features
from atomsurf.utils.helpers import safe_to_torch

#
# def compute_radius_charge_debug(pdb_path, coords):
#     """
#     Adapted from atom2d, hmr_min, l131 pdb_to_atom_info
#     :param pdb_path:
#     :return:
#     """
#     pdb2pqr_bin = shutil.which('pdb2pqr')
#     if pdb2pqr_bin is None:
#         raise RuntimeError('pdb2pqr executable not found')
#
#     try:
#         pdb_path = Path(pdb_path)
#         out_dir = pdb_path.parent
#         pdb_id = pdb_path.stem
#         pqr_path = Path(out_dir / f'{pdb_id}.pqr')
#         if not pqr_path.exists():
#             cmd = [pdb2pqr_bin, '--ff=AMBER', str(pdb_path), str(pqr_path)]
#             proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
#             stdout, stderr = proc.communicate()
#             err = stderr.decode('utf-8').strip('\n')
#             if 'CRITICAL' in err:
#                 print(f'{pdb_id} pdb2pqr failed', flush=True)
#                 return None
#
#         with open(pqr_path, 'r') as f:
#             f_read = f.readlines()
#         os.remove(pqr_path)
#         # atom_info = []
#         atom_charge_radius = []
#         new_coords = []
#         i = 0
#         for line in f_read:
#             if line[:4] == 'ATOM':
#                 assert (len(line) == 70) and (line[69] == '\n')
#                 assert line[11] == line[16] == line[54] == line[62] == ' '
#                 atom_name = line[12:16].strip()
#                 if atom_name.startswith('H'):
#                     continue
#                 if atom_name == 'OXT':
#                     continue
#                 charge = float(line[55:62])
#                 radius = float(line[63:69])
#
#                 # DEBUG :
#                 atom_coords = coords[i]
#                 i += 1
#                 my_coords = np.array(line.split()[5:8], dtype=np.float32)
#                 new_coords.append(my_coords)
#                 diff = atom_coords - my_coords
#                 if np.max(diff) > 0.0001:
#                     a = 1
#                 old_coords_local = coords[i - 10:i]
#                 new_coords_local = np.asarray(new_coords)[i - 10:i]
#                 atom_charge_radius.append([float(charge), float(radius)])
#                 res_name = line[17:20]
#                 # atom_info.append(atom_name)
#         # return np.array(atom_charge_radius, dtype=float), np.array(atom_info)
#         return np.array(atom_charge_radius, dtype=float)
#     except Exception as e:
#         print(e)
#         return None


def compute_radius_charge(pdb_path):
    """
    Adapted from atom2d, hmr_min, l131 pdb_to_atom_info
    :param pdb_path:
    :return:
    """
    pdb2pqr_bin = shutil.which('pdb2pqr')
    if pdb2pqr_bin is None:
        raise RuntimeError('pdb2pqr executable not found')

    try:
        pdb_path = Path(pdb_path)
        out_dir = pdb_path.parent
        pdb_id = pdb_path.stem
        pqr_path = Path(out_dir / f'{pdb_id}.pqr')
        if not pqr_path.exists():
            cmd = [pdb2pqr_bin, '--ff=AMBER', str(pdb_path), str(pqr_path)]
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = proc.communicate()
            err = stderr.decode('utf-8').strip('\n')
            if 'CRITICAL' in err:
                print(f'{pdb_id} pdb2pqr failed', flush=True)
                return None

        with open(pqr_path, 'r') as f:
            f_read = f.readlines()
        os.remove(pqr_path)
        # atom_info = []
        atom_charge_radius = []
        for line in f_read:
            if line[:4] == 'ATOM':
                assert (len(line) == 70) and (line[69] == '\n')
                assert line[11] == line[16] == line[54] == line[62] == ' '
                atom_name = line[12:16].strip()
                if atom_name.startswith('H'):
                    continue
                if atom_name == 'OXT':
                    continue
                charge = float(line[55:62])
                radius = float(line[63:69])
                atom_charge_radius.append([float(charge), float(radius)])
                # res_name = line[17:20]
                # atom_info.append(atom_name)
        # return np.array(atom_charge_radius, dtype=float), np.array(atom_info)
        return np.array(atom_charge_radius, dtype=float)
    except Exception as e:
        print(e)
        return None


class AtomGraph(Data):
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


class AtomGraphBuilder:
    def __init__(self):
        pass

    def pdb_to_atomgraph(self, pdb_path):
        amino_types, atom_amino_id, atom_names, atom_types, atom_pos = parse_pdb_path(pdb_path)
        edge_index, edge_dists = atom_coords_to_edges(atom_pos)
        atom_graph = AtomGraph(node_pos=atom_pos,
                               res_map=atom_amino_id,
                               edge_index=edge_index,
                               edge_attr=edge_dists)
        atom_graph.features.add_named_oh_features('amino_types', amino_types)
        atom_graph.features.add_named_oh_features('atom_types', atom_types)
        atom_info = compute_radius_charge(pdb_path)
        atom_graph.features.add_named_features('charge_radius', atom_info)
        return atom_graph


if __name__ == "__main__":
    pass
    pdb = "../../data/example_files/4kt3.pdb"
    atomgraph_path = "../../data/example_files/4kt3_atomgraph.pt"
    atom_graph_builder = AtomGraphBuilder()
    atom_graph = atom_graph_builder.pdb_to_atomgraph(pdb)
    torch.save(atom_graph, atomgraph_path)
    atom_graph = torch.load(atomgraph_path)
    a = 1
