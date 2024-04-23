import os
import sys

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

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
            pocket = f'{pdb_chains}_patch_{ix}_{lig_type}'
            all_pockets[pocket] = np.reshape(lig_coord, (-1, 3)), type_idx[lig_type]
            a = 1
    if out_path is not None:
        pickle.dump(all_pockets, open(out_path, "wb"))
    return all_pockets


class SurfaceBuilder:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir

    def build(self, pocket_name):
        if not self.config.use_surfaces:
            return Data()
        if self.config.use_whole_graphs:
            pocket_name = pocket_name.split('_patch_')[0]
        surface = torch.load(os.path.join(self.data_dir, f"{pocket_name}.pt"))
        return surface


class GraphBuilder:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir

    def build(self, pocket_name):
        if not self.config.use_graphs:
            return Data()
        pocket_name = pocket_name.split('_patch_')[0]
        graph = torch.load(os.path.join(self.data_dir, f"{pocket_name}.pt"))
        return graph


class DatasetMasifLigand(Dataset):

    def __init__(self, systems, surface_builder, graph_builder):
        self.systems = systems
        self.systems_keys = list(systems.keys())
        self.surface_builder = surface_builder
        self.graph_builder = graph_builder

    # @staticmethod
    # def collate_wrapper(unbatched_list):
    #     unbatched_list = [elt for elt in unbatched_list if elt is not None]
    #     return AtomBatch.from_data_list(unbatched_list)

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        pocket = self.systems_keys[idx]
        lig_coord, lig_type = self.systems[pocket]
        lig_coord = torch.from_numpy(lig_coord)
        # pocket = f'{pdb_chains}_patch_{ix}_{lig_type}'
        surface = self.surface_builder.build(pocket)
        graph = self.graph_builder.build(pocket)
        item = Data(surface=surface, graph=graph, lig_coord=lig_coord, label=lig_type)
        return item


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
                                                 recompute=True)
    # data dir
    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
    cfg_surface = Data()
    cfg_surface.use_surfaces = False
    cfg_surface.use_whole_graphs = True
    # cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, 'surf_hmr')
    # cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, 'surf_ours')
    cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, 'surf_full')
    surface_builder = SurfaceBuilder(cfg_surface)

    cfg_graph = Data()
    cfg_graph.use_graphs = True
    cfg_graph.data_dir = os.path.join(masif_ligand_data_dir, 'rgraph')
    # cfg_graph.data_dir= os.path.join(masif_ligand_data_dir, 'agraph')

    graph_builder = GraphBuilder(cfg_graph)
    dataset = DatasetMasifLigand(train_systems, surface_builder, graph_builder)
    a = dataset[0]
    a = 1
