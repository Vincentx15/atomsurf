import os
import sys

import numpy as np
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.utils.data_utils import AtomBatch

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
        if self.config.use_whole_surfaces:
            pocket_name = pocket_name.split('_patch_')[0]
        surface = torch.load(os.path.join(self.data_dir, f"{pocket_name}.pt"))
        surface.expand_features(remove_feats=True)
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
        graph.expand_features(remove_feats=True)
        return graph


class MasifLigandDataset(Dataset):

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
        # pocket = self.systems_keys[idx]
        pocket = "1DW1_A_patch_0_HEM"
        lig_coord, lig_type = self.systems[pocket]
        lig_coord = torch.from_numpy(lig_coord)
        # pocket = f'{pdb_chains}_patch_{ix}_{lig_type}'
        surface = self.surface_builder.build(pocket)
        graph = self.graph_builder.build(pocket)
        # TODO GDF EXPAND
        item = Data(surface=surface, graph=graph, lig_coord=lig_coord, label=lig_type)
        return item


class MasifLigandDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.surface_builder = SurfaceBuilder(cfg.cfg_surface)
        self.graph_builder = GraphBuilder(cfg.cfg_graph)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
        splits_dir = os.path.join(masif_ligand_data_dir, 'raw_data_MasifLigand', 'splits')
        self.systems = []
        for split in ['train', 'val', 'test']:
            splits_path = os.path.join(splits_dir, f'{split}-list.txt')
            out_path = os.path.join(splits_dir, f'{split}.p')
            self.systems.append(get_systems_from_ligands(splits_path,
                                                         ligands_path=ligands_path,
                                                         out_path=out_path))
        self.cfg = cfg
        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'shuffle': self.cfg.loader.shuffle,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

    def train_dataloader(self):
        dataset = MasifLigandDataset(self.systems[0], self.surface_builder, self.graph_builder)
        return DataLoader(dataset, **self.loader_args)

    def val_dataloader(self):
        dataset = MasifLigandDataset(self.systems[1], self.surface_builder, self.graph_builder)
        return DataLoader(dataset, **self.loader_args)

    def test_dataloader(self):
        dataset = MasifLigandDataset(self.systems[2], self.surface_builder, self.graph_builder)
        return DataLoader(dataset, **self.loader_args)


if __name__ == '__main__':
    pass
    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
    splits_dir = os.path.join(masif_ligand_data_dir, 'raw_data_MasifLigand', 'splits')
    ligands_path = os.path.join(masif_ligand_data_dir, 'raw_data_MasifLigand', 'ligand')
    # for split in ['train', 'val', 'test']:
    #     splits_path = os.path.join(splits_dir, f'{split}-list.txt')
    #     out_path = os.path.join(splits_dir, f'{split}.p')
    #     systems = get_systems_from_ligands(splits_path,
    #                                        ligands_path=ligands_path,
    #                                        out_path=out_path,
    #                                        recompute=True)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_ligand_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_ligand')
    cfg_surface = Data()
    cfg_surface.use_surfaces = True
    # cfg_surface.use_whole_surfaces = False
    # cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, 'surf_hmr')
    # cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, 'surf_ours')
    cfg_surface.use_whole_surfaces = True
    cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, 'surf_full')
    surface_builder = SurfaceBuilder(cfg_surface)

    cfg_graph = Data()
    cfg_graph.use_graphs = True
    cfg_graph.data_dir = os.path.join(masif_ligand_data_dir, 'rgraph')
    # cfg_graph.data_dir= os.path.join(masif_ligand_data_dir, 'agraph')
    graph_builder = GraphBuilder(cfg_graph)

    split = 'train'
    splits_path = os.path.join(splits_dir, f'{split}-list.txt')
    out_path = os.path.join(splits_dir, f'{split}.p')
    systems = get_systems_from_ligands(splits_path,
                                       ligands_path=ligands_path,
                                       out_path=out_path)
    dataset = MasifLigandDataset(systems, surface_builder, graph_builder)
    a = dataset[0]

    loader_cfg = Data(num_workers=0, batch_size=4, pin_memory=False, prefetch_factor=2, shuffle=False)
    simili_cfg = Data(cfg_surface=cfg_surface, cfg_graph=cfg_graph, loader=loader_cfg)
    datamodule = MasifLigandDataModule(cfg=simili_cfg)
    loader = datamodule.train_dataloader()
    for i, batch in enumerate(loader):
        if i > 3:
            break
