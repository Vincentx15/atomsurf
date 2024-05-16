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

from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader, AtomBatch, update_model_input_dim


class MasifSiteDataset(Dataset):
    def __init__(self, systems, surface_builder, graph_builder):
        self.systems = systems
        self.surface_builder = surface_builder
        self.graph_builder = graph_builder

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        pocket = self.systems[idx]
        surface = self.surface_builder.load(pocket)
        graph = self.graph_builder.load(pocket)
        if surface is None or graph is None:
            return None
        item = Data(surface=surface, graph=graph, label=surface.iface_labels)
        return item


class MasifSiteDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.surface_loader = SurfaceLoader(cfg.cfg_surface)
        self.graph_loader = GraphLoader(cfg.cfg_graph)

        # Get the right systems
        script_dir = os.path.dirname(os.path.realpath(__file__))
        masif_site_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')
        train_systems_list = os.path.join(masif_site_data_dir, 'train_list.txt')
        trainval_sys = [name.strip() for name in open(train_systems_list, 'r').readlines()]
        np.random.shuffle(trainval_sys)
        trainval_cut = int(0.9 * len(trainval_sys))
        self.train_sys = trainval_sys[:trainval_cut]
        self.val_sys = trainval_sys[trainval_cut:]
        test_systems_list = os.path.join(masif_site_data_dir, 'test_list.txt')
        self.test_sys = [name.strip() for name in open(test_systems_list, 'r').readlines()]

        self.cfg = cfg
        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'shuffle': self.cfg.loader.shuffle,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        dataset_temp = MasifSiteDataset(self.train_sys, self.surface_loader, self.graph_loader)
        update_model_input_dim(cfg, dataset_temp=dataset_temp)

    def train_dataloader(self):
        dataset = MasifSiteDataset(self.train_sys, self.surface_loader, self.graph_loader)
        return DataLoader(dataset, **self.loader_args)

    def val_dataloader(self):
        dataset = MasifSiteDataset(self.val_sys, self.surface_loader, self.graph_loader)
        return DataLoader(dataset, **self.loader_args)

    def test_dataloader(self):
        dataset = MasifSiteDataset(self.test_sys, self.surface_loader, self.graph_loader)
        return DataLoader(dataset, **self.loader_args)


if __name__ == '__main__':
    pass
    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_site_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')

    # SURFACE
    cfg_surface = Data()
    cfg_surface.use_surfaces = True
    cfg_surface.feat_keys = 'all'
    cfg_surface.oh_keys = 'all'
    cfg_surface.data_dir = os.path.join(masif_site_data_dir, 'surfaces')
    surface_loader = SurfaceLoader(cfg_surface)

    # GRAPHS
    cfg_graph = Data()
    cfg_graph.use_graphs = False
    cfg_graph.feat_keys = 'all'
    cfg_graph.oh_keys = 'all'
    cfg_graph.esm_dir = 'toto'
    cfg_graph.use_esm = False
    cfg_graph.data_dir = os.path.join(masif_site_data_dir, 'rgraph')
    # cfg_graph.data_dir= os.path.join(masif_site_data_dir, 'agraph')
    graph_loader = GraphLoader(cfg_graph)

    test_systems_list = os.path.join(masif_site_data_dir, 'test_list.txt')
    test_sys = [name.strip() for name in open(test_systems_list, 'r').readlines()]
    dataset = MasifSiteDataset(test_sys, surface_loader, graph_loader)
    a = dataset[0]

    loader_cfg = Data(num_workers=0, batch_size=4, pin_memory=False, prefetch_factor=2, shuffle=False)
    simili_cfg = Data(cfg_surface=cfg_surface, cfg_graph=cfg_graph, loader=loader_cfg)
    datamodule = MasifSiteDataModule(cfg=simili_cfg)
    loader = datamodule.train_dataloader()
    for i, batch in enumerate(loader):
        if i > 3:
            break
