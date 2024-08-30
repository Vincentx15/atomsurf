import os
import sys

from atom3d.datasets import LMDBDataset
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import json

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader, AtomBatch, update_model_input_dim


class SurfaceLoaderMSP(SurfaceLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.data_dir = os.path.join(config.data_dir, mode, config.data_name)


class GraphLoaderMSP(GraphLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.data_dir, mode, config.data_name)
        self.esm_dir = os.path.join(config.data_dir, mode, 'esm')


class MSPDataset(Dataset):
    def __init__(self, data_dir, surface_loader, graph_loader, verbose=False):
        self.systems = LMDBDataset(data_dir)
        self.surface_loader = surface_loader
        self.graph_loader = graph_loader
        self.verbose = verbose

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        lmdb_item = self.systems[idx]
        system_name = lmdb_item['id']
        pdb, chains_left, chains_right, mutation = system_name.split('_')
        names = [f"{pdb}_{chains_left}", f"{pdb}_{chains_right}",
                 f"{pdb}_{chains_left}_{mutation}", f"{pdb}_{chains_right}_{mutation}"]
        all_surfs, all_graphs, all_ids = [], [], []
        for name in names:
            surface_name = os.path.join(system_name, name)
            surface = self.surface_loader.load(surface_name)
            graph_name = os.path.join(system_name, name)
            graph = self.graph_loader.load(graph_name)
            if surface is None or surface.n_verts < 128:
                if self.verbose:
                    print('Surface problem', surface_name)
                return None
            if graph is None or graph.node_len < 2:
                if self.verbose:
                    print('Graph problem', graph_name)
                return None
            try:
                all_ids.append(graph.misc_features['interface_node'])
            except KeyError:
                if self.verbose:
                    print('missing interface nodes for', graph_name)
                return None
            all_surfs.append(surface)
            all_graphs.append(graph)
        item = Data(name=system_name, label=torch.tensor([float(lmdb_item['label'])]))
        item.surface_lo = all_surfs[0]
        item.surface_ro = all_surfs[1]
        item.surface_lm = all_surfs[2]
        item.surface_rm = all_surfs[3]
        item.graph_lo = all_graphs[0]
        item.graph_ro = all_graphs[1]
        item.graph_lm = all_graphs[2]
        item.graph_rm = all_graphs[3]
        item.all_ids = all_ids
        return item


class MSPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.surface_loaders = []
        self.graph_loaders = []
        self.lmdb_paths = []
        for mode in ['train', 'val', 'test']:
            self.lmdb_paths.append(os.path.join(cfg.data_dir, mode))
            self.surface_loaders.append(SurfaceLoaderMSP(cfg.cfg_surface, mode=mode))
            self.graph_loaders.append(GraphLoaderMSP(cfg.cfg_graph, mode=mode))
        self.cfg = cfg
        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        train_dataset_temp = MSPDataset(self.lmdb_paths[0], self.surface_loaders[0], self.graph_loaders[0])
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_lo', skey='surface_lo')

    def train_dataloader(self):
        dataset = MSPDataset(self.lmdb_paths[0], self.surface_loaders[0], self.graph_loaders[0])
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = MSPDataset(self.lmdb_paths[1], self.surface_loaders[1], self.graph_loaders[1])
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = MSPDataset(self.lmdb_paths[2], self.surface_loaders[2], self.graph_loaders[2])
        return DataLoader(dataset, shuffle=False, **self.loader_args)


if __name__ == '__main__':
    pass
