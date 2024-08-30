import os
import sys

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import json

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader, AtomBatch, update_model_input_dim


class SurfaceLoaderPSR(SurfaceLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.data_dir = os.path.join(config.data_dir, mode, config.data_name)


class GraphLoaderPSR(GraphLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.data_dir, mode, config.data_name)
        self.esm_dir = os.path.join(config.data_dir, mode, 'esm')


class PSRDataset(Dataset):
    def __init__(self, data_dir, surface_loader, graph_loader):
        sysdict = json.load(open(data_dir, 'r'))
        self.systems = [(k, v) for k, v in sysdict.items()]
        self.surface_loader = surface_loader
        self.graph_loader = graph_loader

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        name, name_score = self.systems[idx]
        _, score = name_score.values()
        surface = self.surface_loader.load(name)
        graph = self.graph_loader.load(name)
        if surface is None or graph is None:
            return None
        if graph.node_pos.shape[0] < 20 or surface.verts.shape[0] < 20:
            return None
        item = Data(name=name, surface=surface, graph=graph, score=score)
        return item


class PSRDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.surface_loaders = []
        self.graph_loaders = []
        self.systems = []
        for mode in ['train', 'val', 'test']:
        # for mode in ['test'] * 3:
            self.systems.append(os.path.join(cfg.data_dir, mode, mode + '_score.json'))
            self.surface_loaders.append(SurfaceLoaderPSR(cfg.cfg_surface, mode=mode))
            self.graph_loaders.append(GraphLoaderPSR(cfg.cfg_graph, mode=mode))
        self.cfg = cfg
        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        train_dataset_temp = PSRDataset(self.systems[0], self.surface_loaders[0], self.graph_loaders[0])
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph', skey='surface')

    def train_dataloader(self):
        dataset = PSRDataset(self.systems[0], self.surface_loaders[0], self.graph_loaders[0])
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = PSRDataset(self.systems[1], self.surface_loaders[1], self.graph_loaders[1])
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = PSRDataset(self.systems[2], self.surface_loaders[2], self.graph_loaders[2])
        return DataLoader(dataset, shuffle=False, **self.loader_args)


if __name__ == '__main__':
    pass
