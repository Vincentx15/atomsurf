
import os
import sys

# from atom3d.datasets import LMDBDataset
import math
import numpy as np
import pickle
import pytorch_lightning as pl
import torch
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
        self.data_dir = os.path.join(config.data_dir, mode, 'surfaces_0.1')


class GraphLoaderPSR(GraphLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.data_dir, mode, 'rgraph')
        self.esm_dir = os.path.join(config.data_dir, mode, 'esm')

class PSRDataset(Dataset):

    def __init__(self, data_dir, surface_builder, graph_builder):
        sysdict = json.load(open(data_dir,'r'))
        self.systems = [(k, v) for k, v in sysdict.items()]
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        name , score = self.systems[idx]
        surface = self.surface_loader.load(name)
        graph = self.graph_loader.load(name)
        if surface is None or graph is None:
            return None
        if graph.node_pos.shape[0]<20 or surface.verts.shape[0]<20:
            return None

        item = Data(name= name,surface = surface , graph = graph , score = score)

        return item


class PSRDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = cfg.data_dir
        self.systems = []
        for mode in ['train', 'val', 'test']:
            self.systems.append(os.path.join(data_dir, mode,mode+'_score.json'))
        self.cfg = cfg
        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}
        self.surface_builder_train = SurfaceLoaderPSR(self.cfg.cfg_surface, mode='train')
        self.graph_builder_train = GraphLoaderPSR(self.cfg.cfg_graph, mode='train')
        self.surface_builder_test = SurfaceLoaderPSR(self.cfg.cfg_surface, mode='test')
        self.graph_builder_test = GraphLoaderPSR(self.cfg.cfg_graph, mode='test')
        self.surface_builder_val = SurfaceLoaderPSR(self.cfg.cfg_surface, mode='val')
        self.graph_builder_val = GraphLoaderPSR(self.cfg.cfg_graph, mode='val')
        # Useful to create a Model of the right input dims
        train_dataset_temp = PSRDataset(self.systems[0], self.surface_builder_train, self.graph_builder_train)
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp,gkey='graph',skey='surface')

    def train_dataloader(self):
        dataset = PSRDataset(self.systems[0], self.surface_builder_train, self.graph_builder_train)
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = PSRDataset(self.systems[1], self.surface_builder_val, self.graph_builder_val)
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = PSRDataset(self.systems[2], self.surface_builder_test, self.graph_builder_test)
        return DataLoader(dataset, shuffle=False, **self.loader_args)


if __name__ == '__main__':
    pass
