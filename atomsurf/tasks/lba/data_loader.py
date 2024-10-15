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


class SurfaceLoaderLBA(SurfaceLoader):
    def __init__(self, config):
        super().__init__(config)
        self.data_dir = os.path.join(config.data_dir, 'surfaces_0.1')


class GraphLoaderLBA(GraphLoader):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.data_dir, 'rgraph')
        self.esm_dir = os.path.join(config.data_dir, 'esm')


class LBADataset(Dataset):
    def __init__(self, data_dir,mode,surface_builder, graph_builder, neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.systems = pickle.load(open(os.path.join(data_dir,mode+'.pkl'),'rb'))
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble
        self.data_dir = data_dir
        self.mfedatadir= os.path.join(data_dir,'MFEdata')
    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        name = self.systems[idx]
        surface = self.surface_loader.load(name)
        graph = self.graph_loader.load(name)
        MFEdata = pickle.load(open(os.path.join(self.mfedatadir,name+'.pkl'),'rb'))
        if surface is None or graph is None :
            return None
        if torch.isnan(surface.x).any() or torch.isnan(graph.x).any() or torch.isnan(MFEdata[0].y).any():
            return None
        item = Data(surface=surface, graph=graph, g_ligand=MFEdata[0],g_inter=MFEdata[1],id=MFEdata[2],ligand_center=MFEdata[3])
        return item

class LBADataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.surface_loader=SurfaceLoaderLBA(self.cfg.cfg_surface)
        self.graph_loader= GraphLoaderLBA(self.cfg.cfg_graph)
        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        train_dataset_temp =  LBADataset(data_dir=self.data_dir,mode='train',surface_builder= self.surface_loader, graph_builder=self.graph_loader)
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph', skey='surface')

    def train_dataloader(self):
        dataset = LBADataset(data_dir=self.data_dir,mode='train',surface_builder= self.surface_loader, graph_builder=self.graph_loader)
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = LBADataset(data_dir=self.data_dir,mode='val',surface_builder= self.surface_loader, graph_builder=self.graph_loader)
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = LBADataset(data_dir=self.data_dir,mode='test',surface_builder= self.surface_loader, graph_builder=self.graph_loader)
        return DataLoader(dataset, shuffle=False, **self.loader_args)


if __name__ == '__main__':
    pass