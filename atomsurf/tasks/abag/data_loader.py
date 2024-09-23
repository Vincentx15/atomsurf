import os
import sys

from atom3d.datasets import LMDBDataset
import math
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import pandas as pd
import json

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader, AtomBatch, update_model_input_dim


class AbAgDataset(Dataset):
    def __init__(self, csv_file, data_dir, surface_builder, graph_builder):
        self.systems = pd.read_csv(csv_file)
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.data_dir = data_dir

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        row = self.systems.iloc[idx]
        pdb, chain_H, chain_L, chain_ag = row['pdb'], row['Hchain'], row['Lchain'], row['antigen_chain']
        system_id = f"{pdb}_{chain_H}{chain_L}_{''.join(chain_ag)}"
        if os.path.exists(os.path.join(self.data_dir, 'pdb', system_id + '.json')):
            interact_idx = json.load(open(os.path.join(self.data_dir, 'pdb', system_id + '.json'), 'r'))
        else:
            return None

        surface_ab = self.surface_loader.load(system_id + '_ab')
        surface_ag = self.surface_loader.load(system_id + '_ag')
        graph_ab = self.graph_loader.load(system_id + '_ab')
        graph_ag = self.graph_loader.load(system_id + '_ag')
        if surface_ab is None or surface_ag is None or graph_ab is None or graph_ag is None:
            return None
        if graph_ab.node_len < 10 or graph_ag.node_len < 10 or surface_ab.n_verts < 20 or surface_ag.n_verts < 20:
            return None

        # extract positive abs
        positive_abs_global = torch.zeros(graph_ab.num_nodes)
        positive_abs_global[interact_idx['cdr_contact']] = 1
        positive_abs_cdr = positive_abs_global[interact_idx['cdr']]
        item = Data(surface_ab=surface_ab, graph_ab=graph_ab,
                    surface_ag=surface_ag, graph_ag=graph_ag,
                    cdr=interact_idx['cdr'],
                    positive_abs_cdr=positive_abs_cdr,
                    positive_ag=interact_idx['ag_contact'],
                    g1_len=graph_ab.node_pos.shape[0],
                    g2_len=graph_ag.node_pos.shape[0])
        return item


class AbAgDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_dir = cfg.data_dir
        self.systems = []
        self.surface_loaders = SurfaceLoader(self.cfg.cfg_surface)
        self.graph_loaders = GraphLoader(self.cfg.cfg_graph)

        for mode in ['train.csv', 'val_pecan_aligned.csv', 'test70.csv']:
            # for mode in ['test'] * 3:
            self.systems.append(os.path.join(self.data_dir, mode))

        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        train_dataset_temp = AbAgDataset(self.systems[0], self.data_dir, self.surface_loaders, self.graph_loaders)
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_ab', skey='surface_ab')

    def train_dataloader(self):
        dataset = AbAgDataset(self.systems[0], self.data_dir, self.surface_loaders, self.graph_loaders)
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = AbAgDataset(self.systems[1], self.data_dir, self.surface_loaders, self.graph_loaders)
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = AbAgDataset(self.systems[2], self.data_dir, self.surface_loaders, self.graph_loaders)
        return DataLoader(dataset, shuffle=False, **self.loader_args)
