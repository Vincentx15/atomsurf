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


class AADataset(Dataset):
    def __init__(self, csv_file,data_dir, surface_builder, graph_builder, neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.systems = pd.read_csv(csv_file)
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble
        self.data_dir = data_dir
    def __len__(self):
        return len(self.systems)

    def _num_to_use(self, num_pos, num_neg):
        """
        Depending on the number of pos and neg of the system, we might want to use
            different amounts of positive or negative coordinates.

        :param num_pos:
        :param num_neg:
        :return:
        """

        if self.neg_to_pos_ratio == -1:
            num_pos_to_use, num_neg_to_use = num_pos, num_neg
        else:
            num_pos_to_use = min(num_pos, num_neg / self.neg_to_pos_ratio)
            if self.max_pos_regions_per_ensemble != -1:
                num_pos_to_use = min(num_pos_to_use, self.max_pos_regions_per_ensemble)
            num_neg_to_use = num_pos_to_use * self.neg_to_pos_ratio
        num_pos_to_use = int(math.ceil(num_pos_to_use))
        num_neg_to_use = int(math.ceil(num_neg_to_use))
        return num_pos_to_use, num_neg_to_use

    def __getitem__(self, idx):
        name= self.systems.iloc[idx]['pdb']
        if os.path.exists(os.path.join(self.data_dir,'pdbs',name+'.json')):
            interact_idx= json.load(open(os.path.join(self.data_dir,'pdbs',name+'.json'),'r'))
        else:
            return None
        pos_pairs = interact_idx['cdr_ag_pair']

        surface_1 = self.surface_loader.load(name+'_ab')
        surface_2 = self.surface_loader.load(name+'_ag')
        graph_1 = self.graph_loader.load(name+'_ab')
        graph_2 =  self.graph_loader.load(name+'_ag')
        if surface_1 is None or surface_2 is None or graph_1 is None or graph_2 is None:
            return None
        if graph_1.node_len < 10 or graph_2.node_len < 10 or surface_1.n_verts < 20 or surface_2.n_verts < 20:
            return None
        dense = np.zeros((len(graph_1.node_pos), len(graph_2.node_pos)))
        pos_pairs=np.array(pos_pairs)
        pos_1,pos_2 = pos_pairs[:,0],pos_pairs[:,1]
        dense[interact_idx['cdr']]=1
        dense[pos_1,pos_2]= 2
        negs_1, negs_2=  np.where(dense == 1)
        num_pos = pos_1.shape[0]
        num_neg = negs_1.shape[0]
        num_pos_to_use, num_neg_to_use = self._num_to_use(num_pos, num_neg)
        pos_array = np.stack((pos_1, pos_2))
        neg_array = np.stack((negs_1, negs_2))
        pos_array_idx = np.random.choice(pos_array.shape[1], size=num_pos_to_use, replace=False)
        neg_array_idx = np.random.choice(neg_array.shape[1], size=num_neg_to_use, replace=False)
        pos_array_sampled = pos_array[:, pos_array_idx]
        neg_array_sampled = neg_array[:, neg_array_idx]
        pos_array_sampled = torch.from_numpy(pos_array_sampled)
        neg_array_sampled = torch.from_numpy(neg_array_sampled)
        idx_left = torch.cat((pos_array_sampled[0], neg_array_sampled[0]))
        idx_right = torch.cat((pos_array_sampled[1], neg_array_sampled[1]))
        labels = torch.cat((torch.ones(num_pos_to_use), torch.zeros(num_neg_to_use)))

        if idx_left.dtype != torch.int64 and idx_right.dtype != torch.int64:
            return None
        if idx_left.max() >= len(graph_1.node_pos) or idx_right.max() >= len(graph_2.node_pos):
            print('idx error', name)
            return None
        item = Data(surface_1=surface_1, graph_1=graph_1, surface_2=surface_2, graph_2=graph_2, idx_left=idx_left,
                    idx_right=idx_right, label=labels, g1_len=graph_1.node_pos.shape[0],
                    g2_len=graph_2.node_pos.shape[0])
        return item
class AADataModule(pl.LightningDataModule):
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
        train_dataset_temp = AADataset(self.systems[0],self.data_dir, self.surface_loaders, self.graph_loaders)
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        dataset = AADataset(self.systems[0],self.data_dir, self.surface_loaders, self.graph_loaders)
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = AADataset(self.systems[1],self.data_dir, self.surface_loaders, self.graph_loaders)
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = AADataset(self.systems[2],self.data_dir, self.surface_loaders, self.graph_loaders,
                             max_pos_regions_per_ensemble=5)
        return DataLoader(dataset, shuffle=False, **self.loader_args)