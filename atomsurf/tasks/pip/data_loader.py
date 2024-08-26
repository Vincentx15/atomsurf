import os
import sys

from atom3d.datasets import LMDBDataset
import math
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.tasks.pip.preprocess import get_subunits
from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader, AtomBatch, update_model_input_dim


class SurfaceLoaderPIP(SurfaceLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.data_dir = os.path.join(config.data_dir, mode, 'surfaces_0.1')


class GraphBuilderPIP(GraphLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.data_dir, mode, 'rgraph')
        self.esm_dir = os.path.join(config.data_dir, mode, 'esm')


class PIPDataset(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.systems = LMDBDataset(data_dir)
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble

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
        protein_pair = self.systems[idx]
        pos_pairs = protein_pair['atoms_neighbors']
        names, dfs = get_subunits(protein_pair['atoms_pairs'])
        pdbca1 = dfs[0][(dfs[0]['name'] == 'CA') & (dfs[0]['hetero'] == ' ') & (dfs[0]['resname'] != 'UNK')]
        pdbca2 = dfs[1][(dfs[1]['name'] == 'CA') & (dfs[1]['hetero'] == ' ') & (dfs[1]['resname'] != 'UNK')]
        pos_pairs_res = pos_pairs[
            (pos_pairs['residue0'].isin(pdbca1.residue)) & (pos_pairs['residue1'].isin(pdbca2.residue))]

        mapping_1 = {resindex: i for i, resindex in enumerate(pdbca1.residue.values)}
        mapping_2 = {resindex: i for i, resindex in enumerate(pdbca2.residue.values)}
        pos_as_array_1 = np.array([mapping_1[resi] for resi in pos_pairs_res['residue0']])
        pos_as_array_2 = np.array([mapping_2[resi] for resi in pos_pairs_res['residue1']])
        dense = np.zeros((len(pdbca1), len(pdbca2)))
        negs_1, negs_2 = np.where(dense == 0)
        pos_array = np.stack((pos_as_array_1, pos_as_array_2))
        neg_array = np.stack((negs_1, negs_2))
        num_pos = pos_as_array_1.shape[0]
        num_neg = negs_1.shape[0]
        num_pos_to_use, num_neg_to_use = self._num_to_use(num_pos, num_neg)
        pos_array_idx = np.random.choice(pos_array.shape[1], size=num_pos_to_use, replace=False)
        neg_array_idx = np.random.choice(neg_array.shape[1], size=num_neg_to_use, replace=False)
        pos_array_sampled = pos_array[:, pos_array_idx]
        neg_array_sampled = neg_array[:, neg_array_idx]
        pos_array_sampled = torch.from_numpy(pos_array_sampled)
        neg_array_sampled = torch.from_numpy(neg_array_sampled)

        idx_left = torch.cat((pos_array_sampled[0], neg_array_sampled[0]))
        idx_right = torch.cat((pos_array_sampled[1], neg_array_sampled[1]))
        labels = torch.cat((torch.ones(num_pos_to_use), torch.zeros(num_neg_to_use)))
        surface_1 = self.surface_loader.load(names[0])
        surface_2 = self.surface_loader.load(names[1])
        graph_1 = self.graph_loader.load(names[0])
        graph_2 = self.graph_loader.load(names[1])
        if surface_1 is None or surface_2 is None or graph_1 is None or graph_2 is None:
            return None
        if graph_1.node_len < 20 or graph_2.node_len < 20 or surface_1.n_verts < 20 or surface_2.n_verts < 20:
            return None
        if idx_left.dtype != torch.int64 and idx_right.dtype != torch.int64:
            return None
        if idx_left.max() >= len(graph_1.node_pos) or idx_right.max() >= len(graph_2.node_pos):
            print('idx error', names)
            return None
        item = Data(surface_1=surface_1, graph_1=graph_1, surface_2=surface_2, graph_2=graph_2, idx_left=idx_left,
                    idx_right=idx_right, label=labels, g1_len=graph_1.node_pos.shape[0],
                    g2_len=graph_2.node_pos.shape[0])
        return item


class PIPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        data_dir = cfg.data_dir
        self.systems = []
        self.surface_builders = []
        self.graph_builders = []
        self.cfg = cfg
        for mode in ['train', 'val', 'test']:
        # for mode in ['test'] * 3:
            self.systems.append(os.path.join(data_dir, mode))
            self.surface_builders.append(SurfaceLoaderPIP(self.cfg.cfg_surface, mode=mode))
            self.graph_builders.append(GraphBuilderPIP(self.cfg.cfg_graph, mode=mode))

        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        train_dataset_temp = PIPDataset(self.systems[0], self.surface_builders[0], self.graph_builders[0])
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        dataset = PIPDataset(self.systems[0], self.surface_builders[0], self.graph_builders[0])
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = PIPDataset(self.systems[1], self.surface_builders[1], self.graph_builders[1])
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = PIPDataset(self.systems[2], self.surface_builders[2], self.graph_builders[2],
                             max_pos_regions_per_ensemble=5)
        return DataLoader(dataset, shuffle=False, **self.loader_args)


if __name__ == '__main__':
    pass
