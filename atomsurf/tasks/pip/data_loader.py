import os
import sys

import numpy as np
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import math

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.utils.data_utils import AtomBatch

from atom3d.datasets import LMDBDataset
from atomsurf.tasks.pip.preprocess import get_subunits


class SurfaceBuilder:
    def __init__(self, config,mode):
        self.config = config
        self.data_dir = config.data_dir
        self.mode= mode
    def build(self, name):
        if not self.config.use_surfaces:
            return Data()
        try:
            surface = torch.load(os.path.join(self.data_dir,self.mode,'surf_full', f"{name}.pt"))
            surface.expand_features(remove_feats=True, feature_keys=self.config.feat_keys, oh_keys=self.config.oh_keys)
            return surface
        except:
            return None


class GraphBuilder:
    def __init__(self, config,mode):
        self.config = config
        self.data_dir = config.data_dir
        self.esm_dir = config.esm_dir
        self.use_esm = config.use_esm
        self.mode= mode
    def build(self, name):
        if not self.config.use_graphs:
            return Data()
        try:
            graph = torch.load(os.path.join(self.data_dir,self.mode,'rgraph', f"{name}.pt"))
            feature_keys = self.config.feat_keys
            if self.use_esm:
                esm_feats_path = os.path.join(self.esm_dir, f"{name}_esm.pt")
                esm_feats = torch.load(esm_feats_path)
                graph.features.add_named_features('esm_feats', esm_feats)
                if feature_keys != 'all':
                    feature_keys.append('esm_feats')
            graph.expand_features(remove_feats=True, feature_keys=feature_keys, oh_keys=self.config.oh_keys)
        except:
            return None
        return graph


class PIPDataset(Dataset):

    def __init__(self, data_dir, surface_builder, graph_builder, neg_to_pos_ratio=1, max_pos_regions_per_ensemble=5):
        self.systems = LMDBDataset(data_dir)
        self.surface_builder = surface_builder
        self.graph_builder = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble=max_pos_regions_per_ensemble
    def __len__(self):
        return len(self.systems)
    
    def _num_to_use(self,num_pos, num_neg):
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
        protein_pair= self.systems[idx]
        pos_pairs = protein_pair['atoms_neighbors']
        names,dfs=get_subunits(protein_pair['atoms_pairs'])
        pdbca1=dfs[0][(dfs[0]['name']=='CA') & (dfs[0]['hetero']==' ')& (dfs[0]['resname']!='UNK')]
        pdbca2=dfs[1][(dfs[1]['name']=='CA') & (dfs[1]['hetero']==' ')& (dfs[1]['resname']!='UNK')]
        pos_pairs_res=pos_pairs[(pos_pairs['residue0'].isin(pdbca1.residue))&(pos_pairs['residue1'].isin(pdbca2.residue))]

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

        idx_left = torch.cat((pos_array_sampled[0],neg_array_sampled[0]))
        idx_right = torch.cat((pos_array_sampled[1],neg_array_sampled[1]))
        
        labels = torch.cat((torch.ones(num_pos_to_use), torch.zeros(num_neg_to_use)))

        surface_1 = self.surface_builder.build(names[0])
        surface_2 = self.surface_builder.build(names[1])
        graph_1 = self.graph_builder.build(names[0])
        graph_2 = self.graph_builder.build(names[1])

        if surface_1 is None or surface_2 is None or graph_1 is None or graph_2 is None:
            return None
        # TODO GDF EXPAND
        locs_left= graph_1.node_pos[idx_left]
        locs_right= graph_2.node_pos[idx_right]
        #TODO MISS transform and normalize
        # item = Data(surface_1=surface_1, graph_1=graph_1,surface_2=surface_2, graph_2=graph_2, idx_left=idx_left,idx_right=idx_right, label=labels)
        item = Data(surface_1=surface_1, graph_1=graph_1,surface_2=surface_2, graph_2=graph_2, locs_left=locs_left,locs_right=locs_right, label=labels)
        return item


class PIPDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir=cfg.data_dir
        self.systems = []
        for mode in ['train', 'val', 'test']:
            self.systems.append(os.path.join(data_dir,mode))
        self.cfg = cfg
        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'shuffle': self.cfg.loader.shuffle,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}
        self.surface_builder_train = SurfaceBuilder(self.cfg.cfg_surface,mode='train')
        self.graph_builder_train = GraphBuilder(self.cfg.cfg_graph,mode='train')
        self.surface_builder_test = SurfaceBuilder(self.cfg.cfg_surface,mode='test')
        self.graph_builder_test = GraphBuilder(self.cfg.cfg_graph,mode='test')        
        self.surface_builder_val = SurfaceBuilder(self.cfg.cfg_surface,mode='val')
        self.graph_builder_val = GraphBuilder(self.cfg.cfg_graph,mode='val')
        # Useful to create a Model of the right input dims
        train_dataset_temp = PIPDataset(self.systems[0], self.surface_builder_train, self.graph_builder_train)
        train_dataset_temp = iter(train_dataset_temp)
        exemple = None
        while exemple is None:
            exemple = next(train_dataset_temp)
        from omegaconf import open_dict
        with open_dict(cfg):
            feat_encoder_kwargs = cfg.encoder.blocks[0].kwargs
            feat_encoder_kwargs['graph_feat_dim'] = exemple.graph_1.x.shape[1]
            feat_encoder_kwargs['surface_feat_dim'] = exemple.surface_1.x.shape[1]

    def train_dataloader(self):
        dataset = PIPDataset(self.systems[0], self.surface_builder_train, self.graph_builder_train)
        return DataLoader(dataset, **self.loader_args)

    def val_dataloader(self):
        dataset = PIPDataset(self.systems[1], self.surface_builder_val, self.graph_builder_val)
        return DataLoader(dataset, **self.loader_args)

    def test_dataloader(self):
        dataset = PIPDataset(self.systems[2], self.surface_builder_test, self.graph_builder_test)
        return DataLoader(dataset, **self.loader_args)


if __name__ == '__main__':
    pass
