import os
import re
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.protein.residue_graph import ResidueGraph
from atomsurf.protein.atom_graph import AtomGraph


class SurfaceLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir

    def load(self, pocket_name):
        if not self.config.use_surfaces:
            return Data()
        try:
            surface = torch.load(os.path.join(self.data_dir, f"{pocket_name}.pt"))
            surface.expand_features(remove_feats=True, feature_keys=self.config.feat_keys, oh_keys=self.config.oh_keys)
            return surface
        except Exception as e:
            return None


class GraphLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.esm_dir = config.esm_dir
        self.use_esm = config.use_esm

    def load(self, pocket_name):
        if not self.config.use_graphs:
            return Data()
        try:
            graph = torch.load(os.path.join(self.data_dir, f"{pocket_name}.pt"))
            feature_keys = self.config.feat_keys
            if self.use_esm:
                esm_feats_path = os.path.join(self.esm_dir, f"{pocket_name}_esm.pt")
                esm_feats = torch.load(esm_feats_path)
                graph.features.add_named_features('esm_feats', esm_feats)
                if feature_keys != 'all':
                    feature_keys.append('esm_feats')
            graph.expand_features(remove_feats=True, feature_keys=feature_keys, oh_keys=self.config.oh_keys)
        except Exception as e:
            return None
        return graph


def update_model_input_dim(cfg, dataset_temp):
    # Useful to create a Model of the right input dims
    try:
        from omegaconf import open_dict
        for example in dataset_temp:
            if example is not None:
                with open_dict(cfg):
                    feat_encoder_kwargs = cfg.encoder.blocks[0].kwargs
                    feat_encoder_kwargs['graph_feat_dim'] = example.graph.x.shape[1]
                    feat_encoder_kwargs['surface_feat_dim'] = example.surface.x.shape[1]
                break
    except Exception as e:
        print('Could not update model input dims because of error: ', e)


class AtomBatch(Data):
    def __init__(self, batch=None, **kwargs):
        super().__init__(**kwargs)
        self.batch = batch
        self.__data_class__ = Data

    @staticmethod
    def from_data_list(data_list):
        # Filter out None
        data_list = [x for x in data_list if x is not None]
        if len(data_list) == 0:
            return None
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = AtomBatch()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
            item = batch[key][0]
            if isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])
            elif bool(re.search('(locs_left|locs_right|neg_stack|pos_stack)', key)):
                batch[key] = batch[key]
            elif key == 'labels_pip':
                batch[key] = torch.cat(batch[key])
            elif torch.is_tensor(item):
                try:
                    # If they are all the same size
                    batch[key] = torch.stack(batch[key])
                except:
                    batch[key] = batch[key]
            elif isinstance(item, SurfaceObject):
                batch[key] = SurfaceObject.batch_from_data_list(batch[key])
            elif isinstance(item, ResidueGraph):
                batch[key] = ResidueGraph.batch_from_data_list(batch[key])
            elif isinstance(item, AtomGraph):
                batch[key] = AtomGraph.batch_from_data_list(batch[key])
            elif isinstance(item, Data):
                batch[key] = Batch.from_data_list(batch[key])
                batch[key] = batch[key] if batch[key].num_graphs > 0 else None
            elif isinstance(item, list):
                batch[key] = batch[key]
            elif isinstance(item, str):
                batch[key] = batch[key]
            elif isinstance(item, SparseTensor):
                batch[key] = batch[key]
            else:
                raise ValueError(f"Unsupported attribute type: {type(item)}, item : {item}, key : {key}")

        batch = batch.contiguous()
        return batch

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
