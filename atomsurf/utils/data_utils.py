import re
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.protein.residue_graph import ResidueGraph
from atomsurf.protein.atom_graph import AtomGraph


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
