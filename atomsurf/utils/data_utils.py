import os
import re

from omegaconf import open_dict
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from atomsurf.protein.surfaces import SurfaceObject, SurfaceBatch
from atomsurf.protein.graphs import parse_pdb_path
from atomsurf.protein.atom_graph import AtomGraph, AGraphBatch, AtomGraphBuilder
from atomsurf.protein.residue_graph import ResidueGraph, RGraphBatch, ResidueGraphBuilder


class GaussianDistance(object):
    def __init__(self, start, stop, num_centers):
        self.filters = torch.linspace(start, stop, num_centers)
        self.var = (stop - start) / (num_centers - 1)

    def __call__(self, d):
        """
        :param d: shape (n,1)
        :return: shape (n,len(filters))
        """
        return torch.exp(-(((d - self.filters) / self.var) ** 2))


class SurfaceLoader:
    """
    This class is used to go load surfaces saved in a .pt file
    Based on the config it's given, it will load the corresponding features, optionnally expand them and
    populate a .x key with them.
    """

    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.feature_expander = None
        if "gdf_expand" in config and config.gdf_expand:
            self.gauss_curv_gdf = GaussianDistance(start=-0.1, stop=0.1, num_centers=16)
            self.mean_curv_gdf = GaussianDistance(start=-0.5, stop=0.5, num_centers=16)
            self.feature_expander = {'geom_feats': self.gdf_expand}

    def gdf_expand(self, geom_features):
        gauss_curvs, mean_curvs, others = torch.split(geom_features, [1, 1, 20], dim=-1)
        gauss_curvs_gdf = self.gauss_curv_gdf(gauss_curvs)
        mean_curvs_gdf = self.mean_curv_gdf(mean_curvs)
        return torch.cat([gauss_curvs_gdf, mean_curvs_gdf, others], dim=-1)

    def load(self, surface_name):
        if not self.config.use_surfaces:
            return Data()
        try:
            surface = torch.load(os.path.join(self.data_dir, f"{surface_name}.pt"))
            with torch.no_grad():
                surface.expand_features(remove_feats=True,
                                        feature_keys=self.config.feat_keys,
                                        oh_keys=self.config.oh_keys,
                                        feature_expander=self.feature_expander)
            if torch.isnan(surface.x).any() or torch.isnan(surface.verts).any():
                return None
            return surface
        except Exception as e:
            return None


class GraphLoader:
    """
    This class is used to go load graphs saved in a .pt file
    It is similar to the Surface one, but can also be extended with ESM or ProNet features
    Based on the config it's given, it will load the corresponding features and populate a .x key with them.
    """

    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.esm_dir = config.esm_dir
        self.use_esm = config.use_esm
        self.feature_expander = None

    def load(self, graph_name):
        if not self.config.use_graphs:
            return Data()
        try:
            graph = torch.load(os.path.join(self.data_dir, f"{graph_name}.pt"))
            # patch
            if "node_len" not in graph.keys:
                graph.node_len = len(graph.node_pos)
            feature_keys = self.config.feat_keys
            if self.use_esm:
                esm_feats_path = os.path.join(self.esm_dir, f"{graph_name}_esm.pt")
                esm_feats = torch.load(esm_feats_path)
                graph.features.add_named_features('esm_feats', esm_feats)
                if feature_keys != 'all':
                    feature_keys.append('esm_feats')
            with torch.no_grad():
                graph.expand_features(remove_feats=True,
                                      feature_keys=feature_keys,
                                      oh_keys=self.config.oh_keys,
                                      feature_expander=self.feature_expander)
            if torch.isnan(graph.x).any() or torch.isnan(graph.node_pos).any():
                return None
        except Exception:
            return None
        return graph


def pdb_to_surf_graphs(pdb_path, surface_dump, agraph_dump, rgraph_dump, face_reduction_rate, max_vert_number,
                       use_pymesh=None, compute_s=True, compute_g=True, recompute_s=False, recompute_g=False):
    """
    Wrapper code to go from a PDB to a surface, an AtomGraph and a ResidueGraph
    """
    try:
        if compute_s and (recompute_s or not os.path.exists(surface_dump)):
            use_pymesh = False if use_pymesh is None else use_pymesh
            surface = SurfaceObject.from_pdb_path(pdb_path, face_reduction_rate=face_reduction_rate,
                                                  use_pymesh=use_pymesh, max_vert_number=max_vert_number)
            surface.add_geom_feats()
            surface.save_torch(surface_dump)

        if compute_g and (recompute_g or not os.path.exists(agraph_dump) or not os.path.exists(rgraph_dump)):
            arrays = parse_pdb_path(pdb_path)
            # create atomgraph
            if recompute_g or not os.path.exists(agraph_dump):
                agraph = AtomGraphBuilder().arrays_to_agraph(arrays)
                torch.save(agraph, open(agraph_dump, 'wb'))

            # create residuegraph
            if recompute_g or not os.path.exists(rgraph_dump):
                rgraph = ResidueGraphBuilder(add_pronet=True, add_esm=False).arrays_to_resgraph(arrays)
                torch.save(rgraph, open(rgraph_dump, 'wb'))
        success = 1
    except Exception as e:
        print('*******failed******', pdb_path, e)
        success = 0
    return success


class PreprocessDataset(Dataset):
    """
    Small utility class that handles the boilerplate code of setting up the right repository structure.
    Given a datadir/ as input, expected to hold a datadir/pdb/{}.pdb of pdb files, this dataset will loop through
    those files and generate rgraphs/ agraphs/ and surfaces/ directories and files.
    """

    def __init__(self, datadir, recompute_s=False, recompute_g=False, compute_s=True, compute_g=True,
                 max_vert_number=100000, face_reduction_rate=0.1, use_pymesh=None):
        self.pdb_dir = os.path.join(datadir, 'pdb')

        # Surf params
        self.max_vert_number = max_vert_number
        self.face_reduction_rate = face_reduction_rate
        self.use_pymesh = use_pymesh
        surface_dirname = f'surfaces_{face_reduction_rate}{f"_{use_pymesh}" if use_pymesh is not None else ""}'
        self.out_surf_dir = os.path.join(datadir, surface_dirname)

        # Graph dirs
        self.out_rgraph_dir = os.path.join(datadir, 'rgraph')
        self.out_agraph_dir = os.path.join(datadir, 'agraph')

        # Setup
        os.makedirs(self.out_surf_dir, exist_ok=True)
        os.makedirs(self.out_rgraph_dir, exist_ok=True)
        os.makedirs(self.out_agraph_dir, exist_ok=True)
        self.recompute_s = recompute_s
        self.recompute_g = recompute_g
        self.compute_s = compute_s
        self.compute_g = compute_g

    def get_all_pdbs(self):
        pdb_list = sorted([file_name for file_name in os.listdir(self.pdb_dir) if '.pdb' in file_name])
        return pdb_list

    def __len__(self):
        return len(self.all_pdbs)

    def path_to_surf_graphs(self, pdb_path, surface_dump, agraph_dump, rgraph_dump):
        return pdb_to_surf_graphs(pdb_path, surface_dump, agraph_dump, rgraph_dump,
                                  face_reduction_rate=self.face_reduction_rate,
                                  max_vert_number=self.max_vert_number,
                                  use_pymesh=self.use_pymesh,
                                  recompute_s=self.recompute_s,
                                  recompute_g=self.recompute_g)

    def name_to_surf_graphs(self, name):
        pdb_path = os.path.join(self.pdb_dir, f'{name}.pdb')
        surface_dump = os.path.join(self.out_surf_dir, f'{name}.pt')
        agraph_dump = os.path.join(self.out_agraph_dir, f'{name}.pt')
        rgraph_dump = os.path.join(self.out_rgraph_dir, f'{name}.pt')
        return self.path_to_surf_graphs(pdb_path, surface_dump, agraph_dump, rgraph_dump)


class AtomBatch(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def batch_keys(batch, key):
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
            except Exception:
                batch[key] = batch[key]
        elif isinstance(item, SurfaceObject):
            batch[key] = SurfaceBatch.batch_from_data_list(batch[key])
        elif isinstance(item, ResidueGraph):
            batch[key] = RGraphBatch.batch_from_data_list(batch[key])
        elif isinstance(item, AtomGraph):
            batch[key] = AGraphBatch.batch_from_data_list(batch[key])
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

    @classmethod
    def from_data_list(cls, data_list):
        # Filter out None
        data_list = [x for x in data_list if x is not None]

        batch = cls()
        if len(data_list) == 0:
            batch.num_graphs = 0
            return batch

        # Get all keys
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        # Create a data containing lists of items for every key
        for key in keys:
            batch[key] = []
        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        # Organize the keys together
        for key in batch.keys:
            cls.batch_keys(batch, key)
        batch = batch.contiguous()
        batch.num_graphs = len(data_list)
        return batch


def update_model_input_dim(cfg, dataset_temp, gkey='graph', skey='surface'):
    # Useful to create a Model of the right input dims
    try:
        found = False
        for i, example in enumerate(dataset_temp):
            if example is not None:
                with open_dict(cfg):
                    feat_encoder_kwargs = cfg.encoder.blocks[0].kwargs
                    feat_encoder_kwargs['graph_feat_dim'] = example[gkey].x.shape[1]
                    feat_encoder_kwargs['surface_feat_dim'] = example[skey].x.shape[1]
                found = True
                break
            if i > 50:
                break
        if not found:
            raise RuntimeError('Train dataloader, returned no data, model input dims could not be infered')
    except Exception as e:
        raise Exception('Could not update model input dims because of error: ', e)
