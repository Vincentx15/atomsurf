import os
import re

from omegaconf import open_dict
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from atomsurf.protein.surfaces import SurfaceObject, SurfaceBatch
from atomsurf.protein.graphs import parse_pdb_path, parse_pdb_path_nopqr
from atomsurf.protein.atom_graph import AtomGraph, AGraphBatch, AtomGraphBuilder
from atomsurf.protein.residue_graph import ResidueGraph, RGraphBatch, ResidueGraphBuilder
from atomsurf.utils.python_utils import makedirs_path


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
        self.data_dir = os.path.join(config.data_dir, config.data_name)
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
            # Early version of the data did not include the normals
            surface.set_vnormals()
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
        self.data_dir = os.path.join(config.data_dir, config.data_name)
        self.use_esm = config.use_esm
        self.esm_dir = config.esm_dir
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


def pdb_to_surf(pdb_path, surface_dump, face_reduction_rate=0.1, max_vert_number=100000, use_pymesh=None,
                recompute_s=False):
    """
    Wrapper code to go from a PDB to a surface
    """
    try:
        if recompute_s or not os.path.exists(surface_dump):
            use_pymesh = False if use_pymesh is None else use_pymesh
            surface = SurfaceObject.from_pdb_path(pdb_path, face_reduction_rate=face_reduction_rate,
                use_pymesh=use_pymesh, max_vert_number=max_vert_number)
            surface.add_geom_feats()
            surface.save_torch(surface_dump)
        success = 1
    except Exception as e:
        print('pdb_to_surf failed for : ', pdb_path, e)
        success = 0
    return success


def pdb_to_graphs(pdb_path, agraph_dump=None, rgraph_dump=None, recompute_g=False):
    """
    Wrapper code to go from a PDB to an AtomGraph and a ResidueGraph
    """
    try:
        do_rgraphs = rgraph_dump is not None and (recompute_g or not os.path.exists(rgraph_dump))
        do_agraphs = agraph_dump is not None and (recompute_g or not os.path.exists(agraph_dump))
        if do_rgraphs or do_agraphs:
            try:
                arrays = parse_pdb_path_nopqr(pdb_path)
            except:
                print('Trying to use pqr to fix sse')
                arrays = parse_pdb_path(pdb_path)
            # create residuegraph
            if do_rgraphs:
                rgraph = ResidueGraphBuilder(add_pronet=True, add_esm=False).arrays_to_resgraph(arrays)
                makedirs_path(rgraph_dump)
                torch.save(rgraph, open(rgraph_dump, 'wb'))
            # create atomgraph
            if do_agraphs:
                agraph = AtomGraphBuilder().arrays_to_agraph(arrays)
                makedirs_path(agraph_dump)
                torch.save(agraph, open(agraph_dump, 'wb'))
        success = 1
    except Exception as e:
        print('pdb_to_graph failed for : ', pdb_path, e)
        success = 0
    return success


class PreprocessDataset(Dataset):
    """
    Small utility class that handles the boilerplate code of setting up the right repository structure.
    Given a data_dir/ as input, expected to hold a data_dir/pdb/{}.pdb of pdb files, this dataset will loop through
    those files and generate rgraphs/ agraphs/ and surfaces/ directories and files.
    """

    def __init__(self, data_dir, recompute_s=False, recompute_g=False, do_agraph=False,
                 max_vert_number=100000, face_reduction_rate=0.1, use_pymesh=None):
        self.pdb_dir = os.path.join(data_dir, 'pdb')

        # Surf params
        self.max_vert_number = max_vert_number
        self.face_reduction_rate = face_reduction_rate
        self.use_pymesh = use_pymesh
        surface_dirname = f'surfaces_{face_reduction_rate}{f"_{use_pymesh}" if use_pymesh is not None else ""}'
        self.out_surf_dir = os.path.join(data_dir, surface_dirname)
        os.makedirs(self.out_surf_dir, exist_ok=True)

        # Graph dirs
        self.out_rgraph_dir = os.path.join(data_dir, 'rgraph')
        os.makedirs(self.out_rgraph_dir, exist_ok=True)

        self.do_agraph = do_agraph
        if do_agraph:
            self.out_agraph_dir = os.path.join(data_dir, 'agraph')
            os.makedirs(self.out_agraph_dir, exist_ok=True)

        self.recompute_s = recompute_s
        self.recompute_g = recompute_g
        self.all_pdbs = self.get_all_pdbs()

    def get_all_pdbs(self):
        pdb_list = sorted([file_name for file_name in os.listdir(self.pdb_dir) if '.pdb' in file_name])
        return pdb_list

    def __len__(self):
        return len(self.all_pdbs)

    def path_to_surf(self, pdb_path, surface_dump):
        return pdb_to_surf(pdb_path, surface_dump,
            face_reduction_rate=self.face_reduction_rate,
            max_vert_number=self.max_vert_number,
            use_pymesh=self.use_pymesh,
            recompute_s=self.recompute_s)

    def path_to_graphs(self, pdb_path, agraph_dump, rgraph_dump):
        return pdb_to_graphs(pdb_path, agraph_dump, rgraph_dump, recompute_g=self.recompute_g)

    def path_to_surf_graphs(self, pdb_path, surface_dump, agraph_dump, rgraph_dump):
        success1 = self.path_to_surf(pdb_path, surface_dump)
        success2 = self.path_to_graphs(pdb_path, agraph_dump, rgraph_dump)
        return success1 and success2

    def name_to_surf(self, name):
        pdb_path = os.path.join(self.pdb_dir, f'{name}.pdb')
        surface_dump = os.path.join(self.out_surf_dir, f'{name}.pt')
        return self.path_to_surf(pdb_path, surface_dump)

    def name_to_graphs(self, name):
        pdb_path = os.path.join(self.pdb_dir, f'{name}.pdb')
        agraph_dump = os.path.join(self.out_agraph_dir, f'{name}.pt') if self.do_agraph else None
        rgraph_dump = os.path.join(self.out_rgraph_dir, f'{name}.pt')
        return self.path_to_graphs(pdb_path, agraph_dump, rgraph_dump)

    def name_to_surf_graphs(self, name):
        success1 = self.name_to_surf(name)
        success2 = self.name_to_graphs(name)
        return success1 and success2

    def __getitem__(self, idx):
        pdb = self.all_pdbs[idx]
        name = pdb[0:-4]
        success = self.name_to_surf_graphs(name)
        return success


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
            # Previously, we were trying torch.stack: batch[key] = torch.stack(batch[key])
            # This was unsafe (lucky shot?)
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
                    feat_encoder_kwargs = cfg.encoder.blocks[0]
                    feat_encoder_kwargs.g_pre_block['dim_in'] = example[gkey].x.shape[1]
                    feat_encoder_kwargs.s_pre_block['dim_in'] = example[skey].x.shape[1]
                found = True
                break
            if i > 50:
                break
        if not found:
            raise RuntimeError('Train dataloader, returned no data, model input dims could not be inferred')
    except Exception as e:
        raise Exception('Could not update model input dims because of error: ', e)
