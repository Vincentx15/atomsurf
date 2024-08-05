import os
import sys
import time
import torch
from atom3d.datasets import LMDBDataset
from torch.utils.data import Dataset

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.atom_graph import AtomGraphBuilder
from atomsurf.protein.create_esm import get_esm_embedding_batch
from atomsurf.protein.graphs import parse_pdb_path
from atomsurf.protein.residue_graph import ResidueGraphBuilder
from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.utils.atom_utils import df_to_pdb
from atomsurf.utils.python_utils import do_all

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)

index_columns = ['ensemble', 'subunit', 'structure', 'model', 'chain', 'residue']


def get_subunits(ensemble):
    subunits = ensemble['subunit'].unique()

    if len(subunits) == 4:
        lb = [x for x in subunits if x.endswith('ligand_bound')][0]
        lu = [x for x in subunits if x.endswith('ligand_unbound')][0]
        rb = [x for x in subunits if x.endswith('receptor_bound')][0]
        ru = [x for x in subunits if x.endswith('receptor_unbound')][0]
        bdf0 = ensemble[ensemble['subunit'] == lb]
        bdf1 = ensemble[ensemble['subunit'] == rb]
        udf0 = ensemble[ensemble['subunit'] == lu]
        udf1 = ensemble[ensemble['subunit'] == ru]
        names = (lb, rb, lu, ru)
    elif len(subunits) == 2:
        udf0, udf1 = None, None
        bdf0 = ensemble[ensemble['subunit'] == subunits[0]]
        bdf1 = ensemble[ensemble['subunit'] == subunits[1]]
        names = (subunits[0], subunits[1], None, None)
    else:
        raise RuntimeError('Incorrect number of subunits for pair')
    return names, (bdf0, bdf1, udf0, udf1)


class ExtractPIPpdbDataset(Dataset):
    def __init__(self, datadir=None, mode='train'):
        if datadir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            datadir = os.path.join(script_dir, '..', '..', '..', 'data', 'pip', 'DIPS-split', 'data', mode)
        else:
            datadir = os.path.join(datadir, mode)
        self.pdb_dir = os.path.join(datadir, 'pdb')
        os.makedirs(self.pdb_dir, exist_ok=True)
        self.dataset = LMDBDataset(datadir)
        self.pdb_list = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        protein_pair = self.dataset[idx]
        names, dfs = get_subunits(protein_pair['atoms_pairs'])
        for name, df in zip(names, dfs):
            if name is not None:
                pdb_path = os.path.join(self.pdb_dir, name + '.pdb')
                df_to_pdb(df, pdb_path, recompute=False)
        return 1


class PreprocessPIPDataset(Dataset):
    def __init__(self, datadir=None, recompute=False, mode='train', face_reduction_rate=1.0, use_pymesh=False):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if datadir is None:
            datadir = os.path.join(script_dir, '..', '..', '..', 'data', 'pip', 'DIPS-split', 'data', mode)
        else:
            datadir = os.path.join(datadir, mode)

        self.recompute = recompute
        self.pdb_dir = os.path.join(datadir, 'pdb')
        self.out_surf_dir_full = os.path.join(datadir, f'surfaces_{face_reduction_rate}')
        self.out_rgraph_dir = os.path.join(datadir, 'rgraph')
        self.out_agraph_dir = os.path.join(datadir, 'agraph')
        os.makedirs(self.out_surf_dir_full, exist_ok=True)
        os.makedirs(self.out_rgraph_dir, exist_ok=True)
        os.makedirs(self.out_agraph_dir, exist_ok=True)
        self.pdb_list = os.listdir(self.pdb_dir)
        self.face_reduction_rate = face_reduction_rate
        self.use_pymesh = use_pymesh

    def __len__(self):
        return len(self.pdb_list)

    def __getitem__(self, idx):
        protein = self.pdb_list[idx]
        name = protein[0:-4]
        pdb_path = os.path.join(self.pdb_dir, protein)
        try:
            surface_full_dump = os.path.join(self.out_surf_dir_full, f'{name}.pt')
            agraph_dump = os.path.join(self.out_agraph_dir, f'{name}.pt')
            rgraph_dump = os.path.join(self.out_rgraph_dir, f'{name}.pt')

            if self.recompute or not os.path.exists(surface_full_dump):
                surface = SurfaceObject.from_pdb_path(pdb_path,
                                                      use_pymesh=self.use_pymesh,
                                                      face_reduction_rate=self.face_reduction_rate)
                surface.add_geom_feats()
                surface.save_torch(surface_full_dump)

            if self.recompute or not os.path.exists(agraph_dump) or not os.path.exists(rgraph_dump):
                arrays = parse_pdb_path(pdb_path)
                # create atomgraph
                if self.recompute or not os.path.exists(agraph_dump):
                    agraph_builder = AtomGraphBuilder()
                    agraph = agraph_builder.arrays_to_agraph(arrays)
                    torch.save(agraph, open(agraph_dump, 'wb'))

                # create residuegraph
                if self.recompute or not os.path.exists(rgraph_dump):
                    rgraph_builder = ResidueGraphBuilder(add_pronet=True, add_esm=False)
                    rgraph = rgraph_builder.arrays_to_resgraph(arrays)
                    torch.save(rgraph, open(rgraph_dump, 'wb'))
            success = 1
        except Exception as e:
            print('*******failed******', protein, e)
            success = 0
        return success


if __name__ == '__main__':
    pass
    recompute = False
    for mode in ['train', 'val', 'test']:
    # for mode in ['test']:
        # dataset = ExtractPIPpdbDataset(mode=mode)
        # do_all(dataset, num_workers=20)

        dataset = PreprocessPIPDataset(mode=mode,
                                       face_reduction_rate=0.1,
                                       recompute=recompute)
        do_all(dataset, num_workers=20)

        script_dir = os.path.dirname(os.path.realpath(__file__))
        pip_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'pip', 'DIPS-split', 'data', mode)
        pdb_dir = os.path.join(pip_data_dir, 'pdb')
        out_esm_dir = os.path.join(pip_data_dir, 'esm')
        get_esm_embedding_batch(in_pdbs_dir=pdb_dir, dump_dir=out_esm_dir)
