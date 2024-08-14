import os
import sys
import time
import torch
from atom3d.datasets import LMDBDataset
from torch.utils.data import Dataset
import json
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


class ExtractPSRpdbDataset(Dataset):
    def __init__(self, datadir=None, mode='train'):
        if datadir is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            datadir = os.path.join(script_dir, '..', '..', '..', 'psrdata', 'split-by-year', 'data', mode)
        else:
            datadir = os.path.join(datadir, mode)
        self.pdb_dir = os.path.join(datadir, 'pdb')
        os.makedirs(self.pdb_dir, exist_ok=True)
        self.dataset = LMDBDataset(datadir)
        self.pdb_list = []
        self.score_dict={}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ensemble = self.dataset[idx]['atoms']
        subunits = ensemble['subunit'].unique()
        df = ensemble[ensemble['subunit'] == subunits[0]]
        # name = ensemble['structure'].unique()[0]
        name = self.dataset[idx]['id'].split('\'')[1]+'_'+self.dataset[idx]['id'].split('\'')[3]+'.pdb'
        # print(name)
        df_to_pdb(df,  os.path.join(self.pdb_dir, name), recompute=True)
        score= self.dataset[idx]['scores']['gdt_ts']
        self.score_dict[name[:-4]] = {'name':self.dataset[idx]['id'],'score':score}
        return 1


class PreprocessPSRDataset(Dataset):
    def __init__(self, datadir=None, recompute_s=False,recompute_g=False,mode='train', max_vert_number=100000,face_reduction_rate=0.1):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if datadir is None:
            datadir = os.path.join(script_dir, '..', '..', '..', 'psrdata', 'split-by-year', 'data', mode)
        else:
            datadir = os.path.join(datadir, mode)

        # self.recompute = recompute
        self.pdb_dir = os.path.join(datadir, 'pdb')
        self.max_vert_number = max_vert_number
        self.face_reduction_rate = face_reduction_rate
        self.out_surf_dir_full = os.path.join(datadir, f'surfaces_{face_reduction_rate}')
        self.out_rgraph_dir = os.path.join(datadir, 'rgraph')
        self.out_agraph_dir = os.path.join(datadir, 'agraph')
        os.makedirs(self.out_surf_dir_full, exist_ok=True)
        os.makedirs(self.out_rgraph_dir, exist_ok=True)
        os.makedirs(self.out_agraph_dir, exist_ok=True)
        self.pdb_list = []
        for i in os.listdir(self.pdb_dir):
            if '.pdb' in i :
                self.pdb_list.append(i)
        self.pdb_list=self.pdb_list
        self.recompute_s = recompute_s
        self.recompute_g = recompute_g
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

            if self.recompute_s or not os.path.exists(surface_full_dump):
                # surface = SurfaceObject.from_pdb_path(pdb_path, out_ply_path=None, max_vert_number=self.max_vert_number)
                surface = SurfaceObject.from_pdb_path(pdb_path, face_reduction_rate=self.face_reduction_rate,use_pymesh=False,max_vert_number=self.max_vert_number)
                surface.add_geom_feats()
                surface.save_torch(surface_full_dump)

            if self.recompute_g or not os.path.exists(agraph_dump) or not os.path.exists(rgraph_dump):
                arrays = parse_pdb_path(pdb_path)
                # create atomgraph
                if self.recompute_g or not os.path.exists(agraph_dump):
                    agraph_builder = AtomGraphBuilder()
                    agraph = agraph_builder.arrays_to_agraph(arrays)
                    torch.save(agraph, open(agraph_dump, 'wb'))

                # create residuegraph
                if self.recompute_g or not os.path.exists(rgraph_dump):
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
    datadir='/work/lpdi/users/ymiao/code/psrdata/split-by-year/data'
    for mode in ['train', 'val', 'test']:
        dataset= ExtractPSRpdbDataset(datadir=datadir,mode=mode)
        do_all(dataset,num_workers=30)
        with open(os.path.join(datadir,mode,mode+'_score.json'), "w") as outfile: 
            json.dump(dataset.score_dict, outfile)
        
        testset= PreprocessPSRDataset(datadir=datadir,recompute_s=True,recompute_g=True,mode=mode, max_vert_number=100000,face_reduction_rate=0.1)
        do_all(testset,num_workers=40)