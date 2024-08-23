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

from atomsurf.utils.atom_utils import df_to_pdb
from atomsurf.utils.data_utils import PreprocessDataset
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
        self.score_dict = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ensemble = self.dataset[idx]['atoms']
        subunits = ensemble['subunit'].unique()
        df = ensemble[ensemble['subunit'] == subunits[0]]
        # name = ensemble['structure'].unique()[0]
        name = self.dataset[idx]['id'].split('\'')[1] + '_' + self.dataset[idx]['id'].split('\'')[3] + '.pdb'
        # print(name)
        df_to_pdb(df, os.path.join(self.pdb_dir, name), recompute=True)
        score = self.dataset[idx]['scores']['gdt_ts']
        self.score_dict[name[:-4]] = {'name': self.dataset[idx]['id'], 'score': score}
        return 1


class PreprocessPSRDataset(PreprocessDataset):
    def __init__(self, datadir=None, recompute_s=False, recompute_g=False, mode='train', max_vert_number=100000,
                 face_reduction_rate=0.1):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if datadir is None:
            datadir = os.path.join(script_dir, '..', '..', '..', 'psrdata', 'split-by-year', 'data', mode)
        else:
            datadir = os.path.join(datadir, mode)

        super().__init__(datadir=datadir, recompute_s=recompute_s, recompute_g=recompute_g,
                         max_vert_number=max_vert_number, face_reduction_rate=face_reduction_rate)

        self.all_pdbs = self.get_all_pdbs()

    def __getitem__(self, idx):
        pdb = self.all_pdbs[idx]
        name = pdb[0:-4]
        success = self.name_to_surf_graphs(name)
        return success


if __name__ == '__main__':
    pass
    recompute = False
    datadir = '/work/lpdi/users/ymiao/code/psrdata/split-by-year/data'
    for mode in ['train', 'val', 'test']:
        dataset = ExtractPSRpdbDataset(datadir=datadir, mode=mode)
        do_all(dataset, num_workers=30)
        with open(os.path.join(datadir, mode, mode + '_score.json'), "w") as outfile:
            json.dump(dataset.score_dict, outfile)

        testset = PreprocessPSRDataset(datadir=datadir, recompute_s=True, recompute_g=True, mode=mode,
                                       max_vert_number=100000, face_reduction_rate=0.1)
        do_all(testset, num_workers=40)
