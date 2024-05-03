import os
import sys
import time
import numpy as np
import torch
from atom3d.util.formats import df_to_bp
import Bio.PDB as bio
import pandas as pd
import scipy.spatial as spa
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.graphs import parse_pdb_path
from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.protein.atom_graph import AtomGraphBuilder
from atomsurf.protein.residue_graph import ResidueGraphBuilder
from atom3d.datasets import LMDBDataset

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


def df_to_pdb(df, out_file_name, discard_hetatm=True):
    """
    Utility function to go from a df object to a PDB file
    :param df:
    :param out_file_name:
    :return:
    """

    def filter_notaa(struct):
        """
        Discard Hetatm, copied from biopython as as_protein() method is not in biopython 2.78
        :param struct:
        :return:
        """
        remove_list = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    if residue.get_id()[0] != ' ' or not bio.Polypeptide.is_aa(residue):
                        remove_list.append(residue)

        for residue in remove_list:
            residue.parent.detach_child(residue.id)

        for chain in struct.get_chains():  # Remove empty chains
            if not len(chain.child_list):
                chain.parent.detach_child(chain.id)
        return struct
    structure = df_to_bp(df)
    structure = filter_notaa(structure) if discard_hetatm else structure
    io = bio.PDBIO()
    io.set_structure(structure)
    io.save(out_file_name)

class ExtractPIPpdbDataset(Dataset):
    def __init__(self, datadir=None,mode='train'):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if datadir==None:
            datadir = os.path.join(script_dir, '..', '..', '..', 'data', 'DIPS-split','data',mode)
        else:
            datadir= os.path.join(datadir,mode)
        self.pdb_dir = os.path.join(datadir, 'pdb')
        os.makedirs(self.pdb_dir, exist_ok=True)        
        self.dataset = LMDBDataset(datadir)
        self.pdb_list=[]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        protein_pair= self.dataset[idx]
        names, dfs = get_subunits(protein_pair['atoms_pairs'])
        for name,df in zip(names,dfs):
            if name!=None and name+'.pdb' not in self.pdb_list:
                pdb_path = os.path.join(self.pdb_dir, name+'.pdb')
                df_to_pdb(df,pdb_path)
                self.pdb_list.append(name+'.pdb')  
        return 1     
             
class PreprocessPIPDataset(Dataset):
    def __init__(self, datadir=None,recompute=False,mode='train',extract=True,max_vert_number=100000):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if datadir==None:
            datadir = os.path.join(script_dir, '..', '..', '..', 'data', 'DIPS-split','data',mode)
        else:
            datadir= os.path.join(datadir,mode)

        self.recompute = recompute
        self.pdb_dir = os.path.join(datadir, 'pdb')
        self.out_surf_dir_full = os.path.join(datadir, 'surf_full')
        self.out_rgraph_dir = os.path.join(datadir, 'rgraph')
        self.out_agraph_dir = os.path.join(datadir, 'agraph')
        os.makedirs(self.out_surf_dir_full, exist_ok=True)
        os.makedirs(self.out_rgraph_dir, exist_ok=True)
        os.makedirs(self.out_agraph_dir, exist_ok=True)
        self.pdb_list = os.listdir(self.pdb_dir)
        self.max_vert_number=max_vert_number

    def __len__(self):
        return len(self.pdb_list)

    def __getitem__(self, idx):
        protein = self.pdb_list[idx]
        name=protein[0:-4]
        pdb_path=os.path.join(self.pdb_dir,protein)
        try:
            surface_full_dump = os.path.join(self.out_surf_dir_full, f'{name}.pt')
            agraph_dump = os.path.join(self.out_agraph_dir, f'{name}.pt')
            rgraph_dump = os.path.join(self.out_rgraph_dir, f'{name}.pt')

            if self.recompute or not os.path.exists(surface_full_dump):
                surface = SurfaceObject.from_pdb_path(pdb_path, out_ply_path=None, max_vert_number=self.max_vert_number)
                surface.add_geom_feats()
                surface.save_torch(surface_full_dump)

            if self.recompute or not os.path.exists(surface_full_dump) or not os.path.exists(surface_full_dump):
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
            print('*******failed******',protein, e)
            success = 0
        return success



def do_all(dataset, num_workers=4, prefetch_factor=100):
    prefetch_factor = prefetch_factor if num_workers > 0 else 2
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=1,
                            prefetch_factor=prefetch_factor,
                            collate_fn=lambda x: x[0])
    total_success = 0
    t0 = time.time()
    for i, success in enumerate(dataloader):
        # if i > 5: break
        total_success += int(success)
        if not i % 5:
            print(f'Processed {i + 1}/{len(dataloader)}, in {time.time() - t0:.3f}s, with {total_success} successes')
            # => Processed 3351/3362, with 3328 successes ~1% failed systems, mostly missing ply files


if __name__ == '__main__':
    pass
    recompute = True
    dataset = PreprocessPIPDataset(recompute=recompute)
    do_all(dataset, num_workers=4)

