from Bio import PDB
import os
import sys

from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
from pathlib import Path
import scipy.spatial as ss
from subprocess import Popen, PIPE
import shutil
import torch
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from Bio.PDB.DSSP import DSSP
import pandas as pd
from scipy.spatial.distance import cdist
import json
from torch.utils.data import Dataset
import time
from atomsurf.protein.atom_graph import AtomGraphBuilder
from atomsurf.protein.create_esm import get_esm_embedding_batch
from atomsurf.protein.graphs import parse_pdb_path
from atomsurf.protein.residue_graph import ResidueGraphBuilder
from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.utils.python_utils import do_all

def extract_chains(input_pdb, output_pdb, chains_to_extract):
    # Initialize the PDB parser and structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', input_pdb)

    # Create a new structure object to store the extracted chains
    new_structure = PDB.Structure.Structure('extracted_chains')

    for model in structure:
        new_model = PDB.Model.Model(model.id)
        for chain in model:
            if chain.id in chains_to_extract:
                new_model.add(chain)
        if len(new_model):
            new_structure.add(new_model)

    # Save the new structure with the selected chains
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)
    print(f"Chains {', '.join(chains_to_extract)} extracted and saved to {output_pdb}")
atom_type_dict = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'P': 6, 'Cl': 7, 'Se': 8,
                  'Br': 9, 'I': 10, 'UNK': 11}
# residue type label for one-hot-encoding
res_type_dict = {
    'ALA': 0, 'GLY': 1, 'SER': 2, 'THR': 3, 'LEU': 4, 'ILE': 5, 'VAL': 6, 'ASN': 7, 'GLN': 8, 'ARG': 9, 'HIS': 10,
    'TRP': 11, 'PHE': 12, 'TYR': 13, 'GLU': 14, 'ASP': 15, 'LYS': 16, 'PRO': 17, 'CYS': 18, 'MET': 19, 'UNK': 20, }

protein_letters_1to3 = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "X": "Unk"
}

protein_letters_3to1 = {value.upper(): key for key, value in protein_letters_1to3.items()}

res_type_idx_to_1 = {
    idx: protein_letters_3to1[res_type] for res_type, idx in res_type_dict.items()
}

# Kyte Doolittle scale for hydrophobicity
hydrophob_dict = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5, 'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7,
    'SER': -0.8, 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5, 'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5,
    'LYS': -3.9, 'ARG': -4.5, 'UNK': 0.0,
}

res_type_to_hphob = {
    idx: hydrophob_dict[res_type] for res_type, idx in res_type_dict.items()
}
SSE_type_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
chothia_cdr_def = {
    "L1" : (24, 34), "L2" : (50, 56), "L3" : (89, 97),
    "H1" : (26, 32), "H2" : (52, 56), "H3" : (95, 102) }


def parse_pdb_path_new(pdb_path):
    pdb2pqr_bin = shutil.which('pdb2pqr')
    if pdb2pqr_bin is None:
        raise RuntimeError('pdb2pqr executable not found')

    pdb_path = Path(pdb_path)
    # process pqr
    out_dir = pdb_path.parent
    pdb_id = pdb_path.stem
    pqr_path = Path(out_dir / f'{pdb_id}.pqr')
    pqr_log_path = Path(out_dir / f'{pdb_id}.log')
    if not pqr_path.exists():
        cmd = [pdb2pqr_bin, '--ff=AMBER', str(pdb_path), str(pqr_path), '--keep-chain']
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        err = stderr.decode('utf-8').strip('\n')
        if 'CRITICAL' in err:
            print(f'{pdb_id} pdb2pqr failed', flush=True)
            return None, None, None, None, None, None, None, None, None
    parser = PDBParser(QUIET=True, is_pqr=True)
    structure = parser.get_structure("toto", pqr_path)

    amino_types = []  # size: (n_amino,)
    atom_chain_id = []  # size:(n_atom,)
    atom_amino_id = []  # size: (n_atom,)
    atom_names = []  # size: (n_atom,)
    atom_types = []  # size: (n_atom,)
    atom_pos = []  # size: (n_atom,3)
    atom_charge = []  # size: (n_atom,1)
    atom_radius = []  # size: (n_atom,1)
    real_resid = []
    res_id = 0
    for residue in structure.get_residues():
        for atom in residue.get_atoms():
            # Add occupancy to write as pdb
            atom.set_occupancy(1.0)
            atom.set_bfactor(1.0)

        # HETATM
        if residue.id[0] != " ":
            continue
        resname = residue.get_resname()
        # resname = protein_letters_3to1[resname.title()]
        if resname.upper() not in res_type_dict:
            resname = 'UNK'
        resname = res_type_dict[resname.upper()]
        amino_types.append(resname)

        for atom in residue.get_atoms():
            # Skip H
            element = atom.element
            if atom.get_name().startswith("H"):
                continue
            if not element in atom_type_dict:
                element = 'UNK'
            atom_chain_id.append(residue.full_id[2])
            atom_types.append(atom_type_dict[element])
            atom_names.append(atom.get_name())
            atom_pos.append(atom.get_coord())
            atom_amino_id.append(res_id)
            atom_charge.append(atom.get_charge())
            atom_radius.append(atom.get_radius())
            real_resid.append((residue.full_id[2]+str(residue.id[1])))
        res_id += 1
    amino_types = np.asarray(amino_types, dtype=np.int32)
    atom_chain_id = np.asarray(atom_chain_id)
    atom_amino_id = np.asarray(atom_amino_id, dtype=np.int32)
    atom_names = np.asarray(atom_names)
    atom_types = np.asarray(atom_types, dtype=np.int32)
    atom_pos = np.asarray(atom_pos, dtype=np.float32)
    atom_charge = np.asarray(atom_charge, dtype=np.float32)
    atom_radius = np.asarray(atom_radius, dtype=np.float32)
    real_resid = np.asarray(real_resid)

    return amino_types, atom_chain_id, atom_amino_id, atom_names, atom_types, atom_pos, atom_charge, atom_radius,real_resid



class ExtractAALabelDataset(Dataset):
    def __init__(self, csv_file,datadir):

        self.pdblist = pd.read_csv(csv_file)
        self.datadir= datadir
    def __len__(self):
        return len(self.pdblist)

    def __getitem__(self, idx):
        row=self.pdblist.loc[idx]
        try:
        # if True:
            chains_HL = [row['Hchain'], row['Lchain']]    # List of chains to extract
            chains_ag = [row['antigen_chain']]
            input_pdb=self.datadir+row['pdb']+'.pdb'
            output_pdbab=self.datadir+row['pdb']+'_ab.pdb'
            output_pdbag=self.datadir+row['pdb']+'_ag.pdb'
            extract_chains(input_pdb, output_pdbab, chains_HL)
            extract_chains(input_pdb, output_pdbag, chains_ag)
            chain={'H':row['Hchain'],'L':row['Lchain'],'ag':row['antigen_chain']}
            arrays1=parse_pdb_path_new(output_pdbab)
            arrays2=parse_pdb_path_new(output_pdbag)
            amino_types1, atom_chain_id1, atom_amino_id1, atom_names1, atom_types1, atom_pos1, atom_charge1, atom_radius1, real_resid1 = arrays1
            amino_types2, atom_chain_id2, atom_amino_id2, atom_names2, atom_types2, atom_pos2, atom_charge2, atom_radius2, real_resid2 = arrays2
            
            cdr_region=[]
            for cdrname, pos in chothia_cdr_def.items():
                cdr_region+=([chain[cdrname[0]]+str(i) for i in range(pos[0]-2,pos[1]+2+1)])
            cdr_mask=np.zeros_like(real_resid1,dtype=bool)
            for i in range(0,len(real_resid1)):
                if real_resid1[i] in cdr_region:
                    cdr_mask[i]=True
            dists= cdist(atom_pos1,atom_pos2)
            idx1,idx2=np.where(dists<4.5)
            contact_mask1=np.zeros_like(real_resid1,dtype=bool)
            contact_mask2=np.zeros_like(real_resid2,dtype=bool)
            contact_mask1[idx1]=True
            contact_mask2[idx2]=True
            cdr_res= np.unique(atom_amino_id1[ cdr_mask ])
            cdr_contact_res= np.unique(atom_amino_id1[ cdr_mask&contact_mask1 ])
            ag_contact_res= np.unique(atom_amino_id2[ contact_mask2 ])
            label_dict={}
            label_dict['cdr'] = cdr_res.tolist()
            label_dict['cdr_contact'] = cdr_contact_res.tolist()
            label_dict['ag_contact'] = ag_contact_res.tolist()
            contact_pair= [(i,j) for i,j in zip(atom_amino_id1[idx1],atom_amino_id2[idx2])]
            contact_pair=list(set(contact_pair))
            label_dict['cdr_ag_pair'] = np.array(contact_pair).tolist()
            # label_dict['cdr_ag_pair'] = 
            with open(self.datadir+row['pdb']+'.json', "w") as outfile: 
                json.dump(label_dict, outfile)
            # print(self.label_dict)
            return 1
        except:
            print('failed',row['pdb'])
            return 0
class PreprocessAAPDataset(Dataset):
    def __init__(self, csv_file=None,datadir=None, recompute_s=True,recompute_g=True, max_vert_number=100000,face_reduction_rate=0.1):

        # self.recompute = recompute
        self.pdb_list = pd.read_csv(csv_file)
        self.datadir = datadir
        self.pdb_dir = os.path.join(datadir, 'pdbs')
        self.max_vert_number = max_vert_number
        self.face_reduction_rate = face_reduction_rate
        self.out_surf_dir_full = os.path.join(datadir, f'surfaces_{face_reduction_rate}')
        self.out_rgraph_dir = os.path.join(datadir, 'rgraph')
        self.out_agraph_dir = os.path.join(datadir, 'agraph')
        os.makedirs(self.out_surf_dir_full, exist_ok=True)
        os.makedirs(self.out_rgraph_dir, exist_ok=True)
        os.makedirs(self.out_agraph_dir, exist_ok=True)
        self.recompute_s = recompute_s
        self.recompute_g = recompute_g
        
    def __len__(self):
        return len(self.pdb_list)

    def __getitem__(self, idx):
        protein = self.pdb_list.loc[idx]
        names= [protein['pdb']+'_ab.pdb',protein['pdb']+'_ag.pdb']
        
        
        # if True:
        try:
            for name in names:
                pdb_path = os.path.join(self.pdb_dir, name)
                print(pdb_path)
                surface_full_dump = os.path.join(self.out_surf_dir_full, f'{name}.pt')
                agraph_dump = os.path.join(self.out_agraph_dir, f'{name}.pt')
                rgraph_dump = os.path.join(self.out_rgraph_dir, f'{name}.pt')
                
                if self.recompute_s or not os.path.exists(surface_full_dump):
                    # surface = SurfaceObject.from_pdb_path(pdb_path, out_ply_path=None, max_vert_number=self.max_vert_number)
                    surface = SurfaceObject.from_pdb_path(pdb_path, face_reduction_rate=self.face_reduction_rate,use_pymesh=False,max_vert_number=self.max_vert_number)
                    surface.add_geom_feats()
                    surface.save_torch(surface_full_dump)
    
                if self.recompute_g or not os.path.exists(surface_full_dump) or not os.path.exists(surface_full_dump):
                    arrays = parse_pdb_path(pdb_path)
                    amino_types, atom_chain_id, atom_amino_id, atom_names, _, atom_pos, atom_charge, atom_radius, res_sse = arrays
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
    valset_extract=ExtractAALabelDataset('./antibody-antigen/val_pecan_aligned.csv','./GEP/pdbs/')
    do_all(valset_extract,num_workers=30)
    valset_pre=PreprocessAAPDataset(csv_file='./antibody-antigen/val_pecan_aligned.csv',datadir='./GEP/')
    do_all(valset_pre,num_workers=30)