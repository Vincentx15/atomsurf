import os
import torch
import openbabel
from openbabel import pybel
import warnings

warnings.filterwarnings('ignore')
from torch_geometric.data import Data, HeteroData
from scipy.spatial import distance_matrix
import torch_geometric.transforms as T
import pickle
from tqdm import tqdm
# from transformers import pipeline
import re
from prody import *
import networkx as nx
import numpy as np
import os
from Bio.PDB import *
import atom3d.util.formats as fo
# from utils.protein_utils import featurize_as_graph
from .openbabel_featurizer import Featurizer


def read_ligand(filepath):
    featurizer = Featurizer(save_molecule_codes=False)
    ligand = next(pybel.readfile("mol2", filepath))
    ligand_coord, atom_fea, h_num = featurizer.get_features(ligand)
    ligand_center = torch.tensor(ligand_coord).mean(dim=-2, keepdim=True)

    return ligand_coord, atom_fea, ligand, h_num, ligand_center


def read_protein(filepath):
    featurizer = Featurizer(save_molecule_codes=False)
    protein_pocket = next(pybel.readfile("pdb", filepath))
    pocket_coord, atom_fea, h_num = featurizer.get_features(protein_pocket)

    return pocket_coord, atom_fea, protein_pocket


def info_3D(a, b, c):
    ab = b - a
    ac = c - a
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


def edgelist_to_tensor(edge_list):
    row = []
    column = []
    coo = []
    for edge in edge_list:
        row.append(edge[0])
        column.append(edge[1])

    coo.append(row)
    coo.append(column)

    coo = torch.Tensor(coo)
    edge_tensor = torch.tensor(coo, dtype=torch.long)
    return edge_tensor


def atomlist_to_tensor(atom_list):
    new_list = []
    for atom in atom_list:
        new_list.append([atom])
    atom_tensor = torch.Tensor(new_list)
    return atom_tensor


def bond_fea(bond, atom1, atom2):
    is_Aromatic = int(bond.IsAromatic())
    is_inring = int(bond.IsInRing())
    d = atom1.GetDistance(atom2)

    node1_idx = atom1.GetIdx()
    node2_idx = atom2.GetIdx()

    neighbour1 = []
    neighbour2 = []
    for neighbour_atom in openbabel.OBAtomAtomIter(atom1):
        if (neighbour_atom.GetAtomicNum() != 1) and (neighbour_atom.GetIdx() != node2_idx):
            neighbour1.append(neighbour_atom)

    for neighbour_atom in openbabel.OBAtomAtomIter(atom2):
        if (neighbour_atom.GetAtomicNum() != 1) and (neighbour_atom.GetIdx() != node1_idx):
            neighbour2.append(neighbour_atom)

    if len(neighbour1) == 0 and len(neighbour2) == 0:
        return [d, 0, 0, 0, 0, 0, 0, 0, 0, 0, is_Aromatic, is_Aromatic]

    angel_list = []
    area_list = []
    distence_list = []

    node1_coord = np.array([atom1.GetX(), atom1.GetY(), atom1.GetZ()])
    node2_coord = np.array([atom2.GetX(), atom2.GetY(), atom2.GetZ()])

    for atom3 in neighbour1:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node1_coord, node2_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    for atom3 in neighbour2:
        node3_coord = np.array([atom3.GetX(), atom3.GetY(), atom3.GetZ()])
        angel, area, distence = info_3D(node2_coord, node1_coord, node3_coord)
        angel_list.append(angel)
        area_list.append(area)
        distence_list.append(distence)

    return [d,
            np.max(angel_list) * 0.01, np.sum(angel_list) * 0.01, np.mean(angel_list) * 0.01,
            np.max(area_list), np.sum(area_list), np.mean(area_list),
            np.max(distence_list) * 0.1, np.sum(distence_list) * 0.1, np.mean(distence_list) * 0.1,
            is_Aromatic, is_inring]


def get_complex_edge_fea(edge_list, coord_list):
    net = nx.Graph()
    net.add_weighted_edges_from(edge_list)
    edges_fea = []
    for edge in edge_list:
        edge_fea = []
        edge_fea.append(edge[2])
        edges_fea.append(edge_fea)

    return edges_fea


def Ligand_graph(lig_atoms_fea, ligand, h_num, score):
    edges = []
    edges_fea = []
    for bond in openbabel.OBMolBondIter(ligand.OBMol):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 1) or (atom2.GetAtomicNum() == 1):
            continue
        else:
            idx_1 = atom1.GetIdx() - h_num[atom1.GetIdx() - 1] - 1
            idx_2 = atom2.GetIdx() - h_num[atom2.GetIdx() - 1] - 1

            edge_fea = bond_fea(bond, atom1, atom2)
            edge = [idx_1, idx_2]
            edges.append(edge)
            edges_fea.append(edge_fea)

            re_edge = [idx_2, idx_1]
            edges.append(re_edge)
            edges_fea.append(edge_fea)

    edge_attr = torch.tensor(edges_fea, dtype=torch.float32)
    x = torch.tensor(lig_atoms_fea, dtype=torch.float32)
    edge_index = edgelist_to_tensor(edges)
    G_lig = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=torch.tensor(score))

    return G_lig


def Inter_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, score, cut=5):
    coord_list = []
    for atom in lig_coord:
        coord_list.append(atom)
    for atom in pocket_coord:
        coord_list.append(atom)

    dis = distance_matrix(x=coord_list, y=coord_list)
    lenth = len(coord_list)
    edge_list = []

    edge_list_fea = []
    # Bipartite Graph; i belongs to ligand, j belongs to protein
    for i in range(len(lig_coord)):
        for j in range(len(lig_coord), lenth):
            if dis[i, j] < cut:
                edge_list.append([i, j - len(lig_coord), dis[i, j]])
                edge_list_fea.append([i, j, dis[i, j]])

    data = HeteroData()
    edge_index = edgelist_to_tensor(edge_list)

    data['ligand'].x = torch.tensor(lig_atom_fea, dtype=torch.float32)
    data['ligand'].y = torch.tensor(score)
    data['protein'].x = torch.tensor(pocket_atom_fea, dtype=torch.float32)
    data['ligand', 'protein'].edge_index = edge_index

    complex_edges_fea = get_complex_edge_fea(edge_list_fea, coord_list)
    edge_attr = torch.tensor(complex_edges_fea, dtype=torch.float32)
    data['ligand', 'protein'].edge_attr = edge_attr
    data = T.ToUndirected()(data)

    return data


def Mult_graph(lig_file_name, pocket_file_name, id, score):
    lig_coord, lig_atom_fea, mol, h_num_lig, ligand_center = read_ligand(lig_file_name)
    pocket_coord, pocket_atom_fea, protein = read_protein(pocket_file_name)
    if (mol != None) and (protein != None):
        G_l = Ligand_graph(lig_atom_fea, mol, h_num_lig, score)
        G_inter = Inter_graph(lig_coord, pocket_coord, lig_atom_fea, pocket_atom_fea, score)
        G_list = [G_l, G_inter, id, ligand_center]
        return G_list
    else:
        return None


def GetPDBDict(Path):
    with open(Path, 'rb') as f:
        lines = f.read().decode().strip().split('\n')
    res = {}
    for line in lines:
        if "//" in line:
            temp = line.split()
            name, score = temp[0], float(temp[3])
            res[name] = score
    return res


import os
import sys
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.utils.data_utils import PreprocessDataset
from atomsurf.utils.python_utils import do_all

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)
from torch.utils.data import Dataset


class ExtractMFEData(Dataset):
    def __init__(self, datadir, idxpath, outputdir):

        os.makedirs(outputdir, exist_ok=True)
        self.datadir = datadir
        self.outputdir = outputdir
        self.res = GetPDBDict(Path=idxpath)
        self.protein_list = list(self.res.keys())

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        item = self.protein_list[idx]
        try:
            score = self.res[item]
            G = Mult_graph(os.path.join(self.datadir, item, item + '_ligand.mol2'),
                os.path.join(self.datadir, item, item + '_pocket.pdb'), item, score)
            processed_file = os.path.join(self.outputdir, item + '.pkl')
            with open(processed_file, 'wb') as f:
                pickle.dump(G, f)
            success = 1
        except Exception as e:
            print(item, 'failed!!!!', e)
            success = 0
        return success


class PreProcessPDBbindDataset(PreprocessDataset):

    def __init__(self, idxpath, data_dir=None, out_dir=None, recompute_s=False, recompute_g=False,
                 max_vert_number=100000, face_reduction_rate=1.0, use_pymesh=True):
        super().__init__(data_dir=data_dir, out_dir=out_dir, recompute_s=recompute_s, recompute_g=recompute_g,
            max_vert_number=max_vert_number, face_reduction_rate=face_reduction_rate)
        # Compared to super(), we redefine the original PDB location and the
        # out_surf dir (since those are "_full", as opposed to patches)
        self.res = GetPDBDict(Path=idxpath)
        self.data_dir = data_dir
        self.pdb_dir = data_dir

        self.all_pdbs = list(self.res.keys())

    def __getitem__(self, idx):
        pdb = self.all_pdbs[idx]
        success = self.name_to_surf_graphs_(pdb)
        return success

    def name_to_surf_graphs_(self, name
                             ):
        pdb_path = os.path.join(self.pdb_dir, name, f'{name}_protein.pdb')
        surface_dump = os.path.join(self.out_surf_dir, f'{name}.pt')
        agraph_dump = os.path.join(self.out_agraph_dir, f'{name}.pt')
        rgraph_dump = os.path.join(self.out_rgraph_dir, f'{name}.pt')
        return self.path_to_surf_graphs(pdb_path, surface_dump, agraph_dump, rgraph_dump)


class PreprocessESM_abag(Dataset):
    def __init__(self, datadir=None, outputdir=None):
        # script_dir = os.path.dirname(os.path.realpath(__file__))

        self.pdb_dir = os.path.join(datadir, 'pdb')
        if outputdir == None:
            self.out_esm_dir = os.path.join(datadir, 'esm')
        else:
            self.out_esm_dir = outputdir
        os.makedirs(self.out_esm_dir, exist_ok=True)
        self.pdb_list = []
        for i in os.listdir(self.pdb_dir):
            if '_ab.pdb' or '_ag.pdb' in i:
                self.pdb_list.append(i)

    def __len__(self):
        return len(self.pdb_list)

    def __getitem__(self, idx):
        protein = self.pdb_list[idx]
        pdb_path = os.path.join(self.pdb_dir, protein)
        name = protein[0:-4]
        try:
            embed = get_esm_embedding_single(pdb_path, self.out_esm_dir)
            success = 1
        except Exception as e:
            print('*******failed******', protein, e)
            success = 0
        return success


if __name__ == 'main':
    idxpath = '/work/lpdi/users/ymiao/code/pdbbind/general-set-except-refined/index/INDEX_general_PL_data.2016'
    datadir = '/work/lpdi/users/ymiao/code/pdbbind/refined-set'
    outputdirMFE = '/work/lpdi/users/ymiao/code/pdbbind/preprocessed/MFEdata'
    outputdir = '/work/lpdi/users/ymiao/code/pdbbind/preprocessed/'
    testMFE = ExtractMFEData(datadir=datadir, idxpath=idxpath, outputdir=outputdirMFE)
    do_all(testMFE, num_workers=30)

    datadir = '/work/lpdi/users/ymiao/code/pdbbind/general-set-except-refined/'
    testMFE = ExtractMFEData(datadir=datadir, idxpath=idxpath, outputdir=outputdirMFE)
    do_all(testMFE, num_workers=30)

    testpdbbind = PreProcessPDBbindDataset(data_dir=datadir, idxpath=idxpath, out_dir=outputdir, recompute_s=False,
        recompute_g=False, max_vert_number=100000, face_reduction_rate=0.1, use_pymesh=False)
    do_all(testpdbbind, num_workers=30)

    esm_val = PreprocessESM_abag(datadir='/work/lpdi/users/ymiao/code/antibody-antigen/')
    do_all(esm_val, num_workers=20)

    # process dataset to train val test

    import os
    import pickle
    from atomsurf.tasks.lba.preprocess import GetPDBDict
    import random

    refine = os.listdir('/work/lpdi/users/ymiao/code/pdbbind/refined-set/')
    core = os.listdir('/work/lpdi/users/ymiao/code/pdbbind/coreset/')
    general = os.listdir('/work/lpdi/users/ymiao/code/pdbbind/general-set-except-refined/')
    test_list = []
    train_list_refine = []
    train_list_general = []
    pdblist = GetPDBDict(
        Path='/work/lpdi/users/ymiao/code/pdbbind/general-set-except-refined/index/INDEX_general_PL_data.2016')
    pdblist = list(pdblist.keys())

    for pdb in pdblist:
        if pdb in core:
            test_list.append(pdb)
        elif pdb in refine:
            train_list_refine.append(pdb)
        elif pdb in general:
            train_list_general.append(pdb)

    val_list = random.sample(train_list_refine, 1000)

    train_list = [item for item in train_list_refine if item not in val_list] + train_list_general

    dataset_system = {'train': train_list, 'val': val_list, 'test': test_list}
    # with open('/work/lpdi/users/ymiao/code/pdbbind/preprocessed/dataset_dict.pkl', 'wb') as f:
    #     pickle.dump(dataset_system,f)
    for mode, datalist in zip(['train', 'val', 'test'], [train_list, val_list, test_list]):
        with open('/work/lpdi/users/ymiao/code/pdbbind/preprocessed/' + mode + '.pkl', 'wb') as f:
            pickle.dump(datalist, f)
