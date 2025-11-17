import os
import sys

from atom3d.datasets import LMDBDataset
import math
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import pandas as pd 
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.tasks.pip_site.preprocess import get_subunits
from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader, AtomBatch, update_model_input_dim


class SurfaceLoaderPIP(SurfaceLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.data_dir = os.path.join(config.data_dir, mode, config.data_name)


class GraphLoaderPIP(GraphLoader):
    def __init__(self, config, mode):
        super().__init__(config)
        self.config = config
        self.data_dir = os.path.join(config.data_dir, mode, config.data_name)
        self.esm_dir = os.path.join(config.data_dir, mode, 'esm')


class PIPsiteDataset(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.systems = LMDBDataset(data_dir)
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble

    def __len__(self):
        return len(self.systems)

    def _num_to_use(self, num_pos, num_neg):
        """
        Depending on the number of pos and neg of the system, we might want to use
            different amounts of positive or negative coordinates.

        :param num_pos:
        :param num_neg:
        :return:
        """

        if self.neg_to_pos_ratio == -1:
            num_pos_to_use, num_neg_to_use = num_pos, num_neg
        else:
            num_pos_to_use = min(num_pos, num_neg / self.neg_to_pos_ratio)
            if self.max_pos_regions_per_ensemble != -1:
                num_pos_to_use = min(num_pos_to_use, self.max_pos_regions_per_ensemble)
            num_neg_to_use = num_pos_to_use * self.neg_to_pos_ratio
        num_pos_to_use = int(math.ceil(num_pos_to_use))
        num_neg_to_use = int(math.ceil(num_neg_to_use))
        return num_pos_to_use, num_neg_to_use

    def __getitem__(self, idx):
        protein_pair = self.systems[idx]
        pos_pairs = protein_pair['atoms_neighbors']
        names, dfs = get_subunits(protein_pair['atoms_pairs'])
        pdbca1 = dfs[0][(dfs[0]['name'] == 'CA') & (dfs[0]['hetero'] == ' ') & (dfs[0]['resname'] != 'UNK')]
        pdbca2 = dfs[1][(dfs[1]['name'] == 'CA') & (dfs[1]['hetero'] == ' ') & (dfs[1]['resname'] != 'UNK')]
        pos_pairs_res = pos_pairs[
            (pos_pairs['residue0'].isin(pdbca1.residue)) & (pos_pairs['residue1'].isin(pdbca2.residue))]

        mapping_1 = {resindex: i for i, resindex in enumerate(pdbca1.residue.values)}
        mapping_2 = {resindex: i for i, resindex in enumerate(pdbca2.residue.values)}
        pos_as_array_1 = np.array([mapping_1[resi] for resi in pos_pairs_res['residue0']])
        pos_as_array_2 = np.array([mapping_2[resi] for resi in pos_pairs_res['residue1']])
        # dense = np.zeros((len(pdbca1), len(pdbca2)))
        dense1 = np.zeros(len(pdbca1))
        dense2 = np.zeros(len(pdbca2))
        if len(pos_as_array_1)< 3 or len(pos_as_array_2)< 3:
            # print('no interact! skip')
            return None
        dense1[pos_as_array_1] = 1
        negs_1 = np.where(dense1 == 0)[0]
        pos_1 = np.where(dense1 == 1)[0]
        dense2[pos_as_array_2] = 1
        negs_2 = np.where(dense2 == 0)[0]
        pos_2 = np.where(dense2 == 1)[0]
        if len(pos_1)< 3 or len(pos_2)< 3:
            # print('no interact! skip')
            return None
        # pos_array = np.stack((pos_as_array_1, pos_as_array_2))
        # neg_array = np.stack((negs_1, negs_2))
        num_pos = min(len(pos_1),len(pos_2))
        num_neg = min(len(negs_1),len(negs_2))
        num_pos_to_use, num_neg_to_use = self._num_to_use(num_pos, num_neg)

        # pos_array_idx1 = np.random.choice(len(pos_1), size=num_pos_to_use, replace=False)
        # pos_array_idx2 = np.random.choice(len(pos_2), size=num_pos_to_use, replace=False)
        # neg_array_idx1 = np.random.choice(len(negs_1), size=num_neg_to_use, replace=False)
        # neg_array_idx2 = np.random.choice(len(negs_2), size=num_neg_to_use, replace=False)
        pos_array_sampled1 = pos_1#[pos_array_idx1]
        pos_array_sampled2 = pos_2#[pos_array_idx2]
        neg_array_sampled1 = negs_1#[neg_array_idx1]
        neg_array_sampled2 = negs_2#[neg_array_idx2]
        
        pos_array_sampled1 = torch.from_numpy(pos_array_sampled1)
        pos_array_sampled2 = torch.from_numpy(pos_array_sampled2)
        neg_array_sampled1 = torch.from_numpy(neg_array_sampled1)
        neg_array_sampled2 = torch.from_numpy(neg_array_sampled2)
        idx_left = torch.cat((pos_array_sampled1, neg_array_sampled1))
        idx_right = torch.cat((pos_array_sampled2, neg_array_sampled2))
        label_l = torch.cat((torch.ones(len(pos_1)),torch.zeros(len(negs_1)) ))
        label_r = torch.cat((torch.ones(len(pos_2)),torch.zeros(len(negs_2)) ))
        # labels = torch.cat((torch.ones(num_pos_to_use), torch.zeros(num_neg_to_use)))
        surface_1 = self.surface_loader.load(names[0])
        surface_2 = self.surface_loader.load(names[1])
        graph_1 = self.graph_loader.load(names[0])
        graph_2 = self.graph_loader.load(names[1])
        
        if surface_1 is None or surface_2 is None or graph_1 is None or graph_2 is None:
            return None
        if graph_1.node_len < 20 or graph_2.node_len < 20 or surface_1.n_verts < 20 or surface_2.n_verts < 20:
            return None
        if idx_left.dtype != torch.int64 and idx_right.dtype != torch.int64 or len(idx_left) <3 or len(idx_right) <3 :
            return None
        if idx_left.max() >= len(graph_1.node_pos) or idx_right.max() >= len(graph_2.node_pos):
            print('idx error', names)
            return None
        item = Data(surface_1=surface_1, graph_1=graph_1, surface_2=surface_2, graph_2=graph_2, idx_left=idx_left,
                    idx_right=idx_right, label_r=label_r,label_l=label_l, g1_len=graph_1.node_pos.shape[0],
                    g2_len=graph_2.node_pos.shape[0],id=[pos_pairs.subunit0[0],pos_pairs.subunit1[0]])
        return item



class PINDERDataset(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, mode,neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.data_dir= data_dir
        # self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster.parquet')) # 
        self.index= pd.read_parquet(os.path.join(data_dir,'exist_all_holo_index.parquet'))
        # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
        self.index= self.index[self.index.split==mode]
        # metadata= pd.read_parquet(data_dir+'/processed_pinder_metadata.parquet')
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble

    def __len__(self):
        return len(self.index)

    def _num_to_use(self, num_pos, num_neg):
        """
        Depending on the number of pos and neg of the system, we might want to use
            different amounts of positive or negative coordinates.

        :param num_pos:
        :param num_neg:
        :return:
        """

        if self.neg_to_pos_ratio == -1:
            num_pos_to_use, num_neg_to_use = num_pos, num_neg
        else:
            num_pos_to_use = min(num_pos, num_neg / self.neg_to_pos_ratio)
            if self.max_pos_regions_per_ensemble != -1:
                num_pos_to_use = min(num_pos_to_use, self.max_pos_regions_per_ensemble)
            num_neg_to_use = num_pos_to_use * self.neg_to_pos_ratio
        num_pos_to_use = int(math.ceil(num_pos_to_use))
        num_neg_to_use = int(math.ceil(num_neg_to_use))
        return num_pos_to_use, num_neg_to_use

    def __getitem__(self, idx):
        protein_pair = self.index.iloc[idx]['id']
        pdb_R = self.index.iloc[idx]['holo_R_pdb']
        pdb_L = self.index.iloc[idx]['holo_L_pdb']
        if os.path.exists(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface/'+protein_pair+'_interface.pt')):
            interface = torch.load(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface/'+protein_pair+'_interface.pt'))
        else:
            return None
        idx_R_pos = interface[pdb_R+'_mapping'][0]
        idx_L_pos = interface[pdb_L+'_mapping'][0]
        pos_pair= torch.from_numpy(interface['pair_mapping'])
        pos_pair_L = pos_pair[:,0]
        pos_pair_R = pos_pair[:,1]
        graph_R= self.graph_loader.load(pdb_R[:-4])
        graph_L=  self.graph_loader.load(pdb_L[:-4])
        surface_R =self.surface_loader.load(pdb_R[:-4])
        surface_L =self.surface_loader.load(pdb_L[:-4])

        if surface_L is None or surface_R is None or graph_L is None or graph_R is None:
            return None
        if graph_L.node_len < 20 or graph_R.node_len < 20 or surface_L.n_verts < 20 or surface_R.n_verts < 20:
            return None
        if idx_R_pos.max() >= graph_R.node_len or idx_L_pos.max() >= graph_L.node_len:
            return None
        denseR = np.zeros(len(graph_R.node_pos))
        denseL = np.zeros(len(graph_L.node_pos))
        # if len(idx_R_pos)< 3 or len(idx_L_pos)< 3:
        #     return None
        denseL[idx_L_pos] = 1 
        denseR[idx_R_pos] = 1
        idx_L_neg = np.where(denseL==0)[0]
        idx_R_neg = np.where(denseR==0)[0]
        
        #num_to_use = min(min(len(idx_R_pos),len(idx_L_pos)),min(len(idx_R_neg),len(idx_L_neg)))
        # pos_array_sampledL = torch.from_numpy(idx_L_pos[np.random.choice(len(idx_L_pos), size=num_to_use, replace=False)])
        # pos_array_sampledR = torch.from_numpy(idx_R_pos[np.random.choice(len(idx_R_pos), size=num_to_use, replace=False)])
        # neg_array_sampledL = torch.from_numpy(idx_L_neg[np.random.choice(len(idx_L_neg), size=num_to_use, replace=False)])
        # neg_array_sampledR = torch.from_numpy(idx_R_neg[np.random.choice(len(idx_R_neg), size=num_to_use, replace=False)])
        pos_array_sampledL = torch.from_numpy(idx_L_pos)
        pos_array_sampledR = torch.from_numpy(idx_R_pos)
        neg_array_sampledL = torch.from_numpy(idx_L_neg)
        neg_array_sampledR = torch.from_numpy(idx_R_neg)
        
        idx_left = torch.cat((pos_array_sampledL, neg_array_sampledL))
        idx_right = torch.cat((pos_array_sampledR, neg_array_sampledR))
        label_l = torch.cat((torch.ones(len(pos_array_sampledL)),torch.zeros(len(neg_array_sampledL))))
        label_r = torch.cat((torch.ones(len(pos_array_sampledR)),torch.zeros(len(neg_array_sampledR))))
        #prepare pair idx
        dense_pair=np.zeros([len(graph_L.node_pos),len(graph_R.node_pos)])
        dense_pair[pos_pair_L,pos_pair_R]=1.0
        pos_pair= np.vstack([pos_pair[:,0],pos_pair[:,1]])
        neg_pair = np.where(dense_pair==0)
        neg_pair = np.vstack([neg_pair[0],neg_pair[1]])
        num_pair_to_use = min(len(neg_pair[0]),len(pos_pair_L))
        pos_array_idx = np.random.choice(len(pos_pair_L), size=num_pair_to_use, replace=False)
        neg_array_idx = np.random.choice(len(neg_pair[0]), size=num_pair_to_use, replace=False)
        pos_pair_sampled_p = torch.from_numpy(pos_pair[:,pos_array_idx])
        neg_pair_sampled_p  =  torch.from_numpy(neg_pair[:, neg_array_idx])
        idx_left_pair = torch.cat((pos_pair_sampled_p[0], neg_pair_sampled_p[0]))
        idx_right_pair = torch.cat((pos_pair_sampled_p[1], neg_pair_sampled_p[1]))
        labels_pair = torch.cat((torch.ones(num_pair_to_use), torch.zeros(num_pair_to_use)))
        if idx_left.dtype != torch.int64 and idx_right.dtype != torch.int64 or len(idx_left) <3 or len(idx_right) <3 :
            return None
        if idx_left.max() >= len(graph_L.node_pos) or idx_right.max() >= len(graph_R.node_pos):
            print('idx error', protein_pair)
            return None
        add_noise = True
        if add_noise:
            std_dev = 0.2
            s_std_dev = 0.02
            noise_L = torch.randn_like(graph_L.node_pos) * std_dev
            noise_R = torch.randn_like(graph_R.node_pos) * std_dev
            graph_L.node_pos += noise_L
            graph_R.node_pos += noise_R
            s_noise_L = torch.randn_like(surface_L.verts) * s_std_dev
            s_noise_R = torch.randn_like(surface_R.verts) * s_std_dev
            surface_L.verts += s_noise_L
            surface_R.verts += s_noise_R
        train_surface=True
        use_balance = False 
        if train_surface:
            
            distmat= torch.cdist(surface_L.verts,surface_R.verts)
            s_idxL,s_idxR = torch.where(distmat<2.0)
            s_site_idxL =s_idxL.unique()
            s_site_idxR =s_idxR.unique()
            s_denseR = torch.zeros(len(surface_R.verts))
            s_denseL = torch.zeros(len(surface_L.verts))
            s_denseL[s_site_idxL] = 1 
            s_denseR[s_site_idxR] = 1
            s_site_idxL_neg = torch.where(s_denseL==0)[0]
            s_site_idxR_neg = torch.where(s_denseR==0)[0]

            S_dense_pair=torch.zeros([len(surface_L.verts),len(surface_R.verts)])
            S_dense_pair[s_idxL,s_idxR]=1.0
            neg_s_idxL,neg_s_idxR = torch.where(S_dense_pair==0)                
            if use_balance:

                s_site_neg_sampleL= s_site_idxL_neg[torch.randperm(len(s_site_idxL_neg))[:len(s_site_idxL)]]
                s_site_neg_sampleR= s_site_idxR_neg[torch.randperm(len(s_site_idxR_neg))[:len(s_site_idxR)]]
                s_site_idxL_sample = torch.cat([s_site_idxL,s_site_neg_sampleL])
                s_site_idxR_sample = torch.cat([s_site_idxR,s_site_neg_sampleR])
                s_site_label_L = torch.cat([torch.ones(len(s_site_idxL)),torch.zeros(len(s_site_neg_sampleL))])
                s_site_label_R = torch.cat([torch.ones(len(s_site_idxR)),torch.zeros(len(s_site_neg_sampleR))])

                neg_sample = torch.randperm(len(neg_s_idxL))[:len(s_idxL)]
                neg_s_idxL_sample = neg_s_idxL[neg_sample]
                neg_s_idxR_sample = neg_s_idxR[neg_sample]
                sidx_left = torch.cat([s_idxL,neg_s_idxL_sample])
                sidx_right = torch.cat([s_idxR,neg_s_idxR_sample])
                s_label = torch.cat([torch.ones(len(s_idxL)),torch.zeros(len(neg_s_idxL_sample))])

            else:
                site_neg_num = min(1*len(s_site_idxL),len(s_site_idxL_neg),1*len(s_site_idxR),len(s_site_idxR_neg)) # 200
                s_site_neg_sampleL= s_site_idxL_neg[torch.randperm(len(s_site_idxL_neg))[:site_neg_num]]
                s_site_neg_sampleR= s_site_idxR_neg[torch.randperm(len(s_site_idxR_neg))[:site_neg_num]]
                s_site_idxL_sample = torch.cat([s_site_idxL,s_site_neg_sampleL])
                s_site_idxR_sample = torch.cat([s_site_idxR,s_site_neg_sampleR])
                s_site_label_L = torch.cat([torch.ones(len(s_site_idxL)),torch.zeros(len(s_site_neg_sampleL))])
                s_site_label_R = torch.cat([torch.ones(len(s_site_idxR)),torch.zeros(len(s_site_neg_sampleR))])

                neg_num = min(200*len(s_idxL),len(neg_s_idxL))
                neg_sample = torch.randperm(len(neg_s_idxL))[:neg_num]
                neg_s_idxL_sample = neg_s_idxL[neg_sample]
                neg_s_idxR_sample = neg_s_idxR[neg_sample]
                sidx_left = torch.cat([s_idxL,neg_s_idxL_sample])
                sidx_right = torch.cat([s_idxR,neg_s_idxR_sample])
                s_label = torch.cat([torch.ones(len(s_idxL)),torch.zeros(len(neg_s_idxL_sample))])
            item = Data(surface_1=surface_L, graph_1=graph_L, surface_2=surface_R, graph_2=graph_R, idx_left=idx_left,
                        idx_right=idx_right, label_r=label_r,label_l=label_l, g1_len=graph_L.node_pos.shape[0],
                        g2_len=graph_R.node_pos.shape[0],idx_left_pair=idx_left_pair,idx_right_pair=idx_right_pair,labels_pair=labels_pair,id=protein_pair,s1_len = len(surface_L.verts),s2_len=len(surface_R.verts),s_idx_left=sidx_left,s_idx_right=sidx_right,s_label=s_label,s_site_idxL=s_site_idxL_sample,s_site_idxR=s_site_idxR_sample,s_site_label_L=s_site_label_L,s_site_label_R=s_site_label_R)
            
        else:  
            item = Data(surface_1=surface_L, graph_1=graph_L, surface_2=surface_R, graph_2=graph_R, idx_left=idx_left,
                        idx_right=idx_right, label_r=label_r,label_l=label_l, g1_len=graph_L.node_pos.shape[0],
                        g2_len=graph_R.node_pos.shape[0],idx_left_pair=idx_left_pair,idx_right_pair=idx_right_pair,labels_pair=labels_pair,id=protein_pair)
        return item
    
class PINDERDataset_test_all_site(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, mode,neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1,pdb_type='holo'):
        self.data_dir= data_dir
        self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster.parquet')) # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
        self.index= self.index[self.index.split==mode]
        # metadata= pd.read_parquet(data_dir+'/processed_pinder_metadata.parquet')
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble
        self.pdb_type= pdb_type
    def __len__(self):
        return len(self.index)

    def _num_to_use(self, num_pos, num_neg):
        """
        Depending on the number of pos and neg of the system, we might want to use
            different amounts of positive or negative coordinates.

        :param num_pos:
        :param num_neg:
        :return:
        """

        if self.neg_to_pos_ratio == -1:
            num_pos_to_use, num_neg_to_use = num_pos, num_neg
        else:
            num_pos_to_use = min(num_pos, num_neg / self.neg_to_pos_ratio)
            if self.max_pos_regions_per_ensemble != -1:
                num_pos_to_use = min(num_pos_to_use, self.max_pos_regions_per_ensemble)
            num_neg_to_use = num_pos_to_use * self.neg_to_pos_ratio
        num_pos_to_use = int(math.ceil(num_pos_to_use))
        num_neg_to_use = int(math.ceil(num_neg_to_use))
        return num_pos_to_use, num_neg_to_use

    def __getitem__(self, idx):
        protein_pair = self.index.iloc[idx]['id']
        pdb_R = self.index.iloc[idx]['holo_R_pdb']
        pdb_L = self.index.iloc[idx]['holo_L_pdb']
        if os.path.exists(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface/'+protein_pair+'_interface.pt')):
            interface = torch.load(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface/'+protein_pair+'_interface.pt'))
        else:
            return None
        idx_R_pos = interface[pdb_R+'_mapping'][0]
        idx_L_pos = interface[pdb_L+'_mapping'][0]
        pos_pair= torch.from_numpy(interface['pair_mapping'])
        pos_pair_L = pos_pair[:,0]
        pos_pair_R = pos_pair[:,1]
        if self.pdb_type == 'holo':
            graph_R= self.graph_loader.load(pdb_R[:-4])
            graph_L=  self.graph_loader.load(pdb_L[:-4])
            surface_R =self.surface_loader.load(pdb_R[:-4])
            surface_L =self.surface_loader.load(pdb_L[:-4])
        elif self.pdb_type == 'af':
            af_R = self.index.iloc[idx]['predicted_R_pdb']
            af_L = self.index.iloc[idx]['predicted_L_pdb']
            graph_R= self.graph_loader.load(af_R[:-4])
            graph_L=  self.graph_loader.load(af_L[:-4])
            surface_R =self.surface_loader.load(af_R[:-4])
            surface_L =self.surface_loader.load(af_L[:-4]) 

        if surface_L is None or surface_R is None or graph_L is None or graph_R is None:
            return None
        if graph_L.node_len < 20 or graph_R.node_len < 20 or surface_L.n_verts < 20 or surface_R.n_verts < 20:
            return None
        if max(idx_R_pos) >= graph_R.node_len or max(idx_L_pos) >= graph_L.node_len:
            return None
        denseR = np.zeros(len(graph_R.node_pos))
        denseL = np.zeros(len(graph_L.node_pos))
        # if len(idx_R_pos)< 3 or len(idx_L_pos)< 3:
        #     return None
        denseL[idx_L_pos] = 1 
        denseR[idx_R_pos] = 1
        idx_L_neg = np.where(denseL==0)[0]
        idx_R_neg = np.where(denseR==0)[0]
        
        #num_to_use = min(min(len(idx_R_pos),len(idx_L_pos)),min(len(idx_R_neg),len(idx_L_neg)))
        # pos_array_sampledL = torch.from_numpy(idx_L_pos[np.random.choice(len(idx_L_pos), size=num_to_use, replace=False)])
        # pos_array_sampledR = torch.from_numpy(idx_R_pos[np.random.choice(len(idx_R_pos), size=num_to_use, replace=False)])
        # neg_array_sampledL = torch.from_numpy(idx_L_neg[np.random.choice(len(idx_L_neg), size=num_to_use, replace=False)])
        # neg_array_sampledR = torch.from_numpy(idx_R_neg[np.random.choice(len(idx_R_neg), size=num_to_use, replace=False)])
        pos_array_sampledL = torch.from_numpy(idx_L_pos)
        pos_array_sampledR = torch.from_numpy(idx_R_pos)
        neg_array_sampledL = torch.from_numpy(idx_L_neg)
        neg_array_sampledR = torch.from_numpy(idx_R_neg)
        
        idx_left = torch.arange(len(graph_L.node_pos))#torch.cat((pos_array_sampledL, neg_array_sampledL))
        idx_right = torch.arange(len(graph_R.node_pos))#torch.cat((pos_array_sampledR, neg_array_sampledR))
        label_l = torch.from_numpy(denseL)#torch.cat((torch.ones(len(pos_array_sampledL)),torch.zeros(len(neg_array_sampledL))))
        label_r =  torch.from_numpy(denseR)#torch.cat((torch.ones(len(pos_array_sampledR)),torch.zeros(len(neg_array_sampledR))))
        #prepare pair idx
        dense_pair=np.zeros([len(graph_L.node_pos),len(graph_R.node_pos)])
        dense_pair[pos_pair_L,pos_pair_R]=1.0
        pos_pair= np.vstack([pos_pair[:,0],pos_pair[:,1]])
        neg_pair = np.where(dense_pair==0)
        neg_pair = np.vstack([neg_pair[0],neg_pair[1]])
        num_pair_to_use = min(len(neg_pair[0]),len(pos_pair_L))
        pos_array_idx = np.random.choice(len(pos_pair_L), size=num_pair_to_use, replace=False)
        neg_array_idx = np.random.choice(len(neg_pair[0]), size=num_pair_to_use, replace=False)
        pos_pair_sampled_p = torch.from_numpy(pos_pair[:,pos_array_idx])
        neg_pair_sampled_p  =  torch.from_numpy(neg_pair[:, neg_array_idx])
        idx_left_pair = torch.cat((pos_pair_sampled_p[0], neg_pair_sampled_p[0]))
        idx_right_pair = torch.cat((pos_pair_sampled_p[1], neg_pair_sampled_p[1]))
        labels_pair = torch.cat((torch.ones(num_pair_to_use), torch.zeros(num_pair_to_use)))
        if idx_left.dtype != torch.int64 and idx_right.dtype != torch.int64 or len(idx_left) <3 or len(idx_right) <3 :
            return None
        if idx_left.max() >= len(graph_L.node_pos) or idx_right.max() >= len(graph_R.node_pos):
            print('idx error', protein_pair)
            return None
        add_noise = False
        if add_noise:
            std_dev = 0.2
            s_std_dev = 0.02
            noise_L = torch.randn_like(graph_L.node_pos) * std_dev
            noise_R = torch.randn_like(graph_R.node_pos) * std_dev
            graph_L.node_pos += noise_L
            graph_R.node_pos += noise_R
            s_noise_L = torch.randn_like(surface_L.verts) * s_std_dev
            s_noise_R = torch.randn_like(surface_R.verts) * s_std_dev
            surface_L.verts += s_noise_L
            surface_R.verts += s_noise_R
        train_surface=True
        use_balance = False 
        if train_surface:
            distmat= torch.cdist(surface_L.verts,surface_R.verts)
            s_idxL,s_idxR = torch.where(distmat<2.0)
            s_site_idxL =s_idxL.unique()
            s_site_idxR =s_idxR.unique()
            s_denseR = torch.zeros(len(surface_R.verts))
            s_denseL = torch.zeros(len(surface_L.verts))
            s_denseL[s_site_idxL] = 1 
            s_denseR[s_site_idxR] = 1
            s_site_idxL_neg = torch.where(s_denseL==0)[0]
            s_site_idxR_neg = torch.where(s_denseR==0)[0]

            S_dense_pair=torch.zeros([len(surface_L.verts),len(surface_R.verts)])
            S_dense_pair[s_idxL,s_idxR]=1.0
            neg_s_idxL,neg_s_idxR = torch.where(S_dense_pair==0)                
            if use_balance:

                s_site_neg_sampleL= s_site_idxL_neg[torch.randperm(len(s_site_idxL_neg))[:len(s_site_idxL)]]
                s_site_neg_sampleR= s_site_idxR_neg[torch.randperm(len(s_site_idxR_neg))[:len(s_site_idxR)]]
                s_site_idxL_sample = torch.cat([s_site_idxL,s_site_neg_sampleL])
                s_site_idxR_sample = torch.cat([s_site_idxR,s_site_neg_sampleR])
                s_site_label_L = torch.cat([torch.ones(len(s_site_idxL)),torch.zeros(len(s_site_neg_sampleL))])
                s_site_label_R = torch.cat([torch.ones(len(s_site_idxR)),torch.zeros(len(s_site_neg_sampleR))])

                neg_sample = torch.randperm(len(neg_s_idxL))[:len(s_idxL)]
                neg_s_idxL_sample = neg_s_idxL[neg_sample]
                neg_s_idxR_sample = neg_s_idxR[neg_sample]
                sidx_left = torch.cat([s_idxL,neg_s_idxL_sample])
                sidx_right = torch.cat([s_idxR,neg_s_idxR_sample])
                s_label = torch.cat([torch.ones(len(s_idxL)),torch.zeros(len(neg_s_idxL_sample))])

            else:
                s_site_idxL_sample = torch.arange(len(surface_L.verts))
                s_site_idxR_sample = torch.arange(len(surface_R.verts))
                s_site_label_L = s_denseL
                s_site_label_R = s_denseR

                neg_num = min(200*len(s_idxL),len(neg_s_idxL))
                neg_sample = torch.randperm(len(neg_s_idxL))[:neg_num]
                neg_s_idxL_sample = neg_s_idxL[neg_sample]
                neg_s_idxR_sample = neg_s_idxR[neg_sample]
                sidx_left = torch.cat([s_idxL,neg_s_idxL_sample])
                sidx_right = torch.cat([s_idxR,neg_s_idxR_sample])
                s_label = torch.cat([torch.ones(len(s_idxL)),torch.zeros(len(neg_s_idxL_sample))])
            item = Data(surface_1=surface_L, graph_1=graph_L, surface_2=surface_R, graph_2=graph_R, idx_left=idx_left,
                        idx_right=idx_right, label_r=label_r,label_l=label_l, g1_len=graph_L.node_pos.shape[0],
                        g2_len=graph_R.node_pos.shape[0],idx_left_pair=idx_left_pair,idx_right_pair=idx_right_pair,labels_pair=labels_pair,id=protein_pair,s1_len = len(surface_L.verts),s2_len=len(surface_R.verts),s_idx_left=sidx_left,s_idx_right=sidx_right,s_label=s_label,s_site_idxL=s_site_idxL_sample,s_site_idxR=s_site_idxR_sample,s_site_label_L=s_site_label_L,s_site_label_R=s_site_label_R)
            
        else:  
            item = Data(surface_1=surface_L, graph_1=graph_L, surface_2=surface_R, graph_2=graph_R, idx_left=idx_left,
                        idx_right=idx_right, label_r=label_r,label_l=label_l, g1_len=graph_L.node_pos.shape[0],
                        g2_len=graph_R.node_pos.shape[0],idx_left_pair=idx_left_pair,idx_right_pair=idx_right_pair,labels_pair=labels_pair,id=protein_pair)
        return item


class PINDERDataset_Af_Apo(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, mode,pdbtype,neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.data_dir= data_dir
        self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster.parquet')) # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
        self.index= self.index[self.index.split==mode]
        self.pdbtype = pdbtype
        if self.pdbtype=='apo':
            self.index= self.index[(self.index['apo_R']==True) & (self.index['apo_L']==True)]
        elif self.pdbtype=='af':
            self.index= self.index[(self.index['predicted_R']==True) & (self.index['predicted_L']==True)] 

        # metadata= pd.read_parquet(data_dir+'/processed_pinder_metadata.parquet')
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble

    def __len__(self):
        return len(self.index)

    def _num_to_use(self, num_pos, num_neg):
        """
        Depending on the number of pos and neg of the system, we might want to use
            different amounts of positive or negative coordinates.

        :param num_pos:
        :param num_neg:
        :return:
        """

        if self.neg_to_pos_ratio == -1:
            num_pos_to_use, num_neg_to_use = num_pos, num_neg
        else:
            num_pos_to_use = min(num_pos, num_neg / self.neg_to_pos_ratio)
            if self.max_pos_regions_per_ensemble != -1:
                num_pos_to_use = min(num_pos_to_use, self.max_pos_regions_per_ensemble)
            num_neg_to_use = num_pos_to_use * self.neg_to_pos_ratio
        num_pos_to_use = int(math.ceil(num_pos_to_use))
        num_neg_to_use = int(math.ceil(num_neg_to_use))
        return num_pos_to_use, num_neg_to_use

    def __getitem__(self, idx):
        protein_pair = self.index.iloc[idx]['id']
        if self.pdbtype=='holo':
            pdb_R = self.index.iloc[idx]['holo_R_pdb']
            pdb_L = self.index.iloc[idx]['holo_L_pdb']
        elif self.pdbtype=='apo':
            pdb_R = self.index.iloc[idx]['apo_R_pdb']
            pdb_L = self.index.iloc[idx]['apo_L_pdb']
        elif self.pdbtype=='af':
            pdb_R = self.index.iloc[idx]['predicted_R_pdb']
            pdb_L = self.index.iloc[idx]['predicted_L_pdb']
        if os.path.exists(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface/'+protein_pair+'_interface.pt')):
            if self.pdbtype=='holo':
                interface = torch.load(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface/'+protein_pair+'_interface.pt'))
            elif self.pdbtype=='apo':
                if os.path.exists(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface_apo_af/'+protein_pair+'_Apo_interface.pt')):
                    interface = torch.load(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface_apo_af/'+protein_pair+'_Apo_interface.pt'))
                else:
                    return None
            elif self.pdbtype=='af':
                if os.path.exists(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface_apo_af/'+protein_pair+'_AF_interface.pt')):
                    interface = torch.load(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface_apo_af/'+protein_pair+'_AF_interface.pt'))
                else:
                    return None
        else:
            return None
        idx_R_pos = np.array(interface[pdb_R+'_mapping'])#[0]
        idx_L_pos = np.array(interface[pdb_L+'_mapping'])#[0]
        pos_pair= torch.from_numpy(interface['pair_mapping'])
        if len(pos_pair)<3:
            return None
        pos_pair_L = pos_pair[:,0]
        pos_pair_R = pos_pair[:,1]
        graph_R= self.graph_loader.load(pdb_R[:-4])
        graph_L=  self.graph_loader.load(pdb_L[:-4])
        surface_R =self.surface_loader.load(pdb_R[:-4])
        surface_L =self.surface_loader.load(pdb_L[:-4])

        if surface_L is None or surface_R is None or graph_L is None or graph_R is None:
            return None
        # if graph_L.node_len < 20 or graph_R.node_len < 20 or surface_L.n_verts < 20 or surface_R.n_verts < 20:
        #     return None
        denseR = np.zeros(len(graph_R.node_pos))
        denseL = np.zeros(len(graph_L.node_pos))
        # if len(idx_R_pos)< 3 or len(idx_L_pos)< 3:
        #     return None
        denseL[idx_L_pos] = 1 
        denseR[idx_R_pos] = 1
        idx_L_neg = np.where(denseL==0)[0]
        idx_R_neg = np.where(denseR==0)[0]
        
        #num_to_use = min(min(len(idx_R_pos),len(idx_L_pos)),min(len(idx_R_neg),len(idx_L_neg)))
        # pos_array_sampledL = torch.from_numpy(idx_L_pos[np.random.choice(len(idx_L_pos), size=num_to_use, replace=False)])
        # pos_array_sampledR = torch.from_numpy(idx_R_pos[np.random.choice(len(idx_R_pos), size=num_to_use, replace=False)])
        # neg_array_sampledL = torch.from_numpy(idx_L_neg[np.random.choice(len(idx_L_neg), size=num_to_use, replace=False)])
        # neg_array_sampledR = torch.from_numpy(idx_R_neg[np.random.choice(len(idx_R_neg), size=num_to_use, replace=False)])
        # import pdb
        # pdb.set_trace() 
        # print(idx_L_pos)
        pos_array_sampledL = torch.from_numpy(idx_L_pos)
        pos_array_sampledR = torch.from_numpy(idx_R_pos)
        neg_array_sampledL = torch.from_numpy(idx_L_neg)
        neg_array_sampledR = torch.from_numpy(idx_R_neg)
        
        idx_left = torch.cat((pos_array_sampledL, neg_array_sampledL))
        idx_right = torch.cat((pos_array_sampledR, neg_array_sampledR))
        label_l = torch.cat((torch.ones(len(pos_array_sampledL)),torch.zeros(len(neg_array_sampledL))))
        label_r = torch.cat((torch.ones(len(pos_array_sampledR)),torch.zeros(len(neg_array_sampledR))))
        #prepare pair idx
        dense_pair=np.zeros([len(graph_L.node_pos),len(graph_R.node_pos)])
        dense_pair[pos_pair_L,pos_pair_R]=1.0
        pos_pair= np.vstack([pos_pair[:,0],pos_pair[:,1]])
        neg_pair = np.where(dense_pair==0)
        neg_pair = np.vstack([neg_pair[0],neg_pair[1]])
        num_pair_to_use = min(len(neg_pair[0]),len(pos_pair_L))
        pos_array_idx = np.random.choice(len(pos_pair_L), size=num_pair_to_use, replace=False)
        neg_array_idx = np.random.choice(len(neg_pair[0]), size=num_pair_to_use, replace=False)
        pos_pair_sampled_p = torch.from_numpy(pos_pair[:,pos_array_idx])
        neg_pair_sampled_p  =  torch.from_numpy(neg_pair[:, neg_array_idx])
        idx_left_pair = torch.cat((pos_pair_sampled_p[0], neg_pair_sampled_p[0]))
        idx_right_pair = torch.cat((pos_pair_sampled_p[1], neg_pair_sampled_p[1]))
        labels_pair = torch.cat((torch.ones(num_pair_to_use), torch.zeros(num_pair_to_use)))
        if idx_left.dtype != torch.int64 and idx_right.dtype != torch.int64 or len(idx_left) <3 or len(idx_right) <3 :
            return None
        if idx_left.max() >= len(graph_L.node_pos) or idx_right.max() >= len(graph_R.node_pos) or max(idx_right_pair)>= len(graph_R.node_pos) or max(idx_left_pair)>= len(graph_L.node_pos):
            print('idx error', protein_pair)
            return None
        item = Data(surface_1=surface_L, graph_1=graph_L, surface_2=surface_R, graph_2=graph_R, idx_left=idx_left,
                        idx_right=idx_right, label_r=label_r,label_l=label_l, g1_len=graph_L.node_pos.shape[0],
                        g2_len=graph_R.node_pos.shape[0],idx_left_pair=idx_left_pair,idx_right_pair=idx_right_pair,labels_pair=labels_pair,id=protein_pair)
        return item

class PINDERDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.surface_loader = SurfaceLoader(self.cfg.cfg_surface)
        self.graph_loader = GraphLoader(self.cfg.cfg_graph)
        

        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        train_dataset_temp = PINDERDataset(self.data_dir, self.surface_loader, self.graph_loader,mode='val')
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        dataset = PINDERDataset(self.data_dir, self.surface_loader, self.graph_loader,mode='train')
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = PINDERDataset(self.data_dir, self.surface_loader, self.graph_loader,mode='val')
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='test')
        # dataset = PINDERDataset(self.data_dir, self.surface_loader, self.graph_loader,mode='test')
        # dataset = PINDERDataset(self.data_dir, self.surface_loader, self.graph_loader,mode='test')
        # dataset =  PINDERDataset_Af_Apo( self.data_dir, self.surface_loader, self.graph_loader,mode='test',pdbtype='apo',neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1)
        return DataLoader(dataset, shuffle=False, **self.loader_args)

class PINDERDataModule_predict(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.surface_loader = SurfaceLoader(self.cfg.cfg_surface)
        self.graph_loader = GraphLoader(self.cfg.cfg_graph)
        

        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        # train_dataset_temp = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='val')
        # update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        dataset = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='train')
        return DataLoader(dataset, shuffle=True, **self.loader_args)

    def val_dataloader(self):
        dataset = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='val')
        return DataLoader(dataset, shuffle=True, **self.loader_args)

    def test_dataloader(self):
        # dataset = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='test',pdb_type='holo')
        dataset= DB5Dataset(self.data_dir, self.surface_loader, self.graph_loader)
        return DataLoader(dataset, shuffle=True, **self.loader_args)
    
from torch.utils.data import DistributedSampler
class PIPsiteDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        data_dir = cfg.data_dir
        self.systems = []
        self.surface_loaders = []
        self.graph_loaders = []
        self.cfg = cfg
        for mode in ['train', 'val', 'test']:
            # for mode in ['test'] * 3:
            self.systems.append(os.path.join(data_dir, mode))
            self.surface_loaders.append(SurfaceLoaderPIP(self.cfg.cfg_surface, mode=mode))
            self.graph_loaders.append(GraphLoaderPIP(self.cfg.cfg_graph, mode=mode))

        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        train_dataset_temp = PIPsiteDataset(self.systems[0], self.surface_loaders[0], self.graph_loaders[0])
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        dataset = PIPsiteDataset(self.systems[0], self.surface_loaders[0], self.graph_loaders[0])
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = PIPsiteDataset(self.systems[1], self.surface_loaders[1], self.graph_loaders[1])
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = PIPsiteDataset(self.systems[2], self.surface_loaders[2], self.graph_loaders[2],
                             max_pos_regions_per_ensemble=5)
        return DataLoader(dataset, shuffle=False, **self.loader_args)



class DB5Dataset(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder):
        self.data_dir= data_dir
        self.index= []
        for i in os.listdir(self.data_dir) :
            if i[0:2]!='._' and '.pdb' in i:
                self.index.append(i)
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pdb_L = self.index[idx]
        graph_L=  self.graph_loader.load(pdb_L[:-4])
        surface_L =self.surface_loader.load(pdb_L[:-4])
        if surface_L is None or graph_L is None:
            return None
        if graph_L.node_len < 20 or surface_L.n_verts < 20:
            return None

        denseL = np.zeros(len(graph_L.node_pos))
        idx_left = torch.arange(len(graph_L.node_pos))#torch.cat((pos_array_sampledL, neg_array_sampledL))

        # label_l = torch.from_numpy(denseL)#torch.cat((torch.ones(len(pos_array_sampledL)),torch.zeros(len(neg_array_sampledL))))
        s_site_idxL_sample = torch.arange(len(surface_L.verts))
        
        item = Data(surface_1=surface_L, graph_1=graph_L, idx_left=idx_left,
                    g1_len=graph_L.node_pos.shape[0],s1_len = len(surface_L.verts),s_site_idxL=s_site_idxL_sample,id=pdb_L[:-4])
            
        return item


####################################
####################################
##### add rfppi dataset here #######
####################################
####################################
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data
from collections import Counter
import os
import math

class BalancedPPIDataset(Dataset):
    def __init__(self, positive_data, negative_data, dataset_type='pdb', 
                 surface_builder=None, graph_builder=None, 
                 interface_dir='./interface',
                 add_noise=True, train_surface=True, pos_only=True):
        """
        Balanced PPI Dataset for both positive and negative samples
        
        Args:
            positive_data: DataFrame with positive samples
            negative_data: DataFrame with negative samples
            dataset_type: 'pdb' or 'afdb'
            surface_builder: Surface loader object
            graph_builder: Graph loader object
            interface_dir: Directory containing interface .pt files
            pdb_dir: Directory containing PDB structures
            add_noise: Whether to add Gaussian noise to coordinates
            train_surface: Whether to include surface training data
        """
        self.dataset_type = dataset_type
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.interface_dir = interface_dir
        self.add_noise = add_noise
        self.train_surface = train_surface
        
        # Prepare data with labels
        self.positive_data = positive_data.copy()
        self.negative_data = negative_data.copy()
        self.positive_data['label'] = 1
        self.negative_data['label'] = 0
        
        # Standardize column names
        if dataset_type == 'pdb':
            self.positive_data['id'] = self.positive_data['CHAIN1:CHAIN2']
            self.negative_data['id'] = self.negative_data['CHAIN1:CHAIN2']
            self.cluster_col = 'CLUSTER'
        else:  # afdb
            self.positive_data['id'] = self.positive_data['PAIRID']
            self.negative_data['id'] = self.negative_data['PAIRID']
            self.cluster_col = 'CLUSTER'
        
        # Combine all data
        if pos_only:
            self.all_data = self.positive_data
        else:
            self.all_data = pd.concat([self.positive_data, self.negative_data], 
                                   ignore_index=True)

        
        # Compute sampling weights for balanced sampling
        print('start computing weight')
        self._compute_sampling_weights()
    
    # def _compute_sampling_weights(self):
    #     """
    #     Compute sampling weights to balance:
    #     1. Positive vs negative samples (50:50)
    #     2. Clusters within each class
    #     """
    #     # Step 1: Balance positive and negative classes
    #     pos_mask = self.all_data['label'] == 1
    #     neg_mask = self.all_data['label'] == 0
        
    #     n_pos = pos_mask.sum()
    #     n_neg = neg_mask.sum()
        
    #     # Equal weight to positive and negative classes
    #     class_weights = torch.zeros(len(self.all_data), dtype=torch.float32)
    #     class_weights[pos_mask] = 0.5 / n_pos
    #     class_weights[neg_mask] = 0.5 / n_neg
        
    #     # Step 2: Balance clusters within each class
    #     cluster_weights = torch.ones(len(self.all_data), dtype=torch.float32)
        
    #     for label in [0, 1]:
    #         label_mask = self.all_data['label'] == label
    #         label_data = self.all_data[label_mask]
            
    #         # Count samples per cluster
    #         cluster_counts = label_data[self.cluster_col].value_counts().to_dict()
            
    #         # Assign inverse frequency weight to each cluster
    #         for cluster_id, count in cluster_counts.items():
    #             cluster_mask = (self.all_data[self.cluster_col] == cluster_id) & label_mask
    #             cluster_weights[cluster_mask] = 1.0 / count
            
    #         # Normalize cluster weights within each class
    #         cluster_weights[label_mask] = (
    #             cluster_weights[label_mask] / cluster_weights[label_mask].sum()
    #         )
        
    #     # Combine weights: class balance * cluster balance
    #     self.sample_weights = class_weights * cluster_weights * len(self.all_data)
        
    #     print(f"Dataset size: {len(self.all_data)} "
    #           f"(Pos: {n_pos}, Neg: {n_neg})")
    #     print(f"Weight range: [{self.sample_weights.min():.6f}, "
    #           f"{self.sample_weights.max():.6f}]")
    def _compute_sampling_weights(self):
        """
        Compute sampling weights that balance clusters within each class.
        Pos/neg balance is NOT enforced (per user request).
        """
    
        labels = self.all_data["label"].to_numpy()
        clusters = self.all_data[self.cluster_col].to_numpy()
        N = len(labels)
    
        # Each class treated separately
        sample_weights = torch.zeros(N, dtype=torch.float32)
    
        for label in [0, 1]:
            class_mask = labels == label
            class_indices = np.where(class_mask)[0]
    
            if len(class_indices) == 0:
                continue
    
            # Clusters for one class
            class_clusters = clusters[class_mask]
    
            # Count cluster sizes
            # value_counts is fast in pandas
            cluster_counts = pd.Series(class_clusters).value_counts()
    
            # Map cluster_id -> inverse frequency
            inv_cluster_freq = 1.0 / cluster_counts
    
            # Assign cluster weight to each sample
            cluster_w = np.array([
                inv_cluster_freq[c] for c in class_clusters
            ], dtype=np.float32)
    
            # Normalize inside the class
            cluster_w /= cluster_w.sum()
    
            sample_weights[class_indices] = torch.from_numpy(cluster_w)
    
        self.sample_weights = sample_weights
    
        print(f"Dataset size: {N}, Pos: {(labels==1).sum()}, Neg: {(labels==0).sum()}")
        print(f"Weight range: [{self.sample_weights.min():.6f}, {self.sample_weights.max():.6f}]")    
    
    def _load_interface(self, sample_id):
        """Load interface data for positive samples"""
        interface_path = os.path.join(
            self.interface_dir, 
            f"{sample_id.replace(':', '-')}_interface.pt"
        )
        
        if os.path.exists(interface_path):
            return torch.load(interface_path)
        return None
    
    def _prepare_positive_sample(self, graph_L, graph_R, surface_L, surface_R, 
                                 interface, pdb1_name, pdb2_name):
        """Prepare training data for positive (interacting) pairs"""
        
        # Get interface residue indices
        idx_L_pos = interface[pdb1_name + '_mapping'][0]
        idx_R_pos = interface[pdb2_name + '_mapping'][0]
        pos_pair = interface['pair_mapping'] #torch.from_numpy(interface['pair_mapping']).numpy()
        
        # Validate indices
        if (idx_R_pos.max() >= graph_R.node_len or 
            idx_L_pos.max() >= graph_L.node_len):
            return None
        
        # Create dense labels for all residues
        dense_L = np.zeros(len(graph_L.node_pos))
        dense_R = np.zeros(len(graph_R.node_pos))
        dense_L[idx_L_pos] = 1
        dense_R[idx_R_pos] = 1
        
        # Get negative residues
        idx_L_neg = np.where(dense_L == 0)[0]
        idx_R_neg = np.where(dense_R == 0)[0]
        
        # Prepare residue-level data
        idx_left = torch.cat([
            torch.from_numpy(idx_L_pos),
            torch.from_numpy(idx_L_neg)
        ])
        idx_right = torch.cat([
            torch.from_numpy(idx_R_pos),
            torch.from_numpy(idx_R_neg)
        ])
        label_l = torch.cat([
            torch.ones(len(idx_L_pos)),
            torch.zeros(len(idx_L_neg))
        ])
        label_r = torch.cat([
            torch.ones(len(idx_R_pos)),
            torch.zeros(len(idx_R_neg))
        ])
        
        # Prepare pair-level data
        dense_pair = np.zeros([len(graph_L.node_pos), len(graph_R.node_pos)])
        dense_pair[pos_pair[:, 0], pos_pair[:, 1]] = 1.0
        
        neg_pair = np.where(dense_pair == 0)
        neg_pair = np.vstack([neg_pair[0], neg_pair[1]])
        
        num_pair = min(len(neg_pair[0]), len(pos_pair))
        pos_idx = np.random.choice(len(pos_pair), size=num_pair, replace=False)
        neg_idx = np.random.choice(len(neg_pair[0]), size=num_pair, replace=False)
        
        pos_pair_sampled = torch.from_numpy(pos_pair[pos_idx].T)
        neg_pair_sampled = torch.from_numpy(neg_pair[:, neg_idx])
        
        idx_left_pair = torch.cat([pos_pair_sampled[0], neg_pair_sampled[0]])
        idx_right_pair = torch.cat([pos_pair_sampled[1], neg_pair_sampled[1]])
        labels_pair = torch.cat([
            torch.ones(num_pair),
            torch.zeros(num_pair)
        ])
        
        result = {
            'idx_left': idx_left,
            'idx_right': idx_right,
            'label_l': label_l,
            'label_r': label_r,
            'idx_left_pair': idx_left_pair,
            'idx_right_pair': idx_right_pair,
            'labels_pair': labels_pair
        }
        
        # Add surface data if needed
        if self.train_surface:
            surface_data = self._prepare_surface_data(surface_L, surface_R)
            if surface_data is not None:
                result.update(surface_data)
        
        return result
    
    def _prepare_negative_sample(self, graph_L, graph_R, surface_L, surface_R):
        """Prepare training data for negative (non-interacting) pairs"""
        
        # All residues are negative
        idx_left = torch.arange(len(graph_L.node_pos))
        idx_right = torch.arange(len(graph_R.node_pos))
        label_l = torch.zeros(len(idx_left))
        label_r = torch.zeros(len(idx_right))
        
        # Sample negative pairs
        num_pairs = min(len(idx_left), len(idx_right), 1000)
        sampled_l = torch.randperm(len(idx_left))[:num_pairs]
        sampled_r = torch.randperm(len(idx_right))[:num_pairs]
        
        idx_left_pair = idx_left[sampled_l]
        idx_right_pair = idx_right[sampled_r]
        labels_pair = torch.zeros(num_pairs)
        
        result = {
            'idx_left': idx_left,
            'idx_right': idx_right,
            'label_l': label_l,
            'label_r': label_r,
            'idx_left_pair': idx_left_pair,
            'idx_right_pair': idx_right_pair,
            'labels_pair': labels_pair
        }
        
        # Add surface data if needed
        if self.train_surface:
            surface_data = self._prepare_surface_data(
                surface_L, surface_R, is_positive=False
            )
            if surface_data is not None:
                result.update(surface_data)
        
        return result
    
    def _prepare_surface_data(self, surface_L, surface_R, is_positive=True):
        """Prepare surface-level training data"""
        
        if is_positive:
            # Find interface vertices (distance < 2.0 Å)
            distmat = torch.cdist(surface_L.verts, surface_R.verts)
            s_idxL, s_idxR = torch.where(distmat < 2.0)
            
            s_site_idxL = s_idxL.unique()
            s_site_idxR = s_idxR.unique()
            
            # Get non-interface vertices
            s_denseL = torch.zeros(len(surface_L.verts))
            s_denseR = torch.zeros(len(surface_R.verts))
            s_denseL[s_site_idxL] = 1
            s_denseR[s_site_idxR] = 1
            
            s_site_idxL_neg = torch.where(s_denseL == 0)[0]
            s_site_idxR_neg = torch.where(s_denseR == 0)[0]
            
            # Sample negative sites
            site_neg_num = min(
                len(s_site_idxL),
                len(s_site_idxL_neg),
                len(s_site_idxR),
                len(s_site_idxR_neg)
            )
            
            s_site_neg_sampleL = s_site_idxL_neg[
                torch.randperm(len(s_site_idxL_neg))[:site_neg_num]
            ]
            s_site_neg_sampleR = s_site_idxR_neg[
                torch.randperm(len(s_site_idxR_neg))[:site_neg_num]
            ]
            
            # Prepare pair-level surface data
            S_dense_pair = torch.zeros([len(surface_L.verts), len(surface_R.verts)])
            S_dense_pair[s_idxL, s_idxR] = 1.0
            neg_s_idxL, neg_s_idxR = torch.where(S_dense_pair == 0)
            
            neg_num = min(200 * len(s_idxL), len(neg_s_idxL))
            neg_sample = torch.randperm(len(neg_s_idxL))[:neg_num]
            
        else:  # Negative sample
            # Sample random vertices as "sites"
            num_sample = min(len(surface_L.verts), len(surface_R.verts), 100)
            s_site_idxL = torch.randperm(len(surface_L.verts))[:num_sample]
            s_site_idxR = torch.randperm(len(surface_R.verts))[:num_sample]
            s_site_neg_sampleL = torch.randperm(len(surface_L.verts))[:num_sample]
            s_site_neg_sampleR = torch.randperm(len(surface_R.verts))[:num_sample]
            
            # All pairs are negative
            s_idxL = s_site_idxL[:min(50, len(s_site_idxL))]
            s_idxR = s_site_idxR[:min(50, len(s_site_idxR))]
            neg_s_idxL = s_site_neg_sampleL
            neg_s_idxR = s_site_neg_sampleR
            neg_sample = torch.arange(min(len(neg_s_idxL), 1000))
        
        return {
            's_site_idxL': torch.cat([s_site_idxL, s_site_neg_sampleL]),
            's_site_idxR': torch.cat([s_site_idxR, s_site_neg_sampleR]),
            's_site_label_L': torch.cat([
                torch.ones(len(s_site_idxL)) if is_positive else torch.zeros(len(s_site_idxL)),
                torch.zeros(len(s_site_neg_sampleL))
            ]),
            's_site_label_R': torch.cat([
                torch.ones(len(s_site_idxR)) if is_positive else torch.zeros(len(s_site_idxR)),
                torch.zeros(len(s_site_neg_sampleR))
            ]),
            's_idx_left': torch.cat([s_idxL, neg_s_idxL[neg_sample]]),
            's_idx_right': torch.cat([s_idxR, neg_s_idxR[neg_sample]]),
            's_label': torch.cat([
                torch.ones(len(s_idxL)) if is_positive else torch.zeros(len(s_idxL)),
                torch.zeros(len(neg_sample))
            ]),
            's1_len': len(surface_L.verts),
            's2_len': len(surface_R.verts)
        }
    
    def __len__(self):
        return len(self.all_data)
        
    def _get_pdb_name(self, sample_id, label):
        """Get PDB file paths for a sample"""
        if self.dataset_type == 'pdb':
            parts = sample_id.split(':')
            chain1, chain2 = parts[0], parts[1]
            pdb1 = f"{chain1}-{chain2}__{chain1}.pdb"
            pdb2 = f"{chain1}-{chain2}__{chain2}.pdb"
                
        else:  # afdb
            parts = sample_id.split(':')
            domain1, domain2 = parts[0], parts[1]
            
            pdb1 = f"{domain1}.pdb"
            pdb2 = f"{domain2}.pdb"
        
        return pdb1, pdb2
        
    def __getitem__(self, idx):
        row = self.all_data.iloc[idx]
        sample_id = row['id']
        label = row['label']
        
        
        # Get PDB paths
        pdb1_name, pdb2_name = self._get_pdb_name(sample_id, label)
        # Load structures
        try:
            graph_L = self.graph_loader.load(pdb1_name[:-4])
            graph_R = self.graph_loader.load(pdb2_name[:-4])
            surface_L = self.surface_loader.load(pdb1_name[:-4])
            surface_R = self.surface_loader.load(pdb2_name[:-4])
        except Exception as e:
            print(f"Error loading structures for {sample_id}: {e}")
            return None
        
        # Validate loaded data
        if (surface_L is None or surface_R is None or 
            graph_L is None or graph_R is None):
            return None
        
        if (graph_L.node_len < 20 or graph_R.node_len < 20 or 
            surface_L.n_verts < 20 or surface_R.n_verts < 20):
            return None
        
        # Add noise if enabled
        if self.add_noise:
            std_dev = 0.2
            s_std_dev = 0.02
            graph_L.node_pos += torch.randn_like(graph_L.node_pos) * std_dev
            graph_R.node_pos += torch.randn_like(graph_R.node_pos) * std_dev
            surface_L.verts += torch.randn_like(surface_L.verts) * s_std_dev
            surface_R.verts += torch.randn_like(surface_R.verts) * s_std_dev
        
        # Prepare sample based on label
        if label == 1:  # Positive sample
            interface = self._load_interface(sample_id)
            if interface is None:
                return None
            
            sample_data = self._prepare_positive_sample(
                graph_L, graph_R, surface_L, surface_R,
                interface, pdb1_name, pdb2_name
            )
        else:  # Negative sample
            sample_data = self._prepare_negative_sample(
                graph_L, graph_R, surface_L, surface_R
            )
        
        if sample_data is None:
            return None
        
        # Validate indices
        if (sample_data['idx_left'].dtype != torch.int64 or 
            sample_data['idx_right'].dtype != torch.int64 or
            len(sample_data['idx_left']) < 3 or 
            len(sample_data['idx_right']) < 3):
            return None
        
        if (sample_data['idx_left'].max() >= len(graph_L.node_pos) or 
            sample_data['idx_right'].max() >= len(graph_R.node_pos)):
            return None
        
        # Create Data object
        item = Data(
            surface_1=surface_L,
            graph_1=graph_L,
            surface_2=surface_R,
            graph_2=graph_R,
            g1_len=graph_L.node_pos.shape[0],
            g2_len=graph_R.node_pos.shape[0],
            id=sample_id,
            label=torch.tensor(label),
            **sample_data
        )
        
        return item


class CombinedPPIDataset(Dataset):
    """
    Combined dataset that merges PDB and AFDB datasets with balanced sampling
    """
    def __init__(self, pdb_dataset, afdb_dataset, pdb_weight=0.5):
        """
        Args:
            pdb_dataset: BalancedPPIDataset for PDB data
            afdb_dataset: BalancedPPIDataset for AFDB data
            pdb_weight: Weight for PDB dataset (0-1), AFDB gets (1-pdb_weight)
        """
        self.pdb_dataset = pdb_dataset
        self.afdb_dataset = afdb_dataset
        self.pdb_weight = pdb_weight
        
        self.pdb_len = len(pdb_dataset)
        self.afdb_len = len(afdb_dataset)
        self.total_len = self.pdb_len + self.afdb_len
        
        # Compute combined sampling weights
        self._compute_combined_weights()
    
    def _compute_combined_weights(self):
        """
        Combine weights from both datasets while maintaining:
        1. Balance between PDB and AFDB (controlled by pdb_weight)
        2. Balance within each dataset (pos/neg and clusters)
        """
        # Normalize weights within each dataset
        pdb_weights = self.pdb_dataset.sample_weights.clone()
        afdb_weights = self.afdb_dataset.sample_weights.clone()
        
        pdb_weights = pdb_weights / pdb_weights.sum()
        afdb_weights = afdb_weights / afdb_weights.sum()
        
        # Apply dataset-level weights
        pdb_weights = pdb_weights * self.pdb_weight
        afdb_weights = afdb_weights * (1 - self.pdb_weight)
        
        # Combine weights
        self.sample_weights = torch.cat([pdb_weights, afdb_weights])
        
        print(f"\n=== Combined Dataset Statistics ===")
        print(f"PDB samples: {self.pdb_len} (weight: {self.pdb_weight:.2f})")
        print(f"AFDB samples: {self.afdb_len} (weight: {1-self.pdb_weight:.2f})")
        print(f"Total samples: {self.total_len}")
        print(f"Combined weight sum: {self.sample_weights.sum():.6f}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < self.pdb_len:
            return self.pdb_dataset[idx]
        else:
            return self.afdb_dataset[idx - self.pdb_len]


def create_balanced_dataloader(positive_data, negative_data, dataset_type,
                                surface_builder, graph_builder,
                                interface_dir, 
                                batch_size=32, num_samples=None,
                                num_workers=4):
    """
    Create balanced DataLoader with weighted sampling
    
    Args:
        positive_data: DataFrame with positive samples
        negative_data: DataFrame with negative samples  
        dataset_type: 'pdb' or 'afdb'
        surface_builder: Surface loader
        graph_builder: Graph loader
        interface_dir: Interface files directory
        batch_size: Batch size
        num_samples: Number of samples per epoch (default: dataset size)
        num_workers: DataLoader workers
    
    Returns:
        dataloader, dataset
    """
    dataset = BalancedPPIDataset(
        positive_data=positive_data,
        negative_data=negative_data,
        dataset_type=dataset_type,
        surface_builder=surface_builder,
        graph_builder=graph_builder,
        interface_dir=interface_dir,
    )
    
    if num_samples is None:
        num_samples = len(dataset)
    
    # Create weighted sampler for balanced sampling
    sampler = WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    # Custom collate function to handle None values
    def collate_fn(batch):
        batch = AtomBatch.from_data_list(batch)
        # batch = [item for item in batch if item is not None]
        # if len(batch) == 0:
        #     return None
        return batch
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset


def create_combined_dataloader(pdb_positive_data, pdb_negative_data,
                                afdb_positive_data, afdb_negative_data,
                                surface_builder, graph_builder,
                                pdb_interface_dir, 
                                afdb_interface_dir,
                                batch_size=32, num_samples=None,
                                pdb_weight=0.5, num_workers=4,prefetch_factor=2):
    """
    Create combined DataLoader with both PDB and AFDB data
    
    Args:
        pdb_positive_data: PDB positive samples DataFrame
        pdb_negative_data: PDB negative samples DataFrame
        afdb_positive_data: AFDB positive samples DataFrame
        afdb_negative_data: AFDB negative samples DataFrame
        surface_builder: Surface loader
        graph_builder: Graph loader
        pdb_interface_dir: PDB interface files directory
        afdb_interface_dir: AFDB interface files directory
        afdb_pdb_dir: AFDB structure files directory
        batch_size: Batch size
        num_samples: Number of samples per epoch (default: combined dataset size)
        pdb_weight: Weight for PDB dataset (0-1), AFDB gets (1-pdb_weight)
        num_workers: DataLoader workers
    
    Returns:
        dataloader, combined_dataset
    """
    # Create individual datasets
    pdb_dataset = BalancedPPIDataset(
        positive_data=pdb_positive_data,
        negative_data=pdb_negative_data,
        dataset_type='pdb',
        surface_builder=surface_builder,
        graph_builder=graph_builder,
        interface_dir=pdb_interface_dir,
    )
    
    afdb_dataset = BalancedPPIDataset(
        positive_data=afdb_positive_data,
        negative_data=afdb_negative_data,
        dataset_type='afdb',
        surface_builder=surface_builder,
        graph_builder=graph_builder,
        interface_dir=afdb_interface_dir,
    )
    
    # Combine datasets
    combined_dataset = CombinedPPIDataset(
        pdb_dataset=pdb_dataset,
        afdb_dataset=afdb_dataset,
        pdb_weight=pdb_weight
    )
    
    if num_samples is None:
        num_samples = len(combined_dataset)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=combined_dataset.sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    # Custom collate function to handle None values
    def collate_fn(batch):
        batch = AtomBatch.from_data_list(batch)
        # batch = [item for item in batch if item is not None]
        # if len(batch) == 0:
        #     return None
        return batch
    
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
    )
    
    return dataloader, combined_dataset


def check_dataloader_balance(dataloader, num_batches=10):
    """Check the balance of labels, clusters, and datasets in sampled batches"""
    all_labels = []
    all_dataset_types = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches or batch is None:
            break
        
        for item in batch:
            all_labels.append(item.label.item())
            # Check if id contains PDB or AFDB pattern
            sample_id = item.id
            if ':' in sample_id and sample_id.count(':') == 1:
                # Simple heuristic: PDB uses chain notation, AFDB uses domain notation
                all_dataset_types.append('PDB' if '-' in sample_id.split(':')[0] else 'AFDB')
            else:
                all_dataset_types.append('Unknown')
    
    label_counts = Counter(all_labels)
    dataset_counts = Counter(all_dataset_types)
    
    print(f"\n=== Sampled {len(all_labels)} samples across {num_batches} batches ===")
    print(f"\nLabel Balance:")
    print(f"  Positive samples: {label_counts.get(1, 0)} "
          f"({100*label_counts.get(1, 0)/len(all_labels):.1f}%)")
    print(f"  Negative samples: {label_counts.get(0, 0)} "
          f"({100*label_counts.get(0, 0)/len(all_labels):.1f}%)")
    
    print(f"\nDataset Balance:")
    for dataset_type, count in dataset_counts.items():
        print(f"  {dataset_type}: {count} ({100*count/len(all_labels):.1f}%)")

class RFPPIDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.surface_loader = SurfaceLoader(self.cfg.cfg_surface)
        self.graph_loader = GraphLoader(self.cfg.cfg_graph)
        if not cfg.nohomo:
            self.pdb_pos = pd.read_csv(os.path.join(self.data_dir,'PDB_PPI/posi_all.csv'))
            self.pdb_neg = pd.read_csv(os.path.join(self.data_dir,'PDB_PPI/nega_all.csv'))
            self.afdb_pos = pd.read_csv(os.path.join(self.data_dir,'AFDB_DDI/posi_all.csv'))
            self.afdb_neg = pd.read_csv(os.path.join(self.data_dir,'AFDB_DDI/nega_all.csv'))
        else:
            self.pdb_pos = pd.read_csv(os.path.join(self.data_dir,'PDB_PPI/posi_nohomo.csv'))
            self.pdb_neg = pd.read_csv(os.path.join(self.data_dir,'PDB_PPI/nega_nohomo.csv'))
            self.afdb_pos = pd.read_csv(os.path.join(self.data_dir,'AFDB_DDI/posi_nohomo.csv'))
            self.afdb_neg = pd.read_csv(os.path.join(self.data_dir,'AFDB_DDI/nega_nohomo.csv'))
        self.pdb_pos_val = pd.read_csv(os.path.join(self.data_dir,'PDB_PPI/posi_val.csv'))   
        self.pdb_neg_val = pd.read_csv(os.path.join(self.data_dir,'PDB_PPI/nega_val.csv'))

        self.loader_args = {'num_workers': self.cfg.loader.num_workers,
                            'batch_size': self.cfg.loader.batch_size,
                            'pin_memory': self.cfg.loader.pin_memory,
                            'prefetch_factor': self.cfg.loader.prefetch_factor,
                            'collate_fn': lambda x: AtomBatch.from_data_list(x)}

        # Useful to create a Model of the right input dims
        train_dataset_temp = BalancedPPIDataset(
                                positive_data=self.pdb_pos,
                                negative_data=self.pdb_neg,
                                dataset_type='pdb',
                                surface_builder=self.surface_loader,
                                graph_builder=self.graph_loader,
                                interface_dir=os.path.join(self.data_dir,'interface'),
                            )
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        combined_dataloader, combined_dataset = create_combined_dataloader(
            pdb_positive_data=self.pdb_pos,
            pdb_negative_data=self.pdb_neg,
            afdb_positive_data=self.afdb_pos,
            afdb_negative_data=self.afdb_neg,
            surface_builder=self.surface_loader,
            graph_builder=self.graph_loader,
            pdb_interface_dir=os.path.join(self.data_dir,'interface'),
            afdb_interface_dir=os.path.join(self.data_dir,'interface'),
            batch_size=self.cfg.loader.batch_size,
            num_samples=900000,  # Sample 20k per epoch
            pdb_weight=0.3,     # 50% PDB, 50% AFDB
            num_workers=self.cfg.loader.num_workers,
            prefetch_factor = self.cfg.loader.prefetch_factor
        )
        return combined_dataloader

    def val_dataloader(self):
        dataset = BalancedPPIDataset(
                                positive_data=self.pdb_pos_val,
                                negative_data=self.pdb_neg_val,
                                dataset_type='pdb',
                                surface_builder=self.surface_loader,
                                graph_builder=self.graph_loader,
                                interface_dir=os.path.join(self.data_dir,'interface'),
                            )
        dataloader = DataLoader(
        dataset,
        batch_size = self.cfg.loader.batch_size,
        num_workers = self.cfg.loader.num_workers,
        pin_memory = True,
        collate_fn = lambda x: AtomBatch.from_data_list(x),
        prefetch_factor = self.cfg.loader.prefetch_factor,
        )
    
        return dataloader

    def test_dataloader(self):
        dataset = BalancedPPIDataset(
                                positive_data=self.pdb_pos_val,
                                negative_data=self.pdb_neg_val,
                                dataset_type='pdb',
                                surface_builder=self.surface_loader,
                                graph_builder=self.graph_loader,
                                interface_dir=os.path.join(self.data_dir,'interface'),
                            )
        dataloader = DataLoader(
        dataset,
        batch_size = self.cfg.loader.batch_size,
        num_workers = self.cfg.loader.num_workers,
        pin_memory = True,
        collate_fn = lambda x: AtomBatch.from_data_list(x),
        prefetch_factor = self.cfg.loader.prefetch_factor,
        )
    
        return dataloader


if __name__ == '__main__':
    pass
