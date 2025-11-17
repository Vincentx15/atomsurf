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
from atomsurf.utils.geometric_processing import atoms_to_points_normals
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
        self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster.parquet')) # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
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
        # import pdb
        # pdb.set_trace()
        # if os.path.exists(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface/'+protein_pair+'_interface.pt')):
        #     interface = torch.load(os.path.join('/work/lpdi/users/ymiao/code/pinderdata/interface/'+protein_pair+'_interface.pt'))
        # else:
        #     return None
        # idx_R_pos = interface[pdb_R+'_mapping'][0]
        # idx_L_pos = interface[pdb_L+'_mapping'][0]
        # pos_pair= torch.from_numpy(interface['pair_mapping'])
        # pos_pair_L = pos_pair[:,0]
        # pos_pair_R = pos_pair[:,1]

        graph_R= self.graph_loader.load(pdb_R[:-4])
        graph_L=  self.graph_loader.load(pdb_L[:-4])
        surface_R =self.surface_loader.load(pdb_R[:-4])
        surface_L =self.surface_loader.load(pdb_L[:-4])

        if surface_L is None or surface_R is None or graph_L is None or graph_R is None:
            return None
        if graph_L.node_len < 20 or graph_R.node_len < 20 or surface_L.n_verts < 20 or surface_R.n_verts < 20:
            return None

        graph_interface= torch.cdist(graph_L.node_pos,graph_R.node_pos)<3.5
        pos_pair= torch.argwhere(graph_interface)
        pos_pair_L = pos_pair[:,0]
        pos_pair_R = pos_pair[:,1]
        R_res= graph_R.res_map[pos_pair_R].unique()
        L_res= graph_L.res_map[pos_pair_L].unique()
        idx_R_pos = torch.where(torch.isin(graph_R.res_map, R_res))[0].numpy()
        idx_L_pos = torch.where(torch.isin(graph_L.res_map, L_res))[0].numpy()

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
                site_neg_num = min(200*len(s_site_idxL),len(s_site_idxL_neg),200*len(s_site_idxR),len(s_site_idxR_neg))
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

from atomsurf.network_utils.misc_arch.dmasif_utils.geometry_processing import curvatures
from torch_geometric.data import Data, Batch
class PINDERDataset_atomgraph_onfly(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, mode,neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1,add_noise=True):
        self.data_dir= data_dir
        self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster.parquet')) # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
        self.index= self.index[self.index.split==mode]
        # metadata= pd.read_parquet(data_dir+'/processed_pinder_metadata.parquet')
        self.surface_loader = surface_builder
        self.graph_loader = graph_builder
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble
        self.add_noise = add_noise
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
        import time
        t0 = time.time()
        """Optimized data loading with ~2-3x speedup."""
        
        # Load basic data
        protein_pair = self.index.iloc[idx]['id']
        pdb_R = self.index.iloc[idx]['holo_R_pdb']
        pdb_L = self.index.iloc[idx]['holo_L_pdb']
        
        graph_R = self.graph_loader.load(pdb_R[:-4])
        graph_L = self.graph_loader.load(pdb_L[:-4])
        
        if graph_L is None or graph_R is None:
            return None
        
        # Early validation to avoid unnecessary computation
        if graph_L.node_len < 20 or graph_R.node_len < 20:
            return None
        
        # === SURFACE COMPUTATION (Optimized GPU usage) ===
        tmp_device = torch.device('cuda')
        
        # Pre-allocate on GPU and add noise in one operation
        coord_L = graph_L.node_pos
        coord_R = graph_R.node_pos
        
        if self.add_noise:
            coord_L = coord_L + torch.randn_like(coord_L) * 0.5
            coord_R = coord_R + torch.randn_like(coord_R) * 0.5
        
        data_R = Data(coord=coord_R, atomtypes=graph_R.x[:, -12:], num_nodes=coord_R.size(0))
        data_L = Data(coord=coord_L, atomtypes=graph_L.x[:, -12:], num_nodes=coord_L.size(0))
        
        tmpbatch = Batch.from_data_list([data_R, data_L]).to(tmp_device)
        
        points, normals, batch_points = atoms_to_points_normals(
            tmpbatch.coord, tmpbatch.batch, distance=1.05, smoothness=0.5,
            resolution=2.5, nits=4, atomtypes=tmpbatch.atomtypes,
            sup_sampling=20, variance=0.1
        )
        
        P_curvatures = curvatures(
            points, triangles=None, normals=normals,
            scales=[1.0, 2.0, 3.0, 5.0, 10.0], batch=batch_points
        )
        
        # Move to CPU once and cleanup GPU memory
        mask_R = batch_points == 0
        mask_L = batch_points == 1
        
        surface_R = Data(
            verts=points[mask_R].cpu(),
            n_verts=mask_R.sum().item(),
            x=P_curvatures[mask_R].cpu(),
            num_nodes=mask_R.sum().item(),
            vnormals=normals[mask_R].cpu()
        )
        
        surface_L = Data(
            verts=points[mask_L].cpu(),
            n_verts=mask_L.sum().item(),
            x=P_curvatures[mask_L].cpu(),
            num_nodes=mask_L.sum().item(),
            vnormals=normals[mask_L].cpu()
        )
        
        del tmpbatch, points, normals, batch_points, P_curvatures, mask_R, mask_L
        torch.cuda.empty_cache()
        
        if surface_L.n_verts < 20 or surface_R.n_verts < 20:
            return None
        
        # === GRAPH INTERFACE COMPUTATION (Vectorized) ===
        # Use squared distance to avoid sqrt
        dist_matrix = torch.cdist(graph_L.node_pos, graph_R.node_pos)
        graph_interface = dist_matrix < 3.5
        pos_pair_L, pos_pair_R = torch.where(graph_interface)
        
        # Vectorized residue mapping
        R_res = graph_R.res_map[pos_pair_R].unique()
        L_res = graph_L.res_map[pos_pair_L].unique()
        
        # Use boolean indexing (faster than torch.isin for large arrays)
        mask_R_pos = torch.isin(graph_R.res_map, R_res)
        mask_L_pos = torch.isin(graph_L.res_map, L_res)
        
        idx_L_pos = torch.where(mask_L_pos)[0]
        idx_R_pos = torch.where(mask_R_pos)[0]
        
        # Compute negative indices efficiently
        idx_L_neg = torch.where(~mask_L_pos)[0]
        idx_R_neg = torch.where(~mask_R_pos)[0]
        
        # Concatenate labels
        idx_left = torch.cat([idx_L_pos, idx_L_neg])
        idx_right = torch.cat([idx_R_pos, idx_R_neg])
        label_l = torch.cat([torch.ones(len(idx_L_pos)), torch.zeros(len(idx_L_neg))])
        label_r = torch.cat([torch.ones(len(idx_R_pos)), torch.zeros(len(idx_R_neg))])
        
        # === PAIR SAMPLING (Optimized) ===
        num_nodes_L = len(graph_L.node_pos)
        num_nodes_R = len(graph_R.node_pos)
        
        # Use sparse representation instead of dense matrix
        num_pos_pairs = len(pos_pair_L)
        total_pairs = num_nodes_L * num_nodes_R
        num_neg_pairs = total_pairs - num_pos_pairs
        
        num_pair_to_use = min(num_neg_pairs, num_pos_pairs)
        
        # Sample positive pairs
        if num_pos_pairs > num_pair_to_use:
            pos_indices = torch.randperm(num_pos_pairs)[:num_pair_to_use]
            pos_pair_L_sampled = pos_pair_L[pos_indices]
            pos_pair_R_sampled = pos_pair_R[pos_indices]
        else:
            pos_pair_L_sampled = pos_pair_L
            pos_pair_R_sampled = pos_pair_R
        
        # Sample negative pairs efficiently without creating dense matrix
        neg_samples = []
        sampled = 0
        max_attempts = num_pair_to_use * 10  # Safety limit
        attempts = 0
        
        # Create set of positive pairs for fast lookup
        pos_set = set(zip(pos_pair_L.tolist(), pos_pair_R.tolist()))
        
        while sampled < num_pair_to_use and attempts < max_attempts:
            batch_size = min(num_pair_to_use - sampled, 1000)
            rand_L = torch.randint(0, num_nodes_L, (batch_size,))
            rand_R = torch.randint(0, num_nodes_R, (batch_size,))
            
            # Filter out positive pairs
            for l, r in zip(rand_L.tolist(), rand_R.tolist()):
                if (l, r) not in pos_set:
                    neg_samples.append((l, r))
                    sampled += 1
                    if sampled >= num_pair_to_use:
                        break
            attempts += 1
        
        if len(neg_samples) > 0:
            neg_pair_L_sampled = torch.tensor([s[0] for s in neg_samples])
            neg_pair_R_sampled = torch.tensor([s[1] for s in neg_samples])
        else:
            # Fallback to old method if sampling fails
            dense_pair = torch.zeros(num_nodes_L, num_nodes_R, dtype=torch.bool)
            dense_pair[pos_pair_L, pos_pair_R] = True
            neg_L, neg_R = torch.where(~dense_pair)
            neg_indices = torch.randperm(len(neg_L))[:num_pair_to_use]
            neg_pair_L_sampled = neg_L[neg_indices]
            neg_pair_R_sampled = neg_R[neg_indices]
        
        idx_left_pair = torch.cat([pos_pair_L_sampled, neg_pair_L_sampled])
        idx_right_pair = torch.cat([pos_pair_R_sampled, neg_pair_R_sampled])
        labels_pair = torch.cat([
            torch.ones(len(pos_pair_L_sampled)),
            torch.zeros(len(neg_pair_L_sampled))
        ])
        
        # Validation
        if idx_left.dtype != torch.int64 or idx_right.dtype != torch.int64:
            return None
        if len(idx_left) < 3 or len(idx_right) < 3:
            return None
        if idx_left.max() >= num_nodes_L or idx_right.max() >= num_nodes_R:
            print('idx error', protein_pair)
            return None
        
        # === SURFACE INTERFACE (Optimized) ===
        train_surface = True
        use_balance = False
        
        if train_surface:
            # Compute distance matrix for surface
            distmat = torch.cdist(surface_L.verts, surface_R.verts)
            s_idxL, s_idxR = torch.where(distmat < 2.0)
            if len(s_idxL)<3 or len(s_idxR)<3:
                return None

            s_site_idxL = s_idxL.unique()
            s_site_idxR = s_idxR.unique()
            
            # Vectorized boolean masks
            s_mask_L = torch.zeros(len(surface_L.verts), dtype=torch.bool)
            s_mask_R = torch.zeros(len(surface_R.verts), dtype=torch.bool)
            s_mask_L[s_site_idxL] = True
            s_mask_R[s_site_idxR] = True
            
            s_site_idxL_neg = torch.where(~s_mask_L)[0]
            s_site_idxR_neg = torch.where(~s_mask_R)[0]
            
            if use_balance:
                # Balanced sampling
                site_neg_num_L = len(s_site_idxL)
                site_neg_num_R = len(s_site_idxR)
                neg_num = len(s_idxL)
            else:
                # Unbalanced sampling
                site_neg_num_L = min(200 * len(s_site_idxL), len(s_site_idxL_neg))
                site_neg_num_R = min(200 * len(s_site_idxR), len(s_site_idxR_neg))
                neg_num = min(200 * len(s_idxL), len(s_idxL) * len(s_idxR) - len(s_idxL))
            
            # Sample negative sites
            s_site_neg_sampleL = s_site_idxL_neg[torch.randperm(len(s_site_idxL_neg))[:site_neg_num_L]]
            s_site_neg_sampleR = s_site_idxR_neg[torch.randperm(len(s_site_idxR_neg))[:site_neg_num_R]]
            
            s_site_idxL_sample = torch.cat([s_site_idxL, s_site_neg_sampleL])
            s_site_idxR_sample = torch.cat([s_site_idxR, s_site_neg_sampleR])
            s_site_label_L = torch.cat([torch.ones(len(s_site_idxL)), torch.zeros(len(s_site_neg_sampleL))])
            s_site_label_R = torch.cat([torch.ones(len(s_site_idxR)), torch.zeros(len(s_site_neg_sampleR))])
            
            # Sample negative pairs for surface
            total_s_pairs = len(surface_L.verts) * len(surface_R.verts)
            if neg_num < total_s_pairs:
                # Efficient negative sampling using set
                pos_s_set = set(zip(s_idxL.tolist(), s_idxR.tolist()))
                neg_s_samples = []
                
                for _ in range(neg_num * 2):  # Try twice as many to ensure we get enough
                    l_idx = torch.randint(0, len(surface_L.verts), (1,)).item()
                    r_idx = torch.randint(0, len(surface_R.verts), (1,)).item()
                    if (l_idx, r_idx) not in pos_s_set:
                        neg_s_samples.append((l_idx, r_idx))
                    if len(neg_s_samples) >= neg_num:
                        break
                
                neg_s_idxL_sample = torch.tensor([s[0] for s in neg_s_samples])
                neg_s_idxR_sample = torch.tensor([s[1] for s in neg_s_samples])
            else:
                # Fallback for small surfaces
                S_dense_pair = torch.zeros(len(surface_L.verts), len(surface_R.verts), dtype=torch.bool)
                S_dense_pair[s_idxL, s_idxR] = True
                neg_s_idxL, neg_s_idxR = torch.where(~S_dense_pair)
                neg_sample = torch.randperm(len(neg_s_idxL))[:neg_num]
                neg_s_idxL_sample = neg_s_idxL[neg_sample]
                neg_s_idxR_sample = neg_s_idxR[neg_sample]
            
            sidx_left = torch.cat([s_idxL, neg_s_idxL_sample])
            sidx_right = torch.cat([s_idxR, neg_s_idxR_sample])
            s_label = torch.cat([torch.ones(len(s_idxL)), torch.zeros(len(neg_s_idxL_sample))])
            
            item = Data(
                surface_1=surface_L, graph_1=graph_L,
                surface_2=surface_R, graph_2=graph_R,
                idx_left=idx_left, idx_right=idx_right,
                label_r=label_r, label_l=label_l,
                g1_len=num_nodes_L, g2_len=num_nodes_R,
                idx_left_pair=idx_left_pair, idx_right_pair=idx_right_pair,
                labels_pair=labels_pair, id=protein_pair,
                s1_len=len(surface_L.verts), s2_len=len(surface_R.verts),
                s_idx_left=sidx_left, s_idx_right=sidx_right,
                s_label=s_label,
                s_site_idxL=s_site_idxL_sample, s_site_idxR=s_site_idxR_sample,
                s_site_label_L=s_site_label_L, s_site_label_R=s_site_label_R
            )
        else:
            item = Data(
                surface_1=surface_L, graph_1=graph_L,
                surface_2=surface_R, graph_2=graph_R,
                idx_left=idx_left, idx_right=idx_right,
                label_r=label_r, label_l=label_l,
                g1_len=num_nodes_L, g2_len=num_nodes_R,
                idx_left_pair=idx_left_pair, idx_right_pair=idx_right_pair,
                labels_pair=labels_pair, id=protein_pair
            )
        # print('used time', time.time()-t0)    
        return item
    # def __getitem__(self, idx):
    #     # import time
    #     # t0=time.time()
    #     protein_pair = self.index.iloc[idx]['id']
    #     pdb_R = self.index.iloc[idx]['holo_R_pdb']
    #     pdb_L = self.index.iloc[idx]['holo_L_pdb']
        
    #     graph_R= self.graph_loader.load(pdb_R[:-4])
    #     graph_L=  self.graph_loader.load(pdb_L[:-4])
    #     if graph_L is None or graph_R is None:
    #         return None
    #     from torch_geometric.data import Data, Batch
    #     from atomsurf.network_utils.misc_arch.dmasif_utils.geometry_processing import curvatures
    #     tmp_device=torch.device('cuda')
    #     data_R = Data(coord=graph_R.node_pos, atomtypes=graph_R.x[:,-12:])
    #     data_L = Data(coord=graph_L.node_pos, atomtypes=graph_L.x[:,-12:])
    #     if self.add_noise:
    #         std_dev = 0.3 # 0.3 Angstrom noise on atom coordinates 
    #         noise_L = torch.randn_like(data_L.coord) * std_dev
    #         noise_R = torch.randn_like(data_R.coord) * std_dev
    #         data_L.coord += noise_L
    #         data_R.coord += noise_R
    #     data_R.num_nodes = data_R.coord.size(0)
    #     data_L.num_nodes = data_L.coord.size(0)
    #     tmpbatch = Batch.from_data_list([data_R,data_L]).to(tmp_device)
    #     points, normals, batch_points=atoms_to_points_normals( tmpbatch.coord,    tmpbatch.batch,    distance=1.05,smoothness=0.5,resolution=2.5,nits=4,atomtypes=tmpbatch.atomtypes,sup_sampling=20, variance=0.1,)
    #     P_curvatures = curvatures(
    #         points,
    #         triangles= None,
    #         normals= normals,
    #         scales=[1.0, 2.0, 3.0, 5.0, 10.0],
    #         batch=batch_points,
    #     )
    #     # delete surface_R and surface_L and create a new Data for surface
    #     points = points.detach().cpu()
    #     normals = normals.detach().cpu()
    #     P_curvatures = P_curvatures.detach().cpu()
    #     batch_points = batch_points.detach().cpu()
        
    #     surface_R = Data(verts=points[batch_points==0],n_verts=len(points[batch_points==0]),x=P_curvatures[batch_points==0],num_nodes=len(points[batch_points==0]),vnormals= normals[batch_points==0])
    #     surface_L = Data(verts=points[batch_points==1],n_verts=len(points[batch_points==1]),x=P_curvatures[batch_points==1],num_nodes=len(points[batch_points==1]),vnormals= normals[batch_points==1])

    #     del tmpbatch, points, normals, batch_points, P_curvatures
    #     torch.cuda.empty_cache()

    #     if surface_L is None or surface_R is None or graph_L is None or graph_R is None:
    #         return None
    #     if graph_L.node_len < 20 or graph_R.node_len < 20 or surface_L.n_verts < 20 or surface_R.n_verts < 20:
    #         return None

    #     graph_interface= torch.cdist(graph_L.node_pos,graph_R.node_pos)<3.5
    #     pos_pair= torch.argwhere(graph_interface)
    #     pos_pair_L = pos_pair[:,0]
    #     pos_pair_R = pos_pair[:,1]
    #     R_res= graph_R.res_map[pos_pair_R].unique()
    #     L_res= graph_L.res_map[pos_pair_L].unique()
    #     idx_R_pos = torch.where(torch.isin(graph_R.res_map, R_res))[0].numpy()
    #     idx_L_pos = torch.where(torch.isin(graph_L.res_map, L_res))[0].numpy()

    #     denseR = np.zeros(len(graph_R.node_pos))
    #     denseL = np.zeros(len(graph_L.node_pos))

    #     denseL[idx_L_pos] = 1 
    #     denseR[idx_R_pos] = 1
    #     idx_L_neg = np.where(denseL==0)[0]
    #     idx_R_neg = np.where(denseR==0)[0]

    #     pos_array_sampledL = torch.from_numpy(idx_L_pos)
    #     pos_array_sampledR = torch.from_numpy(idx_R_pos)
    #     neg_array_sampledL = torch.from_numpy(idx_L_neg)
    #     neg_array_sampledR = torch.from_numpy(idx_R_neg)

    #     idx_left = torch.cat((pos_array_sampledL, neg_array_sampledL))
    #     idx_right = torch.cat((pos_array_sampledR, neg_array_sampledR))
    #     label_l = torch.cat((torch.ones(len(pos_array_sampledL)),torch.zeros(len(neg_array_sampledL))))
    #     label_r = torch.cat((torch.ones(len(pos_array_sampledR)),torch.zeros(len(neg_array_sampledR))))
    #     #prepare pair idx
    #     dense_pair=np.zeros([len(graph_L.node_pos),len(graph_R.node_pos)])
    #     dense_pair[pos_pair_L,pos_pair_R]=1.0
    #     pos_pair= np.vstack([pos_pair[:,0],pos_pair[:,1]])
    #     neg_pair = np.where(dense_pair==0)
    #     neg_pair = np.vstack([neg_pair[0],neg_pair[1]])
    #     num_pair_to_use = min(len(neg_pair[0]),len(pos_pair_L))
    #     pos_array_idx = np.random.choice(len(pos_pair_L), size=num_pair_to_use, replace=False)
    #     neg_array_idx = np.random.choice(len(neg_pair[0]), size=num_pair_to_use, replace=False)
    #     pos_pair_sampled_p = torch.from_numpy(pos_pair[:,pos_array_idx])
    #     neg_pair_sampled_p  =  torch.from_numpy(neg_pair[:, neg_array_idx])
    #     idx_left_pair = torch.cat((pos_pair_sampled_p[0], neg_pair_sampled_p[0]))
    #     idx_right_pair = torch.cat((pos_pair_sampled_p[1], neg_pair_sampled_p[1]))
    #     labels_pair = torch.cat((torch.ones(num_pair_to_use), torch.zeros(num_pair_to_use)))
    #     if idx_left.dtype != torch.int64 and idx_right.dtype != torch.int64 or len(idx_left) <3 or len(idx_right) <3 :
    #         return None
    #     if idx_left.max() >= len(graph_L.node_pos) or idx_right.max() >= len(graph_R.node_pos):
    #         print('idx error', protein_pair)
    #         return None
    #     # add_noise = True
    #     # if add_noise:
    #     #     std_dev = 0.2
    #     #     s_std_dev = 0.02
    #     #     noise_L = torch.randn_like(graph_L.node_pos) * std_dev
    #     #     noise_R = torch.randn_like(graph_R.node_pos) * std_dev
    #     #     graph_L.node_pos += noise_L
    #     #     graph_R.node_pos += noise_R
    #     #     s_noise_L = torch.randn_like(surface_L.verts) * s_std_dev
    #     #     s_noise_R = torch.randn_like(surface_R.verts) * s_std_dev
    #     #     surface_L.verts += s_noise_L
    #     #     surface_R.verts += s_noise_R
    #     train_surface=True
    #     use_balance = False 
    #     if train_surface:
            
    #         distmat= torch.cdist(surface_L.verts,surface_R.verts)
    #         s_idxL,s_idxR = torch.where(distmat<2.0)
    #         s_site_idxL =s_idxL.unique()
    #         s_site_idxR =s_idxR.unique()
    #         s_denseR = torch.zeros(len(surface_R.verts))
    #         s_denseL = torch.zeros(len(surface_L.verts))
    #         s_denseL[s_site_idxL] = 1 
    #         s_denseR[s_site_idxR] = 1
    #         s_site_idxL_neg = torch.where(s_denseL==0)[0]
    #         s_site_idxR_neg = torch.where(s_denseR==0)[0]

    #         S_dense_pair=torch.zeros([len(surface_L.verts),len(surface_R.verts)])
    #         S_dense_pair[s_idxL,s_idxR]=1.0
    #         neg_s_idxL,neg_s_idxR = torch.where(S_dense_pair==0)                
    #         if use_balance:

    #             s_site_neg_sampleL= s_site_idxL_neg[torch.randperm(len(s_site_idxL_neg))[:len(s_site_idxL)]]
    #             s_site_neg_sampleR= s_site_idxR_neg[torch.randperm(len(s_site_idxR_neg))[:len(s_site_idxR)]]
    #             s_site_idxL_sample = torch.cat([s_site_idxL,s_site_neg_sampleL])
    #             s_site_idxR_sample = torch.cat([s_site_idxR,s_site_neg_sampleR])
    #             s_site_label_L = torch.cat([torch.ones(len(s_site_idxL)),torch.zeros(len(s_site_neg_sampleL))])
    #             s_site_label_R = torch.cat([torch.ones(len(s_site_idxR)),torch.zeros(len(s_site_neg_sampleR))])

    #             neg_sample = torch.randperm(len(neg_s_idxL))[:len(s_idxL)]
    #             neg_s_idxL_sample = neg_s_idxL[neg_sample]
    #             neg_s_idxR_sample = neg_s_idxR[neg_sample]
    #             sidx_left = torch.cat([s_idxL,neg_s_idxL_sample])
    #             sidx_right = torch.cat([s_idxR,neg_s_idxR_sample])
    #             s_label = torch.cat([torch.ones(len(s_idxL)),torch.zeros(len(neg_s_idxL_sample))])

    #         else:
    #             site_neg_num = min(200*len(s_site_idxL),len(s_site_idxL_neg),200*len(s_site_idxR),len(s_site_idxR_neg))
    #             s_site_neg_sampleL= s_site_idxL_neg[torch.randperm(len(s_site_idxL_neg))[:site_neg_num]]
    #             s_site_neg_sampleR= s_site_idxR_neg[torch.randperm(len(s_site_idxR_neg))[:site_neg_num]]
    #             s_site_idxL_sample = torch.cat([s_site_idxL,s_site_neg_sampleL])
    #             s_site_idxR_sample = torch.cat([s_site_idxR,s_site_neg_sampleR])
    #             s_site_label_L = torch.cat([torch.ones(len(s_site_idxL)),torch.zeros(len(s_site_neg_sampleL))])
    #             s_site_label_R = torch.cat([torch.ones(len(s_site_idxR)),torch.zeros(len(s_site_neg_sampleR))])

    #             neg_num = min(200*len(s_idxL),len(neg_s_idxL))
    #             neg_sample = torch.randperm(len(neg_s_idxL))[:neg_num]
    #             neg_s_idxL_sample = neg_s_idxL[neg_sample]
    #             neg_s_idxR_sample = neg_s_idxR[neg_sample]
    #             sidx_left = torch.cat([s_idxL,neg_s_idxL_sample])
    #             sidx_right = torch.cat([s_idxR,neg_s_idxR_sample])
    #             s_label = torch.cat([torch.ones(len(s_idxL)),torch.zeros(len(neg_s_idxL_sample))])
    #         item = Data(surface_1=surface_L, graph_1=graph_L, surface_2=surface_R, graph_2=graph_R, idx_left=idx_left,
    #                     idx_right=idx_right, label_r=label_r,label_l=label_l, g1_len=graph_L.node_pos.shape[0],
    #                     g2_len=graph_R.node_pos.shape[0],idx_left_pair=idx_left_pair,idx_right_pair=idx_right_pair,labels_pair=labels_pair,id=protein_pair,s1_len = len(surface_L.verts),s2_len=len(surface_R.verts),s_idx_left=sidx_left,s_idx_right=sidx_right,s_label=s_label,s_site_idxL=s_site_idxL_sample,s_site_idxR=s_site_idxR_sample,s_site_label_L=s_site_label_L,s_site_label_R=s_site_label_R)
            
    #     else:  
    #         item = Data(surface_1=surface_L, graph_1=graph_L, surface_2=surface_R, graph_2=graph_R, idx_left=idx_left,
    #                     idx_right=idx_right, label_r=label_r,label_l=label_l, g1_len=graph_L.node_pos.shape[0],
    #                     g2_len=graph_R.node_pos.shape[0],idx_left_pair=idx_left_pair,idx_right_pair=idx_right_pair,labels_pair=labels_pair,id=protein_pair)
    #     # print('used time', time.time()-t0)    
    #     return item   

class PINDERDataset_atomgraph_nosurf(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, mode,neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.data_dir= data_dir
        self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster.parquet')) # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
        self.index= self.index[self.index.split==mode]
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
        
        graph_R= self.graph_loader.load(pdb_R[:-4])
        graph_L=  self.graph_loader.load(pdb_L[:-4])
        if graph_L is None or graph_R is None:
            return None

        if graph_L.node_len < 20 or graph_R.node_len < 20:
            return None

        graph_interface= torch.cdist(graph_L.node_pos,graph_R.node_pos)<3.5
        pos_pair= torch.argwhere(graph_interface)
        pos_pair_L = pos_pair[:,0]
        pos_pair_R = pos_pair[:,1]
        R_res= graph_R.res_map[pos_pair_R].unique()
        L_res= graph_L.res_map[pos_pair_L].unique()
        idx_R_pos = torch.where(torch.isin(graph_R.res_map, R_res))[0].numpy()
        idx_L_pos = torch.where(torch.isin(graph_L.res_map, L_res))[0].numpy()

        denseR = np.zeros(len(graph_R.node_pos))
        denseL = np.zeros(len(graph_L.node_pos))
        denseL[idx_L_pos] = 1 
        denseR[idx_R_pos] = 1
        idx_L_neg = np.where(denseL==0)[0]
        idx_R_neg = np.where(denseR==0)[0]

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
            noise_L = torch.randn_like(graph_L.node_pos) * std_dev
            noise_R = torch.randn_like(graph_R.node_pos) * std_dev
            graph_L.node_pos += noise_L
            graph_R.node_pos += noise_R

        ### dont precompute surface ###

        try:
            surface_R = torch.load('/work/lpdi/users/ymiao/code/pinderdata/surface_PTC/'+pdb_R[:-4]+'.pt')
            surface_L = torch.load('/work/lpdi/users/ymiao/code/pinderdata/surface_PTC/'+pdb_L[:-4]+'.pt')
        except Exception as e:
            print(f"Failed to load {pdb_R}: {e}")
            return None 
        distmat= torch.cdist(surface_L.verts,surface_R.verts)
        s_idxL,s_idxR = torch.where(distmat<2.0)
        if len(s_idxL)<3:
            return None
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
        
        site_neg_num = min(200*len(s_site_idxL),len(s_site_idxL_neg),200*len(s_site_idxR),len(s_site_idxR_neg))
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
        return item      

class PINDERDataset_resgraph_nosurf(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, mode,neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.data_dir= data_dir
        self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster_existing.parquet')) # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
        self.index= self.index[self.index.split==mode]
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

        if graph_L is None or graph_R is None:
            return None
        if graph_L.node_len < 20 or graph_R.node_len < 20 :
            return None
        denseR = np.zeros(len(graph_R.node_pos))
        denseL = np.zeros(len(graph_L.node_pos))
        # if len(idx_R_pos)< 3 or len(idx_L_pos)< 3:
        #     return None
        denseL[idx_L_pos] = 1 
        denseR[idx_R_pos] = 1
        # idx_L_neg = np.where(denseL==0)[0]
        # idx_R_neg = np.where(denseR==0)[0]

        # pos_array_sampledL = torch.from_numpy(idx_L_pos)
        # pos_array_sampledR = torch.from_numpy(idx_R_pos)
        # neg_array_sampledL = torch.from_numpy(idx_L_neg)
        # neg_array_sampledR = torch.from_numpy(idx_R_neg)
        
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
            noise_L = torch.randn_like(graph_L.node_pos) * std_dev
            noise_R = torch.randn_like(graph_R.node_pos) * std_dev
            graph_L.node_pos += noise_L
            graph_R.node_pos += noise_R

        ### dont precompute surface ###

        try:
            surface_R = torch.load('/work/lpdi/users/ymiao/code/pinderdata/surface_PTC/'+pdb_R[:-4]+'.pt')
            surface_L = torch.load('/work/lpdi/users/ymiao/code/pinderdata/surface_PTC/'+pdb_L[:-4]+'.pt')
        except Exception as e:
            print(f"Failed to load {pdb_R}: {e}")
            return None 
        if surface_R is None or surface_L is None:
            return None
        if len(surface_R.verts) < 100 or len(surface_L.verts) < 100 :
            return None
        distmat= torch.cdist(surface_L.verts,surface_R.verts)
        s_idxL,s_idxR = torch.where(distmat<2.0)
        if len(s_idxL)<3:
            return None
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
        
        site_neg_num = min(200*len(s_site_idxL),len(s_site_idxL_neg),200*len(s_site_idxR),len(s_site_idxR_neg))
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
        return item    

class PINDERDataset_test_all_site(Dataset):
    def __init__(self, data_dir, surface_builder, graph_builder, mode,neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1):
        self.data_dir= data_dir
        self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster.parquet')) # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
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
class PINDERDataset_residuegraph_onfly(Dataset):
    def __init__(self, data_dir, surface_builder, agraph_loader,rgraph_loader, mode,neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1,add_noise=True):
        self.data_dir= data_dir
        self.index= pd.read_parquet(os.path.join(data_dir,'processed_pinder_index_cluster.parquet')) # processed_pinder_index_cluster.parquet #/processed_pinder_index.parquet
        self.index= self.index[self.index.split==mode]
        # metadata= pd.read_parquet(data_dir+'/processed_pinder_metadata.parquet')
        self.surface_loader = surface_builder
        self.agraph_loader = agraph_loader 
        self.rgraph_loader = rgraph_loader
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_regions_per_ensemble = max_pos_regions_per_ensemble
        self.add_noise = add_noise
        self.mode = mode
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
        
    # def obtain_ca(self,agraph,coord):
    #     sorted_pos = coord 
    #     unique_ids, counts = torch.unique_consecutive(agraph.res_map, return_counts=True)
    #     cum_counts = torch.cumsum(counts, dim=0)
    #     start_idx = torch.cat([torch.tensor([0], device=cum_counts.device), cum_counts[:-1]])
    #     ca_idx = torch.minimum(start_idx + 1, start_idx + counts - 1)
    #     ca_coord = sorted_pos[ca_idx]
    #     return ca_coord
    def __getitem__(self, idx):
        import time
        t0 = time.time()
        """Optimized data loading with ~2-3x speedup."""
        
        # Load basic data
        protein_pair = self.index.iloc[idx]['id']
        pdb_R = self.index.iloc[idx]['holo_R_pdb']
        pdb_L = self.index.iloc[idx]['holo_L_pdb']
        
        agraph_R = self.agraph_loader.load(pdb_R[:-4])
        agraph_L = self.agraph_loader.load(pdb_L[:-4])
        rgraph_R = self.rgraph_loader.load(pdb_R[:-4])
        rgraph_L = self.rgraph_loader.load(pdb_L[:-4])    
        if agraph_R is None or agraph_L is None or rgraph_R is None or rgraph_L is None:
            return None
        
        # Early validation to avoid unnecessary computation
        if rgraph_L.node_len < 20 or rgraph_R.node_len < 20:
            return None
        
        # === SURFACE COMPUTATION (Optimized GPU usage) ===
        tmp_device = torch.device('cuda')
        
        # Pre-allocate on GPU and add noise in one operation
        coord_L = agraph_L.node_pos
        coord_R = agraph_R.node_pos
        
        if self.add_noise and self.mode=='train':
            coord_L = coord_L + torch.randn_like(coord_L) * 0.5
            coord_R = coord_R + torch.randn_like(coord_R) * 0.5
            # ca_coord_L = self.obtain_ca(agraph_L,coord_L)
            # ca_coord_R = self.obtain_ca(agraph_R,coord_R)
            # if len(ca_coord_L)!= rgraph_L.node_len or len(ca_coord_R)== rgraph_R.node_len:
            #     print('ca L error', protein_pair)
            #     return None
            # else:
            #     rgraph_L.node_pos = ca_coord_L
            #     rgraph_R.node_pos = ca_coord_R
            # unique_ids_L, counts_L = torch.unique_consecutive(agraph_L.res_map, return_counts=True)
            # cum_counts_L = torch.cumsum(counts_L, dim=0)
            # start_idx_L = torch.cat([torch.tensor([0], device=cum_counts_L.device), cum_counts_L[:-1]])
            # ca_idx_L = torch.minimum(start_idx_L + 1, start_idx_L + counts_L - 1)
            # unique_ids_R, counts_R = torch.unique_consecutive(agraph_R.res_map, return_counts=True)
            # cum_counts_R = torch.cumsum(counts_R, dim=0)
            # start_idx_R = torch.cat([torch.tensor([0], device=cum_counts_R.device), cum_counts_R[:-1]])
            # ca_idx_R = torch.minimum(start_idx_R + 1, start_idx_R + counts_R - 1)
            diff = torch.abs(agraph_L.node_pos.unsqueeze(0) - rgraph_L.node_pos.unsqueeze(1))  # (N, M, 3)
            matches = (diff < 1e-4).all(dim=2)                              # (N, M)
            ca_idx_L = matches.float().argmax(dim=1)
            diff = torch.abs(agraph_R.node_pos.unsqueeze(0) - rgraph_R.node_pos.unsqueeze(1))  # (N, M, 3)
            matches = (diff < 1e-4).all(dim=2)                              #
            ca_idx_R = matches.float().argmax(dim=1)
            # Validate lengths before assignment
            if len(ca_idx_L) != rgraph_L.node_len or len(ca_idx_R) != rgraph_R.node_len:
                print(f'CA index mismatch for {protein_pair}')
                return None
            else:
                rgraph_L.node_pos = coord_L[ca_idx_L]
                rgraph_R.node_pos = coord_R[ca_idx_R]
            
        data_R = Data(coord=coord_R, atomtypes=agraph_R.x[:, -12:], num_nodes=coord_R.size(0))
        data_L = Data(coord=coord_L, atomtypes=agraph_L.x[:, -12:], num_nodes=coord_L.size(0))
        tmpbatch = Batch.from_data_list([data_R, data_L]).to(tmp_device)
        
        points, normals, batch_points = atoms_to_points_normals(
            tmpbatch.coord, tmpbatch.batch, distance=1.05, smoothness=0.5,
            resolution=2.5, nits=4, atomtypes=tmpbatch.atomtypes,
            sup_sampling=20, variance=0.1
        )
        
        P_curvatures = curvatures(
            points, triangles=None, normals=normals,
            scales=[1.0, 2.0, 3.0, 5.0, 10.0], batch=batch_points
        )
        
        # Move to CPU once and cleanup GPU memory
        mask_R = batch_points == 0
        mask_L = batch_points == 1
        
        surface_R = Data(
            verts=points[mask_R].cpu(),
            n_verts=mask_R.sum().item(),
            x=P_curvatures[mask_R].cpu(),
            num_nodes=mask_R.sum().item(),
            vnormals=normals[mask_R].cpu()
        )
        
        surface_L = Data(
            verts=points[mask_L].cpu(),
            n_verts=mask_L.sum().item(),
            x=P_curvatures[mask_L].cpu(),
            num_nodes=mask_L.sum().item(),
            vnormals=normals[mask_L].cpu()
        )
        
        del tmpbatch, points, normals, batch_points, P_curvatures, mask_R, mask_L
        torch.cuda.empty_cache()
        
        if surface_L.n_verts < 20 or surface_R.n_verts < 20:
            return None
        
        # === GRAPH INTERFACE COMPUTATION (Vectorized) ===
        # Use squared distance to avoid sqrt
        dist_matrix = torch.cdist(agraph_L.node_pos, agraph_R.node_pos)
        agraph_interface = dist_matrix < 5.0
        apos_pair_L, apos_pair_R = torch.where(agraph_interface)
        pairs = torch.stack([
            agraph_L.res_map[apos_pair_L],
            agraph_R.res_map[apos_pair_R]
        ], dim=1)
        unique_pairs = torch.unique(pairs, dim=0)
        pos_pair_L = unique_pairs[:, 0]
        pos_pair_R = unique_pairs[:, 1]
        denseL = torch.zeros(len(rgraph_L.node_pos), dtype=torch.bool)
        denseR = torch.zeros(len(rgraph_R.node_pos), dtype=torch.bool)
        denseL[pos_pair_L] = 1 
        denseR[pos_pair_R] = 1
        mask_L_pos = denseL.clone()
        mask_R_pos = denseR.clone()
        idx_R_pos = pos_pair_R.unique()
        idx_L_pos = pos_pair_L.unique()
        # Compute negative indices efficiently
        idx_L_neg = torch.where(~mask_L_pos)[0]
        idx_R_neg = torch.where(~mask_R_pos)[0]
        
        # Concatenate labels
        idx_left = torch.cat([idx_L_pos, idx_L_neg])
        idx_right = torch.cat([idx_R_pos, idx_R_neg])
        label_l = torch.cat([torch.ones(len(idx_L_pos)), torch.zeros(len(idx_L_neg))])
        label_r = torch.cat([torch.ones(len(idx_R_pos)), torch.zeros(len(idx_R_neg))])
        
        # === PAIR SAMPLING (Optimized) ===
        num_nodes_L = len(rgraph_L.node_pos)
        num_nodes_R = len(rgraph_R.node_pos)
        
        # Use sparse representation instead of dense matrix
        num_pos_pairs = len(pos_pair_L)
        total_pairs = num_nodes_L * num_nodes_R
        num_neg_pairs = total_pairs - num_pos_pairs
        
        num_pair_to_use = min(num_neg_pairs, num_pos_pairs)
        
        # Sample positive pairs
        if num_pos_pairs > num_pair_to_use:
            pos_indices = torch.randperm(num_pos_pairs)[:num_pair_to_use]
            pos_pair_L_sampled = pos_pair_L[pos_indices]
            pos_pair_R_sampled = pos_pair_R[pos_indices]
        else:
            pos_pair_L_sampled = pos_pair_L
            pos_pair_R_sampled = pos_pair_R
        
        # Sample negative pairs efficiently without creating dense matrix
        neg_samples = []
        sampled = 0
        max_attempts = num_pair_to_use * 5  # Safety limit
        attempts = 0
        
        # Create set of positive pairs for fast lookup
        pos_set = set(zip(pos_pair_L.tolist(), pos_pair_R.tolist()))
        
        while sampled < num_pair_to_use and attempts < max_attempts:
            batch_size = min(num_pair_to_use - sampled, 1000)
            rand_L = torch.randint(0, num_nodes_L, (batch_size,))
            rand_R = torch.randint(0, num_nodes_R, (batch_size,))
            
            # Filter out positive pairs
            for l, r in zip(rand_L.tolist(), rand_R.tolist()):
                if (l, r) not in pos_set:
                    neg_samples.append((l, r))
                    sampled += 1
                    if sampled >= num_pair_to_use:
                        break
            attempts += 1
        
        if len(neg_samples) > 0:
            neg_pair_L_sampled = torch.tensor([s[0] for s in neg_samples])
            neg_pair_R_sampled = torch.tensor([s[1] for s in neg_samples])
        else:
            # Fallback to old method if sampling fails
            dense_pair = torch.zeros(num_nodes_L, num_nodes_R, dtype=torch.bool)
            dense_pair[pos_pair_L, pos_pair_R] = True
            neg_L, neg_R = torch.where(~dense_pair)
            neg_indices = torch.randperm(len(neg_L))[:num_pair_to_use]
            neg_pair_L_sampled = neg_L[neg_indices]
            neg_pair_R_sampled = neg_R[neg_indices]
        
        idx_left_pair = torch.cat([pos_pair_L_sampled, neg_pair_L_sampled])
        idx_right_pair = torch.cat([pos_pair_R_sampled, neg_pair_R_sampled])
        labels_pair = torch.cat([
            torch.ones(len(pos_pair_L_sampled)),
            torch.zeros(len(neg_pair_L_sampled))
        ])
        
        # Validation
        if idx_left.dtype != torch.int64 or idx_right.dtype != torch.int64:
            return None
        if len(idx_left) < 3 or len(idx_right) < 3:
            return None
        if idx_left.max() >= num_nodes_L or idx_right.max() >= num_nodes_R:
            print('idx error', protein_pair)
            return None
        
        # === SURFACE INTERFACE (Optimized) ===
        train_surface = True
        use_balance = False
        
        if train_surface:
            # Compute distance matrix for surface
            distmat = torch.cdist(surface_L.verts, surface_R.verts)
            s_idxL, s_idxR = torch.where(distmat < 2.0)
            if len(s_idxL)<3 or len(s_idxR)<3:
                return None

            s_site_idxL = s_idxL.unique()
            s_site_idxR = s_idxR.unique()
            
            # Vectorized boolean masks
            s_mask_L = torch.zeros(len(surface_L.verts), dtype=torch.bool)
            s_mask_R = torch.zeros(len(surface_R.verts), dtype=torch.bool)
            s_mask_L[s_site_idxL] = True
            s_mask_R[s_site_idxR] = True
            
            s_site_idxL_neg = torch.where(~s_mask_L)[0]
            s_site_idxR_neg = torch.where(~s_mask_R)[0]
            
            if use_balance:
                # Balanced sampling
                site_neg_num_L = len(s_site_idxL)
                site_neg_num_R = len(s_site_idxR)
                neg_num = len(s_idxL)
            else:
                # Unbalanced sampling
                site_neg_num_L = min(20 * len(s_site_idxL), len(s_site_idxL_neg))
                site_neg_num_R = min(20 * len(s_site_idxR), len(s_site_idxR_neg))
                neg_num = min(20 * len(s_idxL), len(s_idxL) * len(s_idxR) - len(s_idxL))
            
            # Sample negative sites
            s_site_neg_sampleL = s_site_idxL_neg[torch.randperm(len(s_site_idxL_neg))[:site_neg_num_L]]
            s_site_neg_sampleR = s_site_idxR_neg[torch.randperm(len(s_site_idxR_neg))[:site_neg_num_R]]
            
            s_site_idxL_sample = torch.cat([s_site_idxL, s_site_neg_sampleL])
            s_site_idxR_sample = torch.cat([s_site_idxR, s_site_neg_sampleR])
            s_site_label_L = torch.cat([torch.ones(len(s_site_idxL)), torch.zeros(len(s_site_neg_sampleL))])
            s_site_label_R = torch.cat([torch.ones(len(s_site_idxR)), torch.zeros(len(s_site_neg_sampleR))])
            
            # Sample negative pairs for surface
            total_s_pairs = len(surface_L.verts) * len(surface_R.verts)
            if neg_num < total_s_pairs:
                # Efficient negative sampling using set
                pos_s_set = set(zip(s_idxL.tolist(), s_idxR.tolist()))
                neg_s_samples = []
                
                for _ in range(neg_num * 2):  # Try twice as many to ensure we get enough
                    l_idx = torch.randint(0, len(surface_L.verts), (1,)).item()
                    r_idx = torch.randint(0, len(surface_R.verts), (1,)).item()
                    if (l_idx, r_idx) not in pos_s_set:
                        neg_s_samples.append((l_idx, r_idx))
                    if len(neg_s_samples) >= neg_num:
                        break
                
                neg_s_idxL_sample = torch.tensor([s[0] for s in neg_s_samples])
                neg_s_idxR_sample = torch.tensor([s[1] for s in neg_s_samples])
            else:
                # Fallback for small surfaces
                S_dense_pair = torch.zeros(len(surface_L.verts), len(surface_R.verts), dtype=torch.bool)
                S_dense_pair[s_idxL, s_idxR] = True
                neg_s_idxL, neg_s_idxR = torch.where(~S_dense_pair)
                neg_sample = torch.randperm(len(neg_s_idxL))[:neg_num]
                neg_s_idxL_sample = neg_s_idxL[neg_sample]
                neg_s_idxR_sample = neg_s_idxR[neg_sample]
            
            sidx_left = torch.cat([s_idxL, neg_s_idxL_sample])
            sidx_right = torch.cat([s_idxR, neg_s_idxR_sample])
            s_label = torch.cat([torch.ones(len(s_idxL)), torch.zeros(len(neg_s_idxL_sample))])
            
            item = Data(
                surface_1=surface_L, graph_1=rgraph_L,
                surface_2=surface_R, graph_2=rgraph_R,
                idx_left=idx_left, idx_right=idx_right,
                label_r=label_r, label_l=label_l,
                g1_len=num_nodes_L, g2_len=num_nodes_R,
                idx_left_pair=idx_left_pair, idx_right_pair=idx_right_pair,
                labels_pair=labels_pair, id=protein_pair,
                s1_len=len(surface_L.verts), s2_len=len(surface_R.verts),
                s_idx_left=sidx_left, s_idx_right=sidx_right,
                s_label=s_label,
                s_site_idxL=s_site_idxL_sample, s_site_idxR=s_site_idxR_sample,
                s_site_label_L=s_site_label_L, s_site_label_R=s_site_label_R
            )
        else:
            item = Data(
                surface_1=surface_L, graph_1=rgraph_L,
                surface_2=surface_R, graph_2=rgraph_R,
                idx_left=idx_left, idx_right=idx_right,
                label_r=label_r, label_l=label_l,
                g1_len=num_nodes_L, g2_len=num_nodes_R,
                idx_left_pair=idx_left_pair, idx_right_pair=idx_right_pair,
                labels_pair=labels_pair, id=protein_pair
            )
        # print('used time', time.time()-t0)    

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
                            'pin_memory': True, #self.cfg.loader.pin_memory
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
        # dataset =  PINDERDataset_Af_Apo( self.data_dir, self.surface_loader, self.graph_loader,mode='test',pdbtype='apo',neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1)
        return DataLoader(dataset, shuffle=False, **self.loader_args)
def collate_atom_batch(batch):
    return AtomBatch.from_data_list(batch)
class PINDERDataModule_atomgraph_onfly(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.surface_loader = SurfaceLoader(self.cfg.cfg_surface)
        self.graph_loader = GraphLoader(self.cfg.cfg_graph)

        self.loader_args = {
            'num_workers': self.cfg.loader.num_workers,
            'batch_size': self.cfg.loader.batch_size,
            'pin_memory': self.cfg.loader.pin_memory,
            'prefetch_factor': self.cfg.loader.prefetch_factor,
            'collate_fn': collate_atom_batch  # Use the named function here
        }

        # Useful to create a Model of the right input dims
        train_dataset_temp = PINDERDataset_atomgraph_onfly(self.data_dir,
                                                           self.surface_loader,
                                                           self.graph_loader,
                                                           mode='val') #PINDERDataset_atomgraph_nosurf 
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp,
                               gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        dataset = PINDERDataset_atomgraph_onfly(self.data_dir, self.surface_loader, self.graph_loader,mode='train')
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = PINDERDataset_atomgraph_onfly(self.data_dir, self.surface_loader, self.graph_loader,mode='val')
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = PINDERDataset_atomgraph_onfly(self.data_dir, self.surface_loader, self.graph_loader,mode='test')
        # dataset = PINDERDataset(self.data_dir, self.surface_loader, self.graph_loader,mode='test')
        # dataset =  PINDERDataset_Af_Apo( self.data_dir, self.surface_loader, self.graph_loader,mode='test',pdbtype='apo',neg_to_pos_ratio=1, max_pos_regions_per_ensemble=-1)
        return DataLoader(dataset, shuffle=False, **self.loader_args)

class PINDERDataModule_resgraph_onfly(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.surface_loader = SurfaceLoader(self.cfg.cfg_surface)
        self.rgraph_loader = GraphLoader(self.cfg.cfg_graph)
        cfg_copy = self.cfg.cfg_graph.copy()
        cfg_copy.data_name = 'agraph'
        self.agraph_loader = GraphLoader(cfg_copy)
        self.loader_args = {
            'num_workers': self.cfg.loader.num_workers,
            'batch_size': self.cfg.loader.batch_size,
            'pin_memory': self.cfg.loader.pin_memory,
            'prefetch_factor': self.cfg.loader.prefetch_factor,
            'collate_fn': collate_atom_batch  # Use the named function here
        }

        # Useful to create a Model of the right input dims
        train_dataset_temp = PINDERDataset_residuegraph_onfly(self.data_dir,
                                                           self.surface_loader,
                                                           self.agraph_loader,
                                                           self.rgraph_loader,
                                                           mode='val')
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp,
                               gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        dataset = PINDERDataset_residuegraph_onfly(self.data_dir, self.surface_loader, self.agraph_loader,self.rgraph_loader,mode='train')
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = PINDERDataset_residuegraph_onfly(self.data_dir, self.surface_loader, self.agraph_loader,self.rgraph_loader,mode='val')
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = PINDERDataset_residuegraph_onfly(self.data_dir, self.surface_loader, self.agraph_loader,self.rgraph_loader,mode='test')
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
        train_dataset_temp = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='val')
        update_model_input_dim(cfg=cfg, dataset_temp=train_dataset_temp, gkey='graph_1', skey='surface_1')

    def train_dataloader(self):
        dataset = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='train')
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='val')
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = PINDERDataset_test_all_site(self.data_dir, self.surface_loader, self.graph_loader,mode='test')
        return DataLoader(dataset, shuffle=False, **self.loader_args)
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


if __name__ == '__main__':
    pass
