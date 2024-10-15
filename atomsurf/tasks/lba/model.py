from torch_geometric.nn import AttentiveFP
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, MLP, SAGEConv
from torch_scatter import scatter_sum,scatter_max

import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder
import pickle

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata,edge_dim,hidden_channels, out_channels, num_layers):
        super().__init__()
        self.edge_mlp = MLP(channel_list=[128+8,512,64,16],dropout=0.1)
        self.lin_mpl = Linear(in_channels=16,out_channels=16)
        self.edge_lin = Linear(in_channels=1,out_channels=8) 

        self.conv_1 = HeteroConv(
            {
                edge_type: SAGEConv([-1,-1],hidden_channels)
                            for edge_type in metadata[1]
            }
        )
        self.conv_2 = HeteroConv(
            {
                edge_type: SAGEConv([-1,-1],hidden_channels)
                            for edge_type in metadata[1]
            }
        )
        self.conv_3 = HeteroConv(
            {
                edge_type: SAGEConv([-1,-1],hidden_channels)
                            for edge_type in metadata[1]
            }
        )


    def forward(self, x_dict, edge_index_dict,edge_attr_dict,batch_dict):

        x1_dict = self.conv_1(x_dict, edge_index_dict)
        x1_dict = {key: F.leaky_relu(x) for key, x in x1_dict.items()}

        x2_dict = self.conv_2(x1_dict, edge_index_dict)
        x2_dict = {key: F.leaky_relu(x) for key, x in x2_dict.items()}

        x3_dict = self.conv_3(x2_dict, edge_index_dict)
        x3_dict = {key: F.leaky_relu(x) for key, x in x3_dict.items()}

        x_dict['ligand'] = x1_dict['ligand'] + x2_dict['ligand'] + x3_dict['ligand']
        x_dict['protein'] = x1_dict['protein'] + x2_dict['protein'] + x3_dict['protein']

        src, dst = edge_index_dict[('ligand','to','protein')]
        edge_repr = torch.cat([x_dict['ligand'][src], x_dict['protein'][dst]], dim=-1)

        d_pl = self.edge_lin(edge_attr_dict[('ligand','to','protein')])
        edge_repr = torch.cat((edge_repr,d_pl),dim=1)
        m_pl = self.edge_mlp(edge_repr)
        edge_batch = batch_dict['ligand'][src]

        w_pl = torch.tanh(self.lin_mpl(m_pl))
        m_w =  w_pl * m_pl
        m_w = scatter_sum(m_w, edge_batch, dim=0)

        m_max,_ = scatter_max(m_pl,edge_batch,dim=0)
        m_out = torch.cat((m_w, m_max), dim=1)

        return m_out

class LBANet(torch.nn.Module):
    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.hparams_head = cfg_head
        self.hparams_encoder = cfg_encoder
        self.encoder = ProteinEncoder(cfg_encoder)
        #need to get metadata
        exampledata= pickle.load(open('./example_mfedata.pkl','rb'))
        self.heterognn = HeteroGNN(exampledata[1].metadata(), edge_dim=10, hidden_channels=64, out_channels=8,num_layers=3)
        self.ligandgnn = AttentiveFP(in_channels=18,hidden_channels=64,out_channels=16,edge_dim=12,num_timesteps=3,num_layers=3,dropout=0.3)

        self.top_net = nn.Sequential(*[
            nn.Linear(16+32+cfg_head.encoded_dims, cfg_head.encoded_dims),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(cfg_head.encoded_dims),
            nn.SiLU(),
            nn.Linear(cfg_head.encoded_dims, out_features=1)#cfg_head.output_dims)
        ])

    def forward(self, batch):
        # forward pass
        if torch.isnan(batch.graph.x).any():
            print('Nan in graph X')
            import pdb
            pdb.set_trace()
            return None
        if torch.isnan(batch.surface.x).any():
            print('Nan in graph X')
            import pdb
            pdb.set_trace()
            return None
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)
        g_l= batch.g_ligand
        g_pl= batch.g_inter
        l_emb = self.ligandgnn(x=g_l.x,edge_index=g_l.edge_index,edge_attr=g_l.edge_attr,batch=g_l.batch)
        inter_emb = self.heterognn(g_pl.x_dict, g_pl.edge_index_dict, g_pl.edge_attr_dict, g_pl.batch_dict)
        # Now select and average the encoded surface features around each ligand pocket.

        offset=0
        pocket_idx=[]
        mean_pocket_feat=[]
        for bn in set(batch.graph.batch.cpu().numpy()):
            ligand_center =  batch.ligand_center[bn]
            pos = batch.graph.node_pos[batch.graph.batch==bn]
            dists = torch.cdist(pos, ligand_center)
            min_indices = torch.topk(-dists, k=10, dim=0).indices.unique()
            min_indices+= offset
            pocket_idx.append(min_indices)
            offset += len(pos)
            mean_pocket_feat.append(graph.x[min_indices].mean(dim=0))
        
        mean_pocket_feat=torch.vstack(mean_pocket_feat)
        x = torch.cat([l_emb,inter_emb,mean_pocket_feat],dim=1)
        x = self.top_net(x)
        if torch.isnan(x.flatten()).any():
            print('Nan in X, after encoder')
            import pdb
            pdb.set_trace()
            return None
        return x.flatten()
