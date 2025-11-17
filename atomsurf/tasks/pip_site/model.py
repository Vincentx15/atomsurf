import torch
import torch.nn as nn

from atomsurf.networks.protein_encoder import ProteinEncoder


class PIPsiteNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        # in_features = 128 * 2  # 12
        in_features = hparams_head.encoded_dims
        self.top_net = nn.Sequential(*[
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])

    def forward(self, batch):
        # forward pass
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        surface_2, graph_2 = self.encoder(graph=batch.graph_2, surface=batch.surface_2)
        # update idx
        base_l = torch.cumsum(batch.g1_len, dim=0)
        base_r = torch.cumsum(batch.g2_len, dim=0)
        for i in range(1, len(batch.idx_left)):
            batch.idx_left[i] = (batch.idx_left[i] + base_l[i - 1])
            batch.idx_right[i] = (batch.idx_right[i] + base_r[i - 1])

        runtype='both'
        if isinstance(batch.idx_left, list):
            batch.idx_left = torch.cat(batch.idx_left)
            batch.idx_right = torch.cat(batch.idx_right)
        else:
            batch.idx_left = batch.idx_left.reshape(-1)
            batch.idx_right = batch.idx_right.reshape(-1)
        processed_left = graph_1.x[batch.idx_left]
        processed_right = graph_2.x[batch.idx_right]
        if runtype=='both':
            x = self.top_net(torch.cat([processed_left,processed_right]))
        elif runtype=='l':
            x = self.top_net(processed_left)
        elif runtype=='r':
            x = self.top_net(processed_right)
        return x



class PINDERNet(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        # in_features = 128 * 2  # 12
        in_features = hparams_head.encoded_dims
        self.top_net_site = nn.Sequential(*[
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])
        self.top_net_pair = nn.Sequential(*[
            nn.Linear(in_features*2, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])
    def forward(self, batch):
        # forward pass
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        surface_2, graph_2 = self.encoder(graph=batch.graph_2, surface=batch.surface_2)
        # update idx

        base_l = torch.cumsum(batch.g1_len, dim=0)
        base_r = torch.cumsum(batch.g2_len, dim=0)
        for i in range(1, len(batch.idx_left)):
            batch.idx_left[i] = (batch.idx_left[i] + base_l[i - 1])
            batch.idx_right[i] = (batch.idx_right[i] + base_r[i - 1])
        for i in range(1, len(batch.idx_left_pair)): 
            batch.idx_left_pair[i] = (batch.idx_left_pair[i] + base_l[i - 1])
            batch.idx_right_pair[i] = (batch.idx_right_pair[i] + base_r[i - 1])   
        runtype='both'
        if isinstance(batch.idx_left, list):
            batch.idx_left = torch.cat(batch.idx_left)
            batch.idx_right = torch.cat(batch.idx_right)
        else:
            batch.idx_left = batch.idx_left.reshape(-1)
            batch.idx_right = batch.idx_right.reshape(-1)
        if isinstance(batch.idx_left_pair, list):
            batch.idx_left_pair = torch.cat(batch.idx_left_pair)
            batch.idx_right_pair = torch.cat(batch.idx_right_pair)
        else:
            batch.idx_left_pair = batch.idx_left_pair.reshape(-1)
            batch.idx_right_pair = batch.idx_right_pair.reshape(-1)
        processed_left = graph_1.x[batch.idx_left]
        processed_right = graph_2.x[batch.idx_right]
        if runtype=='both':
            x = self.top_net_site(torch.cat([processed_left,processed_right]))
        elif runtype=='l':
            x = self.top_net_site(processed_left)
        elif runtype=='r':
            x = self.top_net_site(processed_right)

        processed_pair_left = graph_1.x[batch.idx_left_pair]
        processed_pair_right = graph_2.x[batch.idx_right_pair]
        x_pair = torch.cat([processed_pair_left, processed_pair_right], dim=1)
        x_pair = self.top_net_pair(x_pair)
        return x,x_pair


class PINDERNet_seed(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
        # in_features = 128 * 2  # 12
        in_features = hparams_head.encoded_dims
        self.g_embed_head1 = nn.Sequential(*[
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features*2, in_features)
        ])
        self.g_embed_head2 = nn.Sequential(*[
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features*2, in_features)
        ])
        self.g_top_net_site = nn.Sequential(*[
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])
        self.g_top_net_pair = nn.Sequential(*[
            nn.Linear(in_features*2, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])
        self.s_embed_head1 = nn.Sequential(*[
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features*2, in_features)
        ])
        self.s_embed_head2 = nn.Sequential(*[
            nn.Linear(in_features, in_features*2),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features*2, in_features)
        ])
        self.s_top_net_pair = nn.Sequential(*[
            nn.Linear(in_features*2, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])
        self.s_top_net_site = nn.Sequential(*[
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)
        ])
    def forward(self, batch,save_emb):
        # forward pass
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        surface_2, graph_2 = self.encoder(graph=batch.graph_2, surface=batch.surface_2)

        # use 2 head to get 2 embed one and inverted one
        graph_1.x_emb = self.g_embed_head1(graph_1.x)
        graph_2.x_emb = self.g_embed_head1(graph_2.x)
        graph_1.x_inv = self.g_embed_head2(graph_1.x)
        graph_2.x_inv = self.g_embed_head2(graph_2.x)
        
        surface_1.x_emb = self.s_embed_head1(surface_1.x)
        surface_2.x_emb = self.s_embed_head1(surface_2.x)
        surface_1.x_inv = self.s_embed_head2(surface_1.x)
        surface_2.x_inv = self.s_embed_head2(surface_2.x)
        # if save_emb:
        #     for batch_idx in graph_1.batch.unique():
        #         import os
        #         os.makedirs('/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/',exist_ok = True)
        #         print('saving embeds.......')
        #         tmp_id2, tmp_id1 = batch.id[batch_idx].split('--')
        #         torch.save(graph_1.x_emb[graph_1.batch==batch_idx].detach().cpu(),'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_id1+'_graph_emb.pt')
        #         torch.save(graph_2.x_emb[graph_2.batch==batch_idx].detach().cpu(),'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_id2+'_graph_emb.pt')
        #         torch.save(graph_1.x_inv[graph_1.batch==batch_idx].detach().cpu(),'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_id1+'_graph_inv_emb.pt')
        #         torch.save(graph_2.x_inv[graph_2.batch==batch_idx].detach().cpu(),'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_id2+'_graph_inv_emb.pt')
        #         torch.save(surface_1.x_emb[surface_1.batch==batch_idx].detach().cpu(),'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_id1+'_surface_emb.pt')
        #         torch.save(surface_2.x_emb[surface_2.batch==batch_idx].detach().cpu(),'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_id2+'_surface_emb.pt')
        #         torch.save(surface_1.x_inv[surface_1.batch==batch_idx].detach().cpu(),'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_id1+'_surface_inv_emb.pt')
        #         torch.save(surface_2.x_inv[surface_2.batch==batch_idx].detach().cpu(),'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_id2+'_surface_inv_emb.pt')
                
        # update idx
        # return g1, g2 , g1_inv, g2_inv, make sure pos g1,g2_inv similar and g2,g1_inv similar 
        # use g1, g2,  g1_inv, g2_inv in top net 
        # train 4 top net at same time g site g pair s site s pair 
        base_l = torch.cumsum(batch.g1_len, dim=0)
        base_r = torch.cumsum(batch.g2_len, dim=0)

        for i in range(1, len(batch.idx_left)):
            batch.idx_left[i] = (batch.idx_left[i] + base_l[i - 1])
        for i in range(1, len(batch.idx_right)):
            batch.idx_right[i] = (batch.idx_right[i] + base_r[i - 1])    
        for i in range(1, len(batch.idx_left_pair)): 
            batch.idx_left_pair[i] = (batch.idx_left_pair[i] + base_l[i - 1])
            batch.idx_right_pair[i] = (batch.idx_right_pair[i] + base_r[i - 1])  

        
        runtype='both'
        if isinstance(batch.idx_left, list):
            batch.idx_left = torch.cat(batch.idx_left)
            batch.idx_right = torch.cat(batch.idx_right)
        else:
            batch.idx_left = batch.idx_left.reshape(-1)
            batch.idx_right = batch.idx_right.reshape(-1)
        if isinstance(batch.idx_left_pair, list):
            batch.idx_left_pair = torch.cat(batch.idx_left_pair)    
            batch.idx_right_pair = torch.cat(batch.idx_right_pair)
        else:
            batch.idx_left_pair = batch.idx_left_pair.reshape(-1)
            batch.idx_right_pair = batch.idx_right_pair.reshape(-1)
        processed_left = graph_1.x_emb[batch.idx_left]
        processed_right = graph_2.x_emb[batch.idx_right]
        if runtype=='both':
            x = self.g_top_net_site(torch.cat([processed_left,processed_right]))
        elif runtype=='l':
            x = self.g_top_net_site(processed_left)
        elif runtype=='r':
            x = self.g_top_net_site(processed_right)

        processed_pair_left = graph_1.x_emb[batch.idx_left_pair]
        processed_pair_right = graph_2.x_emb[batch.idx_right_pair]
        x_pair = torch.cat([processed_pair_left, processed_pair_right], dim=1)
        x_pair = self.g_top_net_pair(x_pair)
        # use pair lable and train for complementarity
        processed_pair_left_inv= graph_1.x_inv[batch.idx_left_pair]
        processed_pair_right_inv= graph_2.x_inv[batch.idx_right_pair]

        # calculate surface
        s_base_l = torch.cumsum(batch.s1_len, dim=0)
        s_base_r = torch.cumsum(batch.s2_len, dim=0)
        for i in range(1, len(batch.s_site_idxL)):
            batch.s_site_idxL[i] = (batch.s_site_idxL[i] + s_base_l[i - 1])
        for i in range(1, len(batch.s_site_idxR)):
            batch.s_site_idxR[i] = (batch.s_site_idxR[i] + s_base_r[i - 1]) 
               
        for i in range(1, len(batch.s_idx_left)):
            batch.s_idx_left[i] = (batch.s_idx_left[i] + s_base_l[i - 1])
            batch.s_idx_right[i] = (batch.s_idx_right[i] + s_base_r[i - 1])

        if isinstance(batch.s_idx_left, list):
            batch.s_idx_left = torch.cat(batch.s_idx_left)    
            batch.s_idx_right = torch.cat(batch.s_idx_right)
        else:
            batch.s_idx_left = batch.s_idx_left.reshape(-1)
            batch.s_idx_right = batch.s_idx_right.reshape(-1)
        if isinstance(batch.s_site_idxL, list):
            batch.s_site_idxL = torch.cat(batch.s_site_idxL)    
            batch.s_site_idxR = torch.cat(batch.s_site_idxR)
        else:
            batch.s_site_idxL = batch.s_site_idxL.reshape(-1)
            batch.s_site_idxR = batch.s_site_idxR.reshape(-1)

        #surf_site
        surface_processed_site_left = surface_1.x_emb[batch.s_site_idxL]
        surface_processed_site_right = surface_2.x_emb[batch.s_site_idxR]
        if runtype=='both':
            s_x = self.s_top_net_site(torch.cat([surface_processed_site_left,surface_processed_site_right]))
        elif runtype=='l':
            s_x = self.s_top_net_site(surface_processed_site_left)
        elif runtype=='r':
            s_x = self.s_top_net_site(surface_processed_site_right)
        # surf_pair
        surface_processed_pair_left = surface_1.x_emb[batch.s_idx_left]
        surface_processed_pair_right = surface_2.x_emb[batch.s_idx_right]
        s_x_pair = torch.cat([surface_processed_pair_left, surface_processed_pair_right], dim=1)
        s_x_pair = self.s_top_net_pair(s_x_pair)
        surface_processed_pair_left_inv= surface_1.x_inv[batch.s_idx_left]
        surface_processed_pair_right_inv= surface_2.x_inv[batch.s_idx_right]
        # if save_emb:
        #     g_site_L_pred = x[:len(graph_1.batch)]
        #     g_site_R_pred = x[len(graph_1.batch):]
        #     s_site_L_pred = s_x[:len(surface_1.batch)]
        #     s_site_R_pred = s_x[len(surface_1.batch):]
        #     for batch_idx in graph_1.batch.unique():
        #         # import os
        #         tmp_idR, tmp_idL = batch.id[batch_idx].split('--')
        #         print('saving site prediction......')
        #         g_L={}
        #         g_R={}
        #         s_L={}
        #         s_R={}
        #         g_L['pred_logits'] = g_site_L_pred[graph_1.batch==batch_idx].detach().cpu()
        #         g_R['pred_logits'] = g_site_R_pred[graph_2.batch==batch_idx].detach().cpu()
        #         s_L['pred_logits'] = s_site_L_pred[surface_1.batch==batch_idx].detach().cpu()
        #         s_R['pred_logits']= s_site_R_pred[surface_2.batch==batch_idx].detach().cpu()
        #         g_L['gd'] = batch.label_l[batch_idx].detach().cpu()
        #         g_R['gd'] = batch.label_r[batch_idx].detach().cpu()
        #         s_L['gd'] = batch.s_site_label_L[batch_idx].detach().cpu()
        #         s_R['gd'] = batch.s_site_label_R[batch_idx].detach().cpu()
        #         torch.save(g_L,'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_idL+'_g_site.pt')
        #         torch.save(g_R,'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_idR+'_g_site.pt')
        #         torch.save(s_L,'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_idL+'_s_site.pt')
        #         torch.save(s_R,'/work/lpdi/users/ymiao/code/sbatch/pinderlog/testemb_new_euloss/'+tmp_idR+'_s_site.pt')

        return x,x_pair,s_x,s_x_pair,processed_pair_left,processed_pair_right,processed_pair_left_inv,processed_pair_right_inv,surface_processed_pair_left,surface_processed_pair_right,surface_processed_pair_left_inv,surface_processed_pair_right_inv

    def extract_embedding(self, batch):
        # forward pass
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        surface_2, graph_2 = self.encoder(graph=batch.graph_2, surface=batch.surface_2)

        graph_1.x_emb = self.g_embed_head1(graph_1.x)
        graph_2.x_emb = self.g_embed_head1(graph_2.x)
        graph_1.x_inv = self.g_embed_head2(graph_1.x)
        graph_2.x_inv = self.g_embed_head2(graph_2.x)
        
        surface_1.x_emb = self.s_embed_head1(surface_1.x)
        surface_2.x_emb = self.s_embed_head1(surface_2.x)
        surface_1.x_inv = self.s_embed_head2(surface_1.x)
        surface_2.x_inv = self.s_embed_head2(surface_2.x)

        base_l = torch.cumsum(batch.g1_len, dim=0)
        base_r = torch.cumsum(batch.g2_len, dim=0)

        for i in range(1, len(batch.idx_left)):
            batch.idx_left[i] = (batch.idx_left[i] + base_l[i - 1])
        for i in range(1, len(batch.idx_right)):
            batch.idx_right[i] = (batch.idx_right[i] + base_r[i - 1])    
        for i in range(1, len(batch.idx_left_pair)): 
            batch.idx_left_pair[i] = (batch.idx_left_pair[i] + base_l[i - 1])
            batch.idx_right_pair[i] = (batch.idx_right_pair[i] + base_r[i - 1])  

        runtype='both'
        if isinstance(batch.idx_left, list):
            batch.idx_left = torch.cat(batch.idx_left)
            batch.idx_right = torch.cat(batch.idx_right)
        else:
            batch.idx_left = batch.idx_left.reshape(-1)
            batch.idx_right = batch.idx_right.reshape(-1)
        if isinstance(batch.idx_left_pair, list):
            batch.idx_left_pair = torch.cat(batch.idx_left_pair)    
            batch.idx_right_pair = torch.cat(batch.idx_right_pair)
        else:
            batch.idx_left_pair = batch.idx_left_pair.reshape(-1)
            batch.idx_right_pair = batch.idx_right_pair.reshape(-1)
        processed_left = graph_1.x_emb[batch.idx_left]
        processed_right = graph_2.x_emb[batch.idx_right]
        if runtype=='both':
            x = self.g_top_net_site(torch.cat([processed_left,processed_right]))
        elif runtype=='l':
            x = self.g_top_net_site(processed_left)
        elif runtype=='r':
            x = self.g_top_net_site(processed_right)

        processed_pair_left = graph_1.x_emb[batch.idx_left_pair]
        processed_pair_right = graph_2.x_emb[batch.idx_right_pair]
        x_pair = torch.cat([processed_pair_left, processed_pair_right], dim=1)
        x_pair = self.g_top_net_pair(x_pair)
        # use pair lable and train for complementarity
        processed_pair_left_inv= graph_1.x_inv[batch.idx_left_pair]
        processed_pair_right_inv= graph_2.x_inv[batch.idx_right_pair]

        # calculate surface
        s_base_l = torch.cumsum(batch.s1_len, dim=0)
        s_base_r = torch.cumsum(batch.s2_len, dim=0)
        for i in range(1, len(batch.s_site_idxL)):
            batch.s_site_idxL[i] = (batch.s_site_idxL[i] + s_base_l[i - 1])
        for i in range(1, len(batch.s_site_idxR)):
            batch.s_site_idxR[i] = (batch.s_site_idxR[i] + s_base_r[i - 1]) 
               
        for i in range(1, len(batch.s_idx_left)):
            batch.s_idx_left[i] = (batch.s_idx_left[i] + s_base_l[i - 1])
            batch.s_idx_right[i] = (batch.s_idx_right[i] + s_base_r[i - 1])

        if isinstance(batch.s_idx_left, list):
            batch.s_idx_left = torch.cat(batch.s_idx_left)    
            batch.s_idx_right = torch.cat(batch.s_idx_right)
        else:
            batch.s_idx_left = batch.s_idx_left.reshape(-1)
            batch.s_idx_right = batch.s_idx_right.reshape(-1)
        if isinstance(batch.s_site_idxL, list):
            batch.s_site_idxL = torch.cat(batch.s_site_idxL)    
            batch.s_site_idxR = torch.cat(batch.s_site_idxR)
        else:
            batch.s_site_idxL = batch.s_site_idxL.reshape(-1)
            batch.s_site_idxR = batch.s_site_idxR.reshape(-1)

        #surf_site
        surface_processed_site_left = surface_1.x_emb[batch.s_site_idxL]
        surface_processed_site_right = surface_2.x_emb[batch.s_site_idxR]
        if runtype=='both':
            s_x = self.s_top_net_site(torch.cat([surface_processed_site_left,surface_processed_site_right]))
        elif runtype=='l':
            s_x = self.s_top_net_site(surface_processed_site_left)
        elif runtype=='r':
            s_x = self.s_top_net_site(surface_processed_site_right)
        # surf_pair
        surface_processed_pair_left = surface_1.x_emb[batch.s_idx_left]
        surface_processed_pair_right = surface_2.x_emb[batch.s_idx_right]
        s_x_pair = torch.cat([surface_processed_pair_left, surface_processed_pair_right], dim=1)
        s_x_pair = self.s_top_net_pair(s_x_pair)
        surface_processed_pair_left_inv= surface_1.x_inv[batch.s_idx_left]
        surface_processed_pair_right_inv= surface_2.x_inv[batch.s_idx_right]

        g_site_L_pred = x[:len(graph_1.batch)]
        g_site_R_pred = x[len(graph_1.batch):]
        s_site_L_pred = s_x[:len(surface_1.batch)]
        s_site_R_pred = s_x[len(surface_1.batch):]


        return graph_1,graph_2,surface_1,surface_2,g_site_L_pred,g_site_R_pred,s_site_L_pred,s_site_R_pred