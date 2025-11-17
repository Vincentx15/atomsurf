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
    def extract_embedding_single(self, batch):
        # forward pass
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        graph_1.x_emb = self.g_embed_head1(graph_1.x)
        graph_1.x_inv = self.g_embed_head2(graph_1.x)

        surface_1.x_emb = self.s_embed_head1(surface_1.x)
        surface_1.x_inv = self.s_embed_head2(surface_1.x)

        base_l = torch.cumsum(batch.g1_len, dim=0)


        for i in range(1, len(batch.idx_left)):
            batch.idx_left[i] = (batch.idx_left[i] + base_l[i - 1])

        if isinstance(batch.idx_left, list):
            batch.idx_left = torch.cat(batch.idx_left)
        else:
            batch.idx_left = batch.idx_left.reshape(-1)

        processed_left = graph_1.x_emb[batch.idx_left]
        x = self.g_top_net_site(torch.cat([processed_left]))

        # calculate surface
        s_base_l = torch.cumsum(batch.s1_len, dim=0)
        for i in range(1, len(batch.s_site_idxL)):
            batch.s_site_idxL[i] = (batch.s_site_idxL[i] + s_base_l[i - 1])

        if isinstance(batch.s_site_idxL, list):
            batch.s_site_idxL = torch.cat(batch.s_site_idxL)    
        else:
            batch.s_site_idxL = batch.s_site_idxL.reshape(-1)

        #surf_site
        surface_processed_site_left = surface_1.x_emb[batch.s_site_idxL]
        s_x = self.s_top_net_site(surface_processed_site_left)
        g_site_L_pred = x[:len(graph_1.batch)]
        s_site_L_pred = s_x[:len(surface_1.batch)]

        return graph_1,surface_1,g_site_L_pred,s_site_L_pred

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool,global_max_pool
import random
class EfficientCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(EfficientCrossAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Normalization & dropout
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(self, query_nodes, key_nodes, value_nodes, query_batch, key_batch):
        """
        Args:
            query_nodes: [total_query_nodes, embed_dim]
            key_nodes: [total_key_nodes, embed_dim]
            value_nodes: [total_value_nodes, embed_dim]
            query_batch: [total_query_nodes] - batch indices for query nodes
            key_batch: [total_key_nodes] - batch indices for key nodes
        """
        batch_size = query_batch.max().item() + 1

        # Pre-norm (important for stability)
        Q = self.q_proj(self.norm_q(query_nodes))
        K = self.k_proj(self.norm_k(key_nodes))
        V = self.v_proj(self.norm_v(value_nodes))

        # Preallocate aligned output tensor
        output = torch.zeros_like(Q)

        for batch_idx in range(batch_size):
            query_mask = query_batch == batch_idx
            key_mask = key_batch == batch_idx

            if not query_mask.any() or not key_mask.any():
                continue

            q_batch = Q[query_mask].unsqueeze(0)  # [1, num_query_nodes, embed_dim]
            k_batch = K[key_mask].unsqueeze(0)    # [1, num_key_nodes, embed_dim]
            v_batch = V[key_mask].unsqueeze(0)    # [1, num_value_nodes, embed_dim]

            # Multi-head reshape
            q_batch = q_batch.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k_batch = k_batch.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_batch = v_batch.view(1, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention
            scores = torch.matmul(q_batch, k_batch.transpose(-2, -1)) * self.scale
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)  # dropout on weights

            attn_output = torch.matmul(attn_weights, v_batch)

            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(1, -1, self.embed_dim)
            attn_output = self.out_proj(attn_output.squeeze(0))
            attn_output = self.dropout(attn_output)

            # Residual + post-norm
            out_nodes = self.norm_out(attn_output + query_nodes[query_mask])
            output[query_mask] = out_nodes

        return output


class PINDERNet_seed_ppi(torch.nn.Module):
    def __init__(self, hparams_encoder, hparams_head):
        super().__init__()
        self.hparams_head = hparams_head
        self.hparams_encoder = hparams_encoder
        self.encoder = ProteinEncoder(hparams_encoder)
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
        # self.norm = nn.LayerNorm(128)
        self.ppi_linear = nn.Sequential(nn.Linear(hparams_encoder['blocks'][0]['kwargs']['graph_feat_dim']+in_features, in_features*2),nn.Linear(in_features*2, in_features))
        self.cross_attn = EfficientCrossAttentionBlock(in_features, 4)
        self.ppi_mlp = nn.Sequential(
            nn.Linear(in_features * 2, in_features),
            nn.ReLU(),
            nn.Dropout(p=hparams_head.dropout),
            nn.Linear(in_features, 1)  
        )
    def forward(self, batch,save_emb):
        # forward pass
        initial_feat1= batch.graph_1.x
        initial_feat2= batch.graph_2.x
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

        # add ppi prediction
        # 对所有节点预测 site score（不只是采样的节点）
        site_scores_1 = self.g_top_net_site(graph_1.x_emb)  # [num_nodes_1, 1]
        site_scores_2 = self.g_top_net_site(graph_2.x_emb)  # [num_nodes_2, 1]
        
        # 转换为权重（使用 sigmoid 归一化到 0-1）
        site_weights_1 = torch.sigmoid(site_scores_1)  # [num_nodes_1, 1]
        site_weights_2 = torch.sigmoid(site_scores_2)  # [num_nodes_2, 1]
        
        batch_size = graph_1.batch.max().item() + 1
        ppi_feat1 = self.ppi_linear(torch.cat([initial_feat1,graph_1.x_emb],dim=-1))
        ppi_feat2 = self.ppi_linear(torch.cat([initial_feat2,graph_2.x_emb],dim=-1)) 
        attn1_pos = self.cross_attn(
            ppi_feat1, ppi_feat2, ppi_feat2, 
            graph_1.batch, graph_2.batch
        )
        attn2_pos = self.cross_attn(
            ppi_feat2, ppi_feat1, ppi_feat1,
            graph_2.batch, graph_1.batch
        )
        # 应用 site 权重（element-wise multiplication）
        weighted_attn1_pos = attn1_pos * site_weights_1  # 强调重要的 site
        weighted_attn2_pos = attn2_pos * site_weights_2
        
        # 加权池化
        graph_repr1_pos = global_max_pool(weighted_attn1_pos, graph_1.batch)
        graph_repr2_pos = global_max_pool(weighted_attn2_pos, graph_2.batch)
        
        # Combine representations
        combined_pos = torch.cat([graph_repr1_pos, graph_repr2_pos], dim=1)
        alt_combined_pos = torch.cat([graph_repr2_pos, graph_repr1_pos], dim=1)
        combined_pos = 0.5 * (combined_pos + alt_combined_pos)
        
        # Predict positive pairs
        out_pos = self.ppi_mlp(combined_pos)
        labels_pos = torch.ones(batch_size, 1, device=out_pos.device)
        
        # 3. 对负样本对也应用相同的加权策略
        graph_1_list = []
        graph_2_list = []
        site_weights_1_list = []
        site_weights_2_list = []
        
        for i in range(batch_size):
            mask_1 = (graph_1.batch == i)
            mask_2 = (graph_2.batch == i)
            # graph_1_list.append(graph_1.x_emb[mask_1])
            # graph_2_list.append(graph_2.x_emb[mask_2]) #
            graph_1_list.append(ppi_feat1[mask_1])
            graph_2_list.append(ppi_feat2[mask_2])
            site_weights_1_list.append(site_weights_1[mask_1])
            site_weights_2_list.append(site_weights_2[mask_2])
        
        # 生成负样本对
        all_negative_pairs = [(i, j) for i in range(batch_size) for j in range(batch_size) if i != j]
        num_negative_samples = min(2 * batch_size, len(all_negative_pairs))
        sampled_negative_pairs = random.sample(all_negative_pairs, num_negative_samples)
        
        out_neg_list = []
        
        for i, j in sampled_negative_pairs:
            g1_nodes = graph_1_list[i]
            g2_nodes = graph_2_list[j]
            sw1 = site_weights_1_list[i]
            sw2 = site_weights_2_list[j]
            
            temp_batch_1 = torch.zeros(g1_nodes.shape[0], dtype=torch.long, device=g1_nodes.device)
            temp_batch_2 = torch.zeros(g2_nodes.shape[0], dtype=torch.long, device=g2_nodes.device)
            
            attn1_neg = self.cross_attn(
                g1_nodes, g2_nodes, g2_nodes,
                temp_batch_1, temp_batch_2
            )
            attn2_neg = self.cross_attn(
                g2_nodes, g1_nodes, g1_nodes,
                temp_batch_2, temp_batch_1
            )
            
            # 应用 site 权重
            weighted_attn1_neg = attn1_neg * sw1
            weighted_attn2_neg = attn2_neg * sw2
            
            # Global pooling
            graph_repr1_neg = global_max_pool(weighted_attn1_neg, temp_batch_1)
            graph_repr2_neg = global_max_pool(weighted_attn2_neg, temp_batch_2)
            
            combined_neg = torch.cat([graph_repr1_neg, graph_repr2_neg], dim=1)
            alt_combined_neg = torch.cat([graph_repr2_neg, graph_repr1_neg], dim=1)
            combined_neg = 0.5 * (combined_neg + alt_combined_neg)
            
            out_neg = self.ppi_mlp(combined_neg)
            out_neg_list.append(out_neg)
        
        # 合并所有负样本的预测结果
        out_neg = torch.cat(out_neg_list, dim=0)  # [num_negative_samples, 1]
        labels_neg = torch.zeros(num_negative_samples, 1, device=out_neg.device)
        ppi_all_outputs = torch.cat([out_pos, out_neg], dim=0)  # [batch_size + num_negative_samples, 1]
        ppi_all_labels = torch.cat([labels_pos, labels_neg], dim=0)  # [batch_size + num_negative_samples, 1]

        

        return x,x_pair,s_x,s_x_pair,processed_pair_left,processed_pair_right,processed_pair_left_inv,processed_pair_right_inv,surface_processed_pair_left,surface_processed_pair_right,surface_processed_pair_left_inv,surface_processed_pair_right_inv,ppi_all_outputs,ppi_all_labels

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
    def extract_embedding_single(self, batch):
        # forward pass
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        graph_1.x_emb = self.g_embed_head1(graph_1.x)
        graph_1.x_inv = self.g_embed_head2(graph_1.x)

        surface_1.x_emb = self.s_embed_head1(surface_1.x)
        surface_1.x_inv = self.s_embed_head2(surface_1.x)

        base_l = torch.cumsum(batch.g1_len, dim=0)


        for i in range(1, len(batch.idx_left)):
            batch.idx_left[i] = (batch.idx_left[i] + base_l[i - 1])

        if isinstance(batch.idx_left, list):
            batch.idx_left = torch.cat(batch.idx_left)
        else:
            batch.idx_left = batch.idx_left.reshape(-1)

        processed_left = graph_1.x_emb[batch.idx_left]
        x = self.g_top_net_site(torch.cat([processed_left]))

        # calculate surface
        s_base_l = torch.cumsum(batch.s1_len, dim=0)
        for i in range(1, len(batch.s_site_idxL)):
            batch.s_site_idxL[i] = (batch.s_site_idxL[i] + s_base_l[i - 1])

        if isinstance(batch.s_site_idxL, list):
            batch.s_site_idxL = torch.cat(batch.s_site_idxL)    
        else:
            batch.s_site_idxL = batch.s_site_idxL.reshape(-1)

        #surf_site
        surface_processed_site_left = surface_1.x_emb[batch.s_site_idxL]
        s_x = self.s_top_net_site(surface_processed_site_left)
        g_site_L_pred = x[:len(graph_1.batch)]
        s_site_L_pred = s_x[:len(surface_1.batch)]

        return graph_1,surface_1,g_site_L_pred,s_site_L_pred