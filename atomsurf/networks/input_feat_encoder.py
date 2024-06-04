import torch
import torch.nn as nn
from torch_scatter import scatter


class ChemGeomFeatEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        h_dim = hparams.h_dim
        dropout = hparams.dropout
        num_gdf = hparams.num_gdf
        num_signatures = hparams.num_signatures
        self.use_neigh = hparams.use_neigh
        chem_feat_dim = hparams.graph_feat_dim
        geom_feat_dim = hparams.surface_feat_dim
        chem_feat_dim_surf = hparams.chem_feat_dim_surf
        # chem feat
        # chem_feat_dim = 2 + num_gdf * 2
        self.chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim,  h_dim),#2 * h_dim
            nn.Dropout(dropout),
            nn.BatchNorm1d( h_dim),#2 * h_dim
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # # geom feats
        # # geom_input_dim = num_gdf * 2 + num_signatures
        # self.geom_mlp = nn.Sequential(
        #     nn.Linear(geom_feat_dim, h_dim // 2),
        #     nn.Dropout(dropout),
        #     nn.BatchNorm1d(h_dim // 2),
        #     nn.SiLU(),
        #     nn.Linear(h_dim // 2, h_dim // 2),
        #     nn.Dropout(dropout),
        #     nn.BatchNorm1d(h_dim // 2),
        # ) 
        self.surf_chem_mlp = nn.Sequential(
            nn.Linear(chem_feat_dim_surf, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 2*h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2*h_dim),
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        # geom feats
        # geom_input_dim = num_gdf * 2 + num_signatures
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_feat_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(h_dim),
        )

        if self.use_neigh:
            # chem + geom feats
            self.feat_mlp = nn.Sequential(
                nn.Linear(h_dim + h_dim, h_dim),
                # nn.Linear(h_dim + h_dim // 2, h_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(h_dim),
                nn.SiLU(),
                nn.Linear(h_dim, h_dim),
                nn.Dropout(dropout),
                nn.BatchNorm1d(h_dim),
            )

    def forward(self, surface, graph):
        # chem_feats, geom_feats, nbr_vids = graph.chem_feats, surface.geom_feats, surface.nbr_vids
        # surface_in = [mini_surface.x for mini_surface in surface]
        # chem_feats, geom_feats = graph.x, torch.concatenate(surface_in, dim=-2)
        ####### preprocess test chem feat#######
        # from sklearn.neighbors import BallTree 
        # from atomsurf.utils.data_utils import GaussianDistanceNP
        # from atomsurf.network_utils.diffusion_net.geometry import mesh_vertex_normals
        # chem_feats = graph.x
        # dist_gdf= GaussianDistanceNP(start=0., stop=8., num_centers=8)
        # angular_gdf =  GaussianDistanceNP(start=-1., stop=1., num_centers=8)
        # atom_bt = BallTree(graph.node_pos)
        # vert_nbr_atoms = 16
        # vert_nbr_dist, vert_nbr_ind = atom_bt.query(surface.verts, k=vert_nbr_atoms)
        # nbr_vid = np.concatenate([[i] * len(vert_nbr_ind[i]) for i in range(len(vert_nbr_ind))])
        # surface.nbr_vid= torch.from_numpy(nbr_vid)
        # dist_flat = np.concatenate(vert_nbr_dist, axis=0)
        # ind_flat = np.concatenate(vert_nbr_ind, axis=0)
        # chem_featstmp= chem_feats.numpy()
        # testchem_feats = [chem_featstmp[ind_flat]]
        # testchem_feats.append(dist_gdf.expand(dist_flat))
        # nbr_vec = graph.node_pos[ind_flat] - surface.verts[nbr_vid]
        # vnormals = mesh_vertex_normals(surface.verts.numpy(),surface.faces.numpy())
        # nbr_vnormals = vnormals[nbr_vid]
        # nbr_angular = np.einsum('vj,vj->v', 
        #                             nbr_vec / np.linalg.norm(nbr_vec, axis=-1, keepdims=True), 
        #                             nbr_vnormals)
        # nbr_angular_gdf = angular_gdf.expand(nbr_angular)
        # testchem_feats.append(nbr_angular_gdf)
        # testchem_feats = np.concatenate(testchem_feats, axis=-1)
        # testchem_feats = torch.from_numpy(testchem_feats)
        # surface.testchem_feats = testchem_feats
        b_nbr_vids = []
        b_vert_nbr_ind=[]
        base_idx = 0
        for nbrvid,numverts in zip(surface.nbr_vid,surface.n_verts):
            b_nbr_vids.append(torch.from_numpy(nbrvid+int(base_idx)))
            # b_vert_nbr_ind.append(torch.from_numpy(vertnbrind+int(base_idx)))
            base_idx+=numverts
        base_idx = 0
        for vertnbrind,numgraph in zip(surface.vert_nbr_ind,graph.num_res):
            b_vert_nbr_ind.append(torch.from_numpy(vertnbrind+int(base_idx)))
            base_idx+=numgraph
        b_nbr_vids = torch.cat(b_nbr_vids).to(surface.x.device)
        b_vert_nbr_ind = torch.cat(b_vert_nbr_ind).to(surface.x.device)
        
        from atomsurf.utils.data_utils import GaussianDistance
        dist_gdf= GaussianDistance(start=0., stop=8., num_centers=8)
        angular_gdf =  GaussianDistance(start=-1., stop=1., num_centers=8)
        chem_feats = graph.x
        testchem_feats = [chem_feats[b_vert_nbr_ind]]
        testchem_feats.append(dist_gdf(surface.vert_nbr_dist[..., None]))
        nbr_vnormals = surface.vnormals[b_nbr_vids]
        nbr_vec = graph.node_pos[b_vert_nbr_ind] - surface.verts[b_nbr_vids]
        nbr_angular = torch.einsum('vj,vj->v', nbr_vec / torch.linalg.norm(nbr_vec, axis=-1, keepdims=True), nbr_vnormals)
        nbr_angular_gdf = angular_gdf(nbr_angular[..., None])
        testchem_feats.append(nbr_angular_gdf)
        testchem_feats = torch.cat(testchem_feats, axis=-1)
        # import pdb
        # pdb.set_trace() 
        ##############end ####################

        chem_feats_graph, geom_feats = graph.x, surface.x
        # chem_feats_suf = surface.testchem_feats
        chem_feats_suf= testchem_feats

        # geometric features
        h_geom = self.geom_mlp(geom_feats)

        # chemical features
        chem_feats_graph = self.chem_mlp(chem_feats_graph)
        chem_feats_suf=chem_feats_suf.to(geom_feats.device,dtype=geom_feats.dtype)
        chem_feats_suf=self.surf_chem_mlp(chem_feats_suf)
        nbr_filter, nbr_core = chem_feats_suf.chunk(2, dim=-1)


        # If self-filter
        if self.use_neigh:
            nbr_vids = b_nbr_vids  # TODO: implement/fix
            nbr_filter = self.sigmoid(nbr_filter)
            nbr_core = self.softplus(nbr_core)
            chem_feats_suf = nbr_filter * nbr_core
            chem_feats_suf = scatter(chem_feats_suf, nbr_vids, dim=0, reduce="sum")
            h_geom = self.feat_mlp(torch.cat((chem_feats_suf, h_geom), dim=-1))
        else:
            h_geom = h_geom
            h_chem = nbr_filter

        # surface_out = torch.split(h_geom, [len(x) for x in surface_in], dim=-2)
        # for mini_surf_out, mini_surf in zip(surface_out, surface):
        #     mini_surf.x = mini_surf_out
        graph.x = chem_feats_graph
        surface.x = h_geom
        return surface, graph
