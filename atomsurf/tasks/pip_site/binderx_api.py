# surface related 
import sys
sys.path.append('/work/lpdi/users/ymiao/code/newcode_dmasif_onfly_binderx/atomsurf/')

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

from atomsurf.utils.geometric_processing import atoms_to_points_normals
from atomsurf.network_utils.misc_arch.dmasif_utils.geometry_processing import curvatures
from atomsurf.protein.residue_graph import ResidueGraphBuilder, RGraphBatch
from atomsurf.protein.atom_graph import AtomGraphBuilder, AGraphBatch
from atomsurf.tasks.pip_site.pl_model import PINDERModule_seed
from atomsurf.protein.constant import *

import pdb
def batched_select(params, indices, dim=None, batch_dims=0):
    params_shape, indices_shape = list(params.shape), list(indices.shape)
    assert params_shape[:batch_dims] == indices_shape[:batch_dims]

    def _permute(dim, dim1, dim2):
        permute = []
        for i in range(dim):
            if i == dim1:
                permute.append(dim2)
            elif i == dim2:
                permute.append(dim1)
            else:
                permute.append(i)
        return permute

    if dim is not None and dim != batch_dims:
        params_permute = _permute(len(params_shape), dim1=batch_dims, dim2=dim)
        indices_permute = _permute(len(indices_shape), dim1=batch_dims, dim2=dim)
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)
        params_shape, indices_shape = list(params.shape), list(indices.shape)

    params, indices = torch.reshape(params, params_shape[:batch_dims+1] + [-1]), torch.reshape(indices, list(indices_shape[:batch_dims]) + [-1, 1])

    # indices = torch.tile(indices, params.shape[-1:])
    indices = indices.repeat([1] * (params.ndim - 1) + [params.shape[-1]])

    batch_params = torch.gather(params, batch_dims, indices.to(dtype=torch.int64))

    output_shape = params_shape[:batch_dims] + indices_shape[batch_dims:] + params_shape[batch_dims+1:]

    if dim is not None and dim != batch_dims:
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)

    return torch.reshape(batch_params, output_shape)

def transform_seq_to_atomsurf_res(seq):
    convert_aatype_table = np.zeros((len(resname_to_idx),))
    for a, i in resname_to_idx.items():
        convert_aatype_table[i] = atom_surf_res_type_dict[a]
    convert_aatype_table = torch.tensor(convert_aatype_table, dtype=torch.float32).reshape((-1,1))    
    atomsurf_seq = F.embedding(seq, convert_aatype_table.to(device=seq.device)).to(dtype=torch.int64)
    atomsurf_seq = torch.squeeze(atomsurf_seq, dim=-1)
    return atomsurf_seq


@torch.no_grad()
def compute_surface_model(batch, model, graph_type='rgraph'): # 'agraph' or 'rgraph'
    B, N = batch['mask'].shape
    device = batch['batch_pred_seq'].device
    pred_seq = transform_seq_to_atomsurf_res(batch['batch_pred_seq'])
    seq_idx = pred_seq.cpu()
    # seq_pred_atomsurf = [[atom_surf_res_type_dict[idx_to_resname[int(x)]] for x in row] for row in dummy_data_batch['seq']]
    # seq_pred_atomsurf =torch.tensor(seq_pred_atomsurf)

    aa_14_mask = batched_select(atomsurf_atom14_mask.to(device), pred_seq).to(device)
    aa_14_type = batched_select(atomsurf_atom14_type.to(device), pred_seq).to(device)
    aa_14_name = restype_name_to_atom14_names_np[seq_idx]
    resid_14_matrix = torch.arange(N, device=device).unsqueeze(0).unsqueeze(-1).expand(B, N, 14)

    rg_builder = ResidueGraphBuilder(add_esm=False)
    atomsurf_agraph_list = []
    atomsurf_rgraph_list = []

    for b in range(B):
        mask_b = batch['mask'][b]
        coords_b = batch['batch_pred_atom14'][b][mask_b]  # (M, 14, 3)
        type_b = aa_14_type[b][mask_b]                     # (M, 14)
        mask14_b = aa_14_mask[b][mask_b].bool()            # (M, 14)

        flat_coords = coords_b.view(-1, 3)
        flat_types = type_b.view(-1)
        flat_mask = mask14_b.view(-1)

        atom_coords = flat_coords[flat_mask]
        atom_types = flat_types[flat_mask].long()
        onehot_atypes = F.one_hot(atom_types, num_classes=12).float()

        atomsurf_agraph_list.append(Data(x=onehot_atypes, node_pos=atom_coords))
        seq_b = batch['batch_pred_seq'][b][mask_b]
        
        res_names = [idx_to_resname[i.item()] for i in seq_b]
        amino_types = [atom_surf_res_type_dict[r] for r in res_names]

        atom_names = aa_14_name[b][mask_b.cpu()].reshape(-1)[flat_mask.cpu()]
        amino_ids = resid_14_matrix[b][mask_b].reshape(-1)[flat_mask]

        rgraph = rg_builder.arrays_to_resgraph(
            (np.array(amino_types), amino_ids.cpu().numpy(), atom_names, atom_coords.cpu().numpy())
        )
        rgraph.expand_features(remove_feats=True, feature_keys='all', oh_keys='all', feature_expander=None)
        atomsurf_rgraph_list.append(rgraph)

        atomsurf_agraph = Batch.from_data_list(atomsurf_agraph_list)

        points, normals, batch_pts = atoms_to_points_normals(
            atomsurf_agraph.node_pos, atomsurf_agraph.batch,
            distance=1.05, smoothness=0.5, resolution=2.5, nits=4,
            atomtypes=atomsurf_agraph.x[:, -12:], sup_sampling=20, variance=0.1,
        )

        P_curvs = curvatures(
            points, triangles=None, normals=normals,
            scales=[1.0, 2.0, 3.0, 5.0, 10.0], batch=batch_pts
        )

        # Build surface batch
        surface_data_list = []
        for b_id in batch_pts.unique(sorted=True):
            mask_b = batch_pts == b_id
            surface_data_list.append(Data(
                verts=points[mask_b],
                n_verts=mask_b.sum().item(),
                x=P_curvs[mask_b],
                num_nodes=mask_b.sum().item(),
                vnormals=normals[mask_b],
            ))

    surface_batch = Batch.from_data_list(surface_data_list)
    rgraph_batch = RGraphBatch.batch_from_data_list(atomsurf_rgraph_list)

    surface, graph = model(
        graph=rgraph_batch.to(device),
        surface=surface_batch.to(device)
    )

    dense_x, mask = to_dense_batch(graph.x, graph.batch)
    return dense_x

if __name__ == '__main__':
    model_path = '/work/lpdi/users/ymiao/code/newcode_dmasif_onfly_binderx/ckpt/test_rgraph_onfly.ckpt'
    as_model = PINDERModule_seed.load_from_checkpoint(model_path) 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    as_model.to(device)
    as_model.eval()
    surface_encoder= as_model.model.encoder
    dummy_data_batch = torch.load('/work/lpdi/users/tian/git_repo/BinderX/debug_surface_input.pt')
    for k, v in dummy_data_batch.items(): # put to cuda 
        if isinstance(v, torch.Tensor):
            dummy_data_batch[k] = v.cuda(non_blocking=True)
    compute_surface_model(dummy_data_batch,surface_encoder)