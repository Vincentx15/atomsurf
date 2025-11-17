import torch
from tqdm import tqdm
from .geometry_processing import (
    atoms_to_points_normals,
)
from torch_geometric.utils import to_dense_batch

def iterate_surface_precompute(dataloader, net, device):
    processed_dataset = []
    for it, data in enumerate(tqdm(dataloader)):
        protein=data[4].to(device)
        ligand_center=data[3].to(device)
        P = process(protein,ligand_center, net)
        data[4].gen_xyz = P["xyz"]
        data[4].gen_normals = P["normals"]
        data[4].gen_batch = P["batch"]
        processed_dataset.append(data)
    return processed_dataset


def process(protein_single, ligand_center, net):
    P = {}

    # Atom information:
    P["atoms"] = protein_single.atom_coords
    P["batch_atoms"] = protein_single.atom_coords_batch

    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = protein_single.atom_coords
    P["atomtypes"] = protein_single.atom_types

    if not "gen_xyz" in protein_single.keys():
        P["xyz"], P["normals"], P["batch"] = atoms_to_points_normals(
            P["atoms"],
            P["batch_atoms"],
            atomtypes=P["atomtypes"],
            num_atoms=22,
            resolution=1.0,
            sup_sampling=20,
            distance=1.05,
        )
        P["xyz"], P["normals"], P["batch"] = select_pocket(P,ligand_center)
    else:
        P["xyz"]=protein_single.gen_xyz
        P["normals"]=protein_single.gen_normals
        P["batch"]=protein_single.gen_batch

    return P

def extract_single(P_batch, number):
    P = {}  
    suface_batch = P_batch["batch"] == number

    P["batch"] = P_batch["batch"][suface_batch]

    # Surface information:
    P["xyz"] = P_batch["xyz"][suface_batch]
    P["normals"] = P_batch["normals"][suface_batch]

    return P

def select_pocket(P_batch, ligand_center):

    surface_list = []
    batch_list = []
    normal_list = []
    protein_batch_size = P_batch["batch_atoms"][-1].item() + 1
    for i in range(protein_batch_size):
        P = extract_single(P_batch, i)

        distances = torch.norm(P["xyz"] - ligand_center[i].squeeze(), dim=1)
        sorted_indices = torch.argsort(distances)
        point_nums = 512
        closest_protein_indices = sorted_indices[:point_nums]

        # Append surface embeddings and batches to the lists
        surface_list.append(P["xyz"][closest_protein_indices])
        normal_list.append(P["normals"][closest_protein_indices])
        batch_list.append(P["batch"][:closest_protein_indices.shape[0]])

    p_xyz = torch.cat(surface_list, dim=0)
    p_batch = torch.cat(batch_list, dim=0)
    p_normals = torch.cat(normal_list, dim=0)

    return p_xyz, p_normals, p_batch


def iterate(net, ligand_center, protein, device='cuda'):
    
    P_processed = process(protein, ligand_center, net)
    
    outputs = net(P_processed)
    surface_emb, mask = to_dense_batch(outputs["embedding"], outputs["batch"])

    return surface_emb


