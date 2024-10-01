import os
import sys

from atom3d.datasets import LMDBDataset
from atom3d.util.formats import get_coordinates_from_df
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..', '..'))

from atomsurf.protein.create_esm import get_esm_embedding_batch
from atomsurf.utils.atom_utils import df_to_pdb
from atomsurf.utils.data_utils import PreprocessDataset
from atomsurf.utils.python_utils import do_all

torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(1)


class PreprocessMSPDataset(PreprocessDataset):
    def __init__(self, data_dir, recompute_pdb=False, recompute_s=False, recompute_g=False, recompute_interfaces=False,
                 mode='train', max_vert_number=100000, face_reduction_rate=0.1):
        # Stuff to get PDBs right
        data_dir = os.path.join(data_dir, mode)
        self.dataset = LMDBDataset(data_dir)
        self.recompute_pdb = recompute_pdb
        self.pdb_dir = os.path.join(data_dir, 'pdb')
        os.makedirs(self.pdb_dir, exist_ok=True)

        super().__init__(data_dir=data_dir, recompute_s=recompute_s, recompute_g=recompute_g,
                         max_vert_number=max_vert_number, face_reduction_rate=face_reduction_rate)

        self.recompute_interfaces = recompute_interfaces

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _extract_mut_idx(df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[(df.chain.values == chain) & (df.residue.values == res)].values
        return torch.LongTensor(idx)

    def __getitem__(self, idx):
        # Extract relevant PDB
        # Name is PDB_CHAINSLEFT_CHAINSRIGHT_MUTATION
        # mutation is like AD56G which means Alanine (A) in chain D resnum 56 (D56) -> Glycine (G)
        lmdb_item = self.dataset[idx]
        system_name = lmdb_item['id']
        pdb, chains_left, chains_right, mutation = system_name.split('_')
        names = [f"{pdb}_{chains_left}", f"{pdb}_{chains_right}",
                 f"{pdb}_{chains_left}_{mutation}", f"{pdb}_{chains_right}_{mutation}"]
        try:
            # Extract relevant dataframes
            orig_df = lmdb_item['original_atoms']
            mut_df = lmdb_item['mutated_atoms']

            # Remove hydrogens as they have weird names like 2HB (instead of HB2)
            orig_df = orig_df[orig_df['element'] != 'H'].reset_index(drop=True)
            mut_df = mut_df[mut_df['element'] != 'H'].reset_index(drop=True)

            left_orig = orig_df[orig_df['chain'].isin(list(chains_left))]
            right_orig = orig_df[orig_df['chain'].isin(list(chains_right))]
            left_mut = mut_df[mut_df['chain'].isin(list(chains_left))]
            right_mut = mut_df[mut_df['chain'].isin(list(chains_right))]
            dfs = [left_orig, right_orig, left_mut, right_mut]

            # Get all pdbs, graphs and surfaces.
            # Let's split those files in separate folders, otherwise it creates a race condition for duplicates
            # First create all dirs
            dump_dirs = [os.path.join(dump, system_name) for dump in [self.pdb_dir,
                                                                      self.out_surf_dir,
                                                                      self.out_agraph_dir,
                                                                      self.out_rgraph_dir]]
            for dir in dump_dirs:
                os.makedirs(dir, exist_ok=True)
            # Then create pdbs and save graphs in a list
            graphs_results = []
            for name, df in zip(names, dfs):
                pdb_path = os.path.join(self.pdb_dir, system_name, f"{name}.pdb")
                surface_dump = os.path.join(self.out_surf_dir, system_name, f'{name}.pt')
                agraph_dump = os.path.join(self.out_agraph_dir, system_name, f'{name}.pt')
                rgraph_dump = os.path.join(self.out_rgraph_dir, system_name, f'{name}.pt')

                df_to_pdb(df, pdb_path, recompute=self.recompute_pdb)
                success = self.path_to_surf_graphs(pdb_path, surface_dump, agraph_dump, rgraph_dump)
                if not success:
                    print(f"Preprocess failed for {system_name} at the surface/graph creation step")
                    return 0
                graphs_results.append((agraph_dump, torch.load(agraph_dump), rgraph_dump, torch.load(rgraph_dump)))

            if not self.recompute_interfaces:
                need_compute_any = False
                for _, agraph, _, rgraph in graphs_results:
                    if not ("misc_features" in agraph.features
                            and "misc_features" in rgraph.features
                            and "interface_node" in agraph.features.misc_features
                            and "interface_node" in rgraph.features.misc_features):
                        need_compute_any = True
                if not need_compute_any:
                    return 1
            # Get interfacial coords
            orig_idx = self._extract_mut_idx(orig_df, mutation)
            mut_idx = self._extract_mut_idx(mut_df, mutation)
            orig_coords = get_coordinates_from_df(orig_df.iloc[orig_idx])
            mut_coords = get_coordinates_from_df(mut_df.iloc[mut_idx])
            orig_coords = torch.from_numpy(orig_coords).float()
            mut_coords = torch.from_numpy(mut_coords).float()

            # Finally, add this information in the graphs
            for i, (agraph_dump, agraph, rgraph_dump, rgraph) in enumerate(graphs_results):
                # Get the right target coords
                interface_coords = orig_coords if i < 2 else mut_coords

                # Get the nearest neighbor on each side for the atomgraph
                dists = torch.cdist(interface_coords, agraph.node_pos)
                k = min(32, len(agraph.node_pos))
                min_idx_agraph = torch.topk(-dists, k=k, dim=1).indices.unique()
                agraph.features.add_misc_features("interface_node", min_idx_agraph)
                torch.save(agraph, open(agraph_dump, 'wb'))

                # Do the same for rgraphs
                dists = torch.cdist(interface_coords, rgraph.node_pos)
                k = min(8, len(rgraph.node_pos))
                min_idx_rgraph = torch.topk(-dists, k=k, dim=1).indices.unique()
                rgraph.features.add_misc_features("interface_node", min_idx_rgraph)
                torch.save(rgraph, open(rgraph_dump, 'wb'))
            return 1
        except Exception as e:
            print(f"Preprocess failed for {system_name} at the interface computation step", e)
            return 0


if __name__ == '__main__':
    recompute = False
    data_dir = '../../../data/msp/MSP-split-by-sequence-identity-30/split-by-sequence-identity-30/data'
    recompute_pdb = False
    recompute_s = False
    recompute_g = False
    recompute_interface = False

    # for mode in ['test']:
    for mode in ['train', 'val', 'test']:
        dataset = PreprocessMSPDataset(data_dir=data_dir, recompute_s=recompute_s, recompute_g=recompute_g,
                                       recompute_pdb=recompute_pdb, recompute_interfaces=recompute_interface,
                                       mode=mode, max_vert_number=100000, face_reduction_rate=0.1)
        do_all(dataset, num_workers=20)

        pdb_dir = os.path.join(data_dir, mode, 'pdb')
        out_esm_dir = os.path.join(data_dir, mode, 'esm')
        os.makedirs(out_esm_dir, exist_ok=True)
        get_esm_embedding_batch(in_pdbs_dir=pdb_dir, dump_dir=out_esm_dir)
