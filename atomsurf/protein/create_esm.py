import os
import sys

import esm
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from atomsurf.protein.graphs import parse_pdb_path, res_type_idx_to_1


def compute_one(pdb_path, outpath=None, model_objs=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_objs is None:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
    else:
        model, batch_converter = model_objs
    model.eval()
    model.to(device)
    name = pdb_path.split('/')[-1][0:-4]
    seq = parse_pdb_path(pdb_path)[0]
    seq = ''.join([res_type_idx_to_1[i] for i in seq])
    tmpdata = [(name, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(tmpdata)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    embed = results['representations'][33][:, 1:-1, :][0]
    embed = embed.cpu()
    if outpath is not None:
        torch.save(embed, outpath)
    return embed


def get_esm_embedding_single(pdb_path, esm_path=None, recompute=False, model_objs=None):
    name = pdb_path.split('/')[-1][0:4]
    if esm_path is not None and not os.path.exists(esm_path):
        os.makedirs(esm_path, exist_ok=True)
    embs_path = os.path.join(esm_path, f"{name}_esm.pt")
    if esm_path is not None and not recompute and os.path.exists(embs_path):
        embed = torch.load(embs_path)
    else:
        embed = compute_one(pdb_path, outpath=embs_path, model_objs=model_objs)
    return embed


def get_esm_embedding_batch_old(all_pdb_path, dump_dir):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_objs = (model, batch_converter)
    for i, pdb_path in enumerate(tqdm(all_pdb_path)):
        try:
            get_esm_embedding_single(pdb_path, esm_path=dump_dir, model_objs=model_objs)
        except IOError:
            with open('./fail_esm_list.log', 'a') as f:
                f.write(pdb_path + '\n')
        # return embeds


class PreProcessPDBDataset(Dataset):
    def __init__(self, in_pdbs_dir, dump_dir, recompute=False):
        os.makedirs(dump_dir, exist_ok=True)
        self.dump_dir = dump_dir
        self.in_pdbs_dir = in_pdbs_dir

        in_pdbs_names = [pdb.split('/')[-1][0:-4] for pdb in os.listdir(in_pdbs_dir) if pdb.endswith('.pdb')]
        if not recompute:
            existing_out_files = set([name[:-7] for name in os.listdir(dump_dir)])
            self.in_pdbs = [pdb for pdb in in_pdbs_names if pdb not in existing_out_files]
        else:
            self.in_pdbs = in_pdbs_names

    def __len__(self):
        return len(self.in_pdbs)

    def __getitem__(self, idx):
        pdb_name = self.in_pdbs[idx]
        pdb_path = os.path.join(self.in_pdbs_dir, f"{pdb_name}.pdb")
        seq = parse_pdb_path(pdb_path)[0]
        seq = ''.join([res_type_idx_to_1[i] for i in seq])
        return pdb_name, seq


def get_esm_embedding_batch(in_pdbs_dir, dump_dir):
    dataset = PreProcessPDBDataset(in_pdbs_dir, dump_dir)
    dataloader = DataLoader(dataset,
                            collate_fn=lambda samples: samples,
                            num_workers=12,
                            batch_size=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    for batch in tqdm(dataloader):
        try:
            names, data = list(map(list, zip(*batch)))
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            batch_tokens = batch_tokens.to(device)

            # Extract per-residue representations (on CPU)
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            token_representations = results["representations"][33]
            for i, (name, tokens_len) in enumerate(zip(names, batch_lens)):
                embeddings = token_representations[i, 1: tokens_len - 1]
                embeddings = embeddings.cpu()
                assert len(data[i]) == len(embeddings)
                out_path = os.path.join(dump_dir, f"{name}_esm.pt")
                torch.save(embeddings, out_path)
            a = 1

        except IOError:
            pass


if __name__ == "__main__":
    in_pdbs_dir = "../../data/masif_site/01-benchmark_pdbs"
    dump_dir = "../../data/masif_site/01-benchmark_esm_embs"
    # in_pdbs = [os.path.join(in_pdbs_dir, pdb) for pdb in os.listdir(in_pdbs_dir) if pdb.endswith('.pdb')]
    # get_esm_embedding_batch_old(all_pdb_path=in_pdbs, dump_dir=out_embs)

    get_esm_embedding_batch(in_pdbs_dir=in_pdbs_dir, dump_dir=dump_dir)
