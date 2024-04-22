import torch
import esm
import os
from atomsurf.protein.graphs import parse_pdb_path,res_type_idx_to_1

def get_esm_embedding_single(pdb_path,dump_dir=None):
    name=pdb_path.split('/')[-1][0:4]
    seq=parse_pdb_path(pdb_path)[0]
    seq=''.join([res_type_idx_to_1[i] for i in seq])
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    tmpdata=[(name,seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(tmpdata)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    embed=results['representations'][33][:,1:-1,:][0]
    if dump_dir!=None:
        torch.save(embed,dump_dir+'/'+name+'_esm.pt')
    return embed
    
def get_esm_embedding_batch(pdb_path,dump_dir):
    for pdb in pdb_path:
        try:
            name=pdb.split('/')[-1][0:4]
            seq=parse_pdb_path(pdb)[0]
            seq=''.join([res_type_idx_to_1[i] for i in seq])
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            tmpdata=[('temp',seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(tmpdata)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])
            embed=results['representations'][33][:,1:-1,:][0]
            torch.save(embed,dump_dir+'/'+name+'_esm.pt')
        except:
            with open('./fail_esm_list.log','a') as f:
                f.write(pdb+'\n')
        # return embeds
if __name__ == "__main__":
    pass