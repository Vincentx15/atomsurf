import os
from torch_geometric.data import Data

from atomsurf.protein.create_esm import get_esm_embedding_single, get_esm_embedding_batch
from atomsurf.utils.data_utils import AtomBatch, PreprocessDataset, pdb_to_surf, pdb_to_graphs
from atomsurf.utils.python_utils import do_all
from atomsurf.utils.wrappers import DefaultLoader, get_default_model

# Set up data paths
pdb_dir = "example_data/pdb"
surface_dir = "example_data/surfaces_0.1"
rgraph_dir = "example_data/rgraph"
esm_dir = "example_data/esm_emb"
example_name = "1ycr"

# Individual computation
# Set up paths
pdb_path = os.path.join(pdb_dir, f"{example_name}.pdb")
surface_dump = os.path.join(surface_dir, f"{example_name}.pt")
rgraph_dump = os.path.join(rgraph_dir, f"{example_name}.pt")

# Pre-compute surface, graphs and esm embeddings
pdb_to_surf(pdb_path, surface_dump)
pdb_to_graphs(pdb_path, rgraph_dump=rgraph_dump)
get_esm_embedding_single(pdb_path, esm_dir)


# Do the same but automatically on a directory
dataset = PreprocessDataset(data_dir="example_data")
do_all(dataset, num_workers=2)
get_esm_embedding_batch(in_pdbs_dir=pdb_dir, dump_dir=esm_dir)

# Load precomputed files
default_loader = DefaultLoader(surface_dir=surface_dir, graph_dir=rgraph_dir, embeddings_dir=esm_dir)
surface, graph = default_loader(example_name)

# Artifically group in a container and "batch"
protein = Data(surface=surface, graph=graph)
batch = AtomBatch.from_data_list([protein, protein])
print(batch)

# Instantiate a model, based on the dimensionality of the input
in_dim_surface, in_dim_graph = surface.x.shape[-1], graph.x.shape[-1]
atomsurf_model = get_default_model(in_dim_surface, in_dim_graph, model_dim=12)

# Encode your input batch !
surface, graph = atomsurf_model(graph=batch.graph, surface=batch.surface)
surface.x  # (total_n_verts, hidden_dim)
graph.x  # (total_n_nodes, hidden_dim)
print(graph.x.shape)
