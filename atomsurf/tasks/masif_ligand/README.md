## Ours, data production
We start from: [Zenodo](https://zenodo.org/records/7686423)
Then we will use data/masif_ligand/{raw_data_MasifLigand, dataset_MasifLigand} to produce our data by running 
`python preprocess.py`

## Context

The original data is proposed in Masif, and a raw data of 10G can be downloaded from [Zenodo](https://zenodo.org/records/2625420). 
Those include all PDB with a cofactor (over 10k PDBs), along with a mesh and label. 
Actually only a fraction of those are used, as defined by the splits.

A more compact version is proposed by HMR, that includes only the required PDBs as three .txt files, along with
pdb files and the cofactor types and coords grouped by PDB, also downloadable from [Zenodo](https://zenodo.org/records/7686423)

Then, HMR uses 4 steps to turn those PDBs into small meshes around the ligand
Step 1/2 just go through the .pdb files to create a mesh stored as a .npz.
Step 3 uses those meshes and ligands files that give type+coordinates for the ligands, and saves the result as patches.
Step 4 takes the patch mesh and the initial .xyzrn files and computes geometric and chemical features.

!! 
We use slightly different surfaces:
Masif:density=3.; hdensity=3.; probe radius=1.5
HMR:density=2.; probe radius=1.
Ours: density=1.; radius=1.5
!!


