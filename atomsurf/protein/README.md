This is where the data is processed to create surfaces, resgraphs and atomgraphs, as well as their corresponding features.

The first object we create is a Features object (in `features.py`, that is an extension of the Data() object to also potentially
hold one hot representations (making it more compact), as well as expandable features (to go from res to atom for all
atom graphs)

The code to obtain the surface object is split into three steps: 
- creating a .ply mesh `create_surface.py`
- creating operators `create_operators.py`
- creating a Surface object, possibly with features `Surface.py`. Note that this class does not require pymesh

The code to obtain graphs is split into three files: 
- parsing PDB files and holding generic chemical constants in `graphs.py`
- creating residue-level graphs, along with their features in `residue_graph.py`
- creating atomic-level graphs, along with their features in `atom_graph.py`

Finally, a factorized routine to create all these representations for all graphs is provided in `create_all.py`

