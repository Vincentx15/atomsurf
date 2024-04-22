import os
import sys

import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

from atomsurf.protein.create_surface import get_surface
from atomsurf.protein.create_operators import compute_operators
from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.protein.atom_graph import AtomGraphBuilder
from atomsurf.protein.residue_graph import ResidueGraphBuilder


def create_protein(pdb_path, dump_ply, dump_surf, dump_agraph, dump_rgraph):
    # Create surface
    # surface = SurfaceObject.from_pdb_path(pdb_path, out_ply_path=dump_ply)
    # surface.add_geom_feats()
    # surface.save_torch(dump_surf)

    # create atomgraph
    agraph_builder = AtomGraphBuilder()
    agraph = agraph_builder.pdb_to_atomgraph(pdb_path)
    torch.save(agraph, open(dump_agraph, 'wb'))

    # create atomgraph
    rgraph_builder = ResidueGraphBuilder()
    rgraph = rgraph_builder.pdb_to_resgraph(pdb_path,esmpath)
    torch.save(rgraph, open(dump_rgraph, 'wb'))


if __name__ == '__main__':
    pdb_path = "../../data/example_files/4kt3.pdb"
    dump_ply = "../../data/example_files/test_main.ply"
    dump_surf = "../../data/example_files/test_main.surf"
    dump_agraph = "../../data/example_files/test_main.agraph"
    dump_rgraph = "../../data/example_files/test_main.rgraph"
    create_protein(pdb_path=pdb_path,
                   dump_ply=dump_ply,
                   dump_surf=dump_surf,
                   dump_agraph=dump_agraph,
                   dump_rgraph=dump_rgraph)

