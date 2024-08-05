import os
import sys

from atom3d.util.formats import df_to_bp
import Bio.PDB as bio
import numpy as np
import pandas as pd
import scipy.spatial as spa
import time
import torch


def df_to_pdb(df, out_file_name, discard_hetatm=True, recompute=True):
    """
    Utility function to go from a df object to a PDB file
    :param df:
    :param out_file_name:
    :return:
    """

    def filter_notaa(struct):
        """
        Discard Hetatm, copied from biopython as as_protein() method is not in biopython 2.78
        :param struct:
        :return:
        """
        remove_list = []
        for residue in structure.get_residues():
            if residue.get_id()[0] != ' ' or not bio.Polypeptide.is_aa(residue):
                remove_list.append(residue)

        for residue in remove_list:
            residue.parent.detach_child(residue.id)

        for chain in struct.get_chains():  # Remove empty chains
            if not len(chain.child_list):
                chain.parent.detach_child(chain.id)
        return struct

    if os.path.exists(out_file_name) and not recompute:
        return
    structure = df_to_bp(df)
    structure = filter_notaa(structure) if discard_hetatm else structure
    io = bio.PDBIO()
    io.set_structure(structure)
    io.save(out_file_name)
