import os

import numpy as np
import scipy
import torch


def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device("cpu")).numpy()


# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)
    ).coalesce()


# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat
