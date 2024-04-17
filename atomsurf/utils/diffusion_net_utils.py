import os

import numpy as np
import scipy
import torch


def safe_to_torch(value):
    """
    :param value:
    :return:
    """
    if not isinstance(value, torch.Tensor):
        value = np.asarray(value)
        value = torch.from_numpy(value)
    return value


def toNP(x, dtype=None):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    np_x = x.detach().to(torch.device("cpu")).numpy()
    if dtype is not None:
        np_x = np_x.astype(dtype)
    return np_x


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
def sparse_torch_to_np(A, dtype=None):
    if isinstance(A, scipy.sparse.spmatrix):
        return A
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices(), dtype=dtype)
    values = toNP(A.values(), dtype=dtype)

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()
    return mat


def read_sp_mat(npzfile, prefix):
    data = npzfile[prefix + "_data"]
    indices = npzfile[prefix + "_indices"]
    indptr = npzfile[prefix + "_indptr"]
    shape = npzfile[prefix + "_shape"]
    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
    return mat
