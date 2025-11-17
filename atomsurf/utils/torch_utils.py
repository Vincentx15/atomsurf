import os

import numpy as np
import scipy
import torch
from torch_sparse.tensor import SparseTensor, from_scipy, to_scipy


def safe_to_torch(value):
    """
    :param value:
    :return:
    """
    # Needed to replicate Data default behavior of construction: Data(x=None) -> Data()
    if value is None:
        return None
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
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'x must be a torch tensor or a numpy array, got {type(x)}')
    np_x = x.detach().to(torch.device("cpu")).numpy()
    if dtype is not None:
        np_x = np_x.astype(dtype)
    return np_x


# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    if isinstance(A, torch.Tensor):
        return A
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()


# Numpy sparse to pyg_sparse
def sparse_np_to_pyg(A):
    if isinstance(A, SparseTensor):
        return A
    return from_scipy(A)


# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A, dtype=None):
    if isinstance(A, scipy.sparse.spmatrix):
        return A
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    if not A.is_coalesced():
        old_values = A._values()
        A = A.coalesce()
        new_values = A.values()
        assert old_values.shape == new_values.shape, "Trying to convert uncoalesced tensors"
    indices = toNP(A.indices())
    values = toNP(A.values(), dtype=dtype)
    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()
    return mat


# Pytorch sparse to numpy csc matrix
def sparse_pyg_to_np(A):
    if isinstance(A, scipy.sparse.spmatrix):
        return A
    return to_scipy(A, layout='csr')


def read_sp_mat(npzfile, prefix):
    data = npzfile[prefix + "_data"]
    indices = npzfile[prefix + "_indices"]
    indptr = npzfile[prefix + "_indptr"]
    shape = npzfile[prefix + "_shape"]
    mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
    return mat
