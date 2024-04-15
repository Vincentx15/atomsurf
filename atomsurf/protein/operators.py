import os
import sys

import os.path
import scipy
import scipy.sparse.linalg as sla
import numpy as np
import potpourri3d as pp3d
import scipy.spatial
import torch

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

import atomsurf.utils.diffusion_net_utils as diff_utils

"""
In this file, we define functions to make the following transformations :
.ply -> DiffNets operators in .npz format
"""


def normalize(x, divide_eps=1e-6):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if len(x.shape) == 1:
        raise ValueError(
            "called normalize() on single vector of dim "
            + str(x.shape)
            + " are you sure?"
        )
    return x / (torch.norm(x, dim=len(x.shape) - 1) + divide_eps).unsqueeze(-1)


def cross(vec_A, vec_B):
    return torch.cross(vec_A, vec_B, dim=-1)


def dot(vec_A, vec_B):
    return torch.sum(vec_A * vec_B, dim=-1)


# Given (..., 3) vectors and normals, projects out any components of vecs
# which lies in the direction of normals. Normals are assumed to be unit.
def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def face_area(verts, faces):
    coords = verts[faces.long()]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)
    return 0.5 * torch.norm(raw_normal, dim=len(raw_normal.shape) - 1)


def face_normals(verts, faces, normalized=True):
    coords = verts[faces.long()]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal


# NP
def neighborhood_normal(points):
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    # numpy in, numpy out
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


# NP
def mesh_vertex_normals(verts, faces):
    # numpy in / out
    face_n = diff_utils.toNP(
        face_normals(torch.tensor(verts), torch.tensor(faces))
    )  # ugly torch <---> numpy

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(
        vertex_normals, axis=-1, keepdims=True
    )

    return vertex_normals


# torch -> NP -> torch
def vertex_normals(verts, faces):
    verts_np = diff_utils.toNP(verts)
    normals = mesh_vertex_normals(verts_np, diff_utils.toNP(faces))

    # if any are NaN, wiggle slightly and recompute
    bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
    if bad_normals_mask.any():
        bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
        scale = np.linalg.norm(bbox) * 1e-4
        wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
        wiggle_verts = verts_np + bad_normals_mask * wiggle
        normals = mesh_vertex_normals(wiggle_verts, diff_utils.toNP(faces))

    # if still NaN assign random normals (probably means unreferenced verts in mesh)
    bad_normals_mask = np.isnan(normals).any(axis=1)
    if bad_normals_mask.any():
        normals[bad_normals_mask, :] = (
                                               np.random.RandomState(seed=777).rand(*verts.shape) - 0.5
                                       )[bad_normals_mask, :]
        normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    normals = torch.from_numpy(normals).to(device=verts.device, dtype=verts.dtype)

    if torch.any(torch.isnan(normals)):
        raise ValueError("NaN normals :(")
    return normals


#  could be both
def build_tangent_frames(verts, faces, normals=None):
    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals is None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

    # = find an orthogonal basis

    basis_cand1 = torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    basis_cand2 = torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)

    basisX = torch.where(
        (torch.abs(dot(vert_normals, basis_cand1)) < 0.9).unsqueeze(-1),
        basis_cand1,
        basis_cand2,
    )
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = cross(vert_normals, basisX)
    frames = torch.stack((basisX, basisY, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames


def edge_tangent_vectors(verts, frames, edges):
    edges = edges.long()
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient.
    All values pointwise.
    - edges: (2, E)
    """

    edges_np = diff_utils.toNP(edges)

    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges_np.shape[1]):
        tail_ind = edges_np[0, iE]
        tip_ind = edges_np[1, iE]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(iE)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for iV in range(N):
        n_neigh = len(vert_edge_outgoing[iV])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [iV]
        for i_neigh in range(n_neigh):
            iE = vert_edge_outgoing[iV][i_neigh]
            jV = edges_np[1, iE]
            ind_lookup.append(jV)

            edge_vec = edge_tangent_vectors[iE][:]
            w_e = 1.0

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(iV)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(N, N)
    ).tocsc()

    return mat


def compute_operators(verts, faces, k_eig=128, normals=None):
    """
    Builds spectral operators for a mesh/point cloud. Constructs mass matrix, eigenvalues/vectors for Laplacian, and gradient matrix.
    See get_operators() for a similar routine that wraps this one with a layer of caching.
    Torch in / torch out.
    Arguments:
      - vertices: (V,3) vertex positions
      - faces: (F,3) list of triangular faces. If empty, assumed to be a point cloud.
      - k_eig: number of eigenvectors to use
    Returns:
      - frames: (V,3,3) X/Y/Z coordinate frame at each vertex. Z coordinate is normal (e.g. [:,2,:] for normals)
      - massvec: (V) real diagonal of lumped mass matrix
      - L: (VxV) real sparse matrix of (weak) Laplacian
      - evals: (k) list of eigenvalues of the Laplacian
      - evecs: (V,k) list of eigenvectors of the Laplacian
      - gradX: (VxV) sparse matrix which gives X-component of gradient in the local basis at the vertex
      - gradY: same as gradX but for Y-component of gradient
    PyTorch doesn't seem to like complex sparse matrices, so we store the "real" and "imaginary" (aka X and Y) gradient matrices separately,
    rather than as one complex sparse matrix.
    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only othrthonormal with respect to the mass matrix,
    like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """

    verts = torch.from_numpy(np.ascontiguousarray(verts))
    faces = torch.from_numpy(np.ascontiguousarray(faces))
    device = verts.device
    dtype = verts.dtype
    eps = 1e-8

    verts_np = diff_utils.toNP(verts).astype(np.float64)
    faces_np = diff_utils.toNP(faces)
    frames = build_tangent_frames(verts, faces, normals=normals)

    # Build the scalar Laplacian
    L = pp3d.cotan_laplacian(verts_np, faces_np, denom_eps=1e-10)
    massvec_np = pp3d.vertex_areas(verts_np, faces_np)
    massvec_np += eps * np.mean(massvec_np)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec_np).any():
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # === Compute the eigenbasis
    if k_eig > 0:

        # Prepare matrices
        L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
        massvec_eigsh = massvec_np
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise,
                # but for some reason those parameters seem to have no effect
                evals_np, evecs_np = sla.eigsh(
                    L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma
                )

                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals_np = np.clip(evals_np, a_min=0.0, a_max=float("inf"))

                break
            except Exception as e:
                print(e)
                if failcount > 3:
                    raise ValueError("failed to compute eigendecomp")
                failcount += 1
                print("--- decomp failed; adding eps ===> count: " + str(failcount))
                L_eigsh = L_eigsh + scipy.sparse.identity(L.shape[0]) * (
                        eps * 10 ** failcount
                )

    else:  # k_eig == 0
        evals_np = np.zeros((0))
        evecs_np = np.zeros((verts.shape[0], 0))

    # == Build gradient matrices

    # For meshes, we use the same edges as were used to build the Laplacian.
    edges = torch.tensor(
        np.stack((inds_row, inds_col), axis=0), device=device, dtype=faces.dtype
    )
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    grad_mat_np = build_grad(verts, edges, edge_vecs)

    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # === Convert back to torch
    massvec = torch.from_numpy(massvec_np).to(device=device, dtype=dtype)
    L = diff_utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
    evals = torch.from_numpy(evals_np).to(device=device, dtype=dtype)
    evecs = torch.from_numpy(evecs_np).to(device=device, dtype=dtype)
    gradX = diff_utils.sparse_np_to_torch(gradX_np).to(device=device, dtype=dtype)
    gradY = diff_utils.sparse_np_to_torch(gradY_np).to(device=device, dtype=dtype)
    return frames, massvec, L, evals, evecs, gradX, gradY


def get_operators(npz_path, verts, faces, k_eig=128, normals=None, recompute=False):
    """
    We remove the hashing util and add a filename for the npz instead.
    """
    frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts, faces, k_eig, normals=normals)
    dtype_np = np.float32
    L_np = diff_utils.sparse_torch_to_np(L).astype(dtype_np)
    gradX_np = diff_utils.sparse_torch_to_np(gradX).astype(dtype_np)
    gradY_np = diff_utils.sparse_torch_to_np(gradY).astype(dtype_np)
    np.savez(npz_path,
             verts=verts.astype(dtype_np),
             frames=diff_utils.toNP(frames).astype(dtype_np),
             faces=faces,
             k_eig=k_eig,
             mass=diff_utils.toNP(mass).astype(dtype_np),
             L_data=L_np.data.astype(dtype_np),
             L_indices=L_np.indices,
             L_indptr=L_np.indptr,
             L_shape=L_np.shape,
             evals=diff_utils.toNP(evals).astype(dtype_np),
             evecs=diff_utils.toNP(evecs).astype(dtype_np),
             gradX_data=gradX_np.data.astype(dtype_np),
             gradX_indices=gradX_np.indices,
             gradX_indptr=gradX_np.indptr,
             gradX_shape=gradX_np.shape,
             gradY_data=gradY_np.data.astype(dtype_np),
             gradY_indices=gradY_np.indices,
             gradY_indptr=gradY_np.indptr,
             gradY_shape=gradY_np.shape,
             )
    return frames, mass, L, evals, evecs, gradX, gradY


def load_operators(npz_path):
    """
    We remove the hashing util and add a filename for the npz instead.
    """

    def read_sp_mat(prefix):
        data = npzfile[prefix + "_data"]
        indices = npzfile[prefix + "_indices"]
        indptr = npzfile[prefix + "_indptr"]
        shape = npzfile[prefix + "_shape"]
        mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
        return mat

    npzfile = np.load(npz_path, allow_pickle=True)
    frames = npzfile["frames"]
    mass = npzfile["mass"]
    L = read_sp_mat("L")
    evals = npzfile["evals"]
    evecs = npzfile["evecs"]
    gradX = read_sp_mat("gradX")
    gradY = read_sp_mat("gradY")

    # frames = torch.from_numpy(frames).to(device=device, dtype=dtype)
    # mass = torch.from_numpy(mass).to(device=device, dtype=dtype)
    # L = diff_utils.sparse_np_to_torch(L).to(device=device, dtype=dtype)
    # evals = torch.from_numpy(evals).to(device=device, dtype=dtype)
    # evecs = torch.from_numpy(evecs).to(device=device, dtype=dtype)
    # gradX = diff_utils.sparse_np_to_torch(gradX).to(device=device, dtype=dtype)
    # gradY = diff_utils.sparse_np_to_torch(gradY).to(device=device, dtype=dtype)
    return frames, mass, L, evals, evecs, gradX, gradY


if __name__ == "__main__":
    import open3d as o3d

    ply_file = "../../data/example_files/example_mesh.ply"
    mesh = o3d.io.read_triangle_mesh(filename=ply_file)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    operator_file = "../../data/example_files/example_operator.npz"
    operators = compute_operators(vertices, faces, k_eig=128)
