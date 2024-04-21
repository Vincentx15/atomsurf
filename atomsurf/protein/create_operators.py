import os
import sys

import os.path
import scipy
import scipy.sparse.linalg as sla
import numpy as np
import potpourri3d as pp3d
import scipy.spatial
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix, diags

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..', '..'))

import atomsurf.utils.helpers as diff_utils

"""
In this file, we define functions to make the following transformations :
.ply -> DiffNets operators in .npz format
"""


class TriMesh(object):
    def __init__(self, verts, faces):
        self.verts = verts
        self.faces = faces
        self.stiffness = None
        self.eigen_vals = None
        self.eigen_vecs = None
        self.mass = None

    def LB_decomposition(self, k=None):
        # stiffness matrix
        self.stiffness = self.compute_stiffness_matrix()
        # mass matrix
        self.mass = self.compute_fem_mass_matrix()
        # compute Laplace-Beltrami basis (eigen-vectors are stored column-wise)
        self.eigen_vals, self.eigen_vecs = eigsh(A=self.stiffness, k=k, M=self.mass, sigma=-0.01)
        self.eigen_vals[0] = 0

    def compute_stiffness_matrix(self):
        verts = self.verts
        faces = self.faces
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]

        e1 = v3 - v2
        e2 = v1 - v3
        e3 = v2 - v1

        # compute cosine alpha/beta
        L1 = np.linalg.norm(e1, axis=1)
        L2 = np.linalg.norm(e2, axis=1)
        L3 = np.linalg.norm(e3, axis=1)
        cos1 = np.einsum('ij,ij->i', -e2, e3) / (L2 * L3)
        cos2 = np.einsum('ij,ij->i', e1, -e3) / (L1 * L3)
        cos3 = np.einsum('ij,ij->i', -e1, e2) / (L1 * L2)

        # cot(arccos(x)) = x/sqrt(1-x^2)
        I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        S = np.concatenate([cos3, cos1, cos2])
        S = 0.5 * S / np.sqrt(1 - S ** 2)

        In = np.concatenate([I, J, I, J])
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([-S, -S, S, S])

        N = verts.shape[0]
        stiffness = coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
        return stiffness

    def compute_fem_mass_matrix(self):
        verts = self.verts
        faces = self.faces
        # compute face areas
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

        I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        S = np.concatenate([face_areas, face_areas, face_areas])

        In = np.concatenate([I, J, I])
        Jn = np.concatenate([J, I, I])
        Sn = 1. / 12. * np.concatenate([S, S, 2 * S])

        N = verts.shape[0]
        mass = coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()

        return mass


def hmr_decomp(verts, faces, max_eigen_val=5, k=128):
    trimesh = TriMesh(verts=verts, faces=faces.astype(int))

    # HMR uses a growing and variable number of eigen vecs. This only makes sense for small surfaces such as pockets.
    # In our case, even after 600 evecs, the eval is 1.06 << 5=hmr_cutoff
    # num_verts = len(verts)
    # num_eigs = int(0.16 * num_verts)
    # max_val = 0
    # while max_val < max_eigen_val:
    #     num_eigs += 5
    #     print(num_eigs, max_val)
    #     trimesh.LB_decomposition(k=num_eigs)  # scipy eigsh must have k < N
    #     max_val = np.max(trimesh.eigen_vals)
    # cutoff = np.argmax(trimesh.eigen_vals > max_eigen_val)
    # eigen_vals = trimesh.eigen_vals[:cutoff]
    # eigen_vecs = trimesh.eigen_vecs[:, :cutoff]

    trimesh.LB_decomposition(k=k)  # scipy eigsh must have k < N
    eigen_vals = trimesh.eigen_vals
    eigen_vecs = trimesh.eigen_vecs

    # save features
    eigen_vals = eigen_vals.astype(np.float32)
    eigen_vecs = eigen_vecs.astype(np.float32)
    mass = trimesh.mass.astype(np.float32)
    return eigen_vals, eigen_vecs, mass


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
    return x / (np.linalg.norm(x, axis=len(x.shape) - 1) + divide_eps)[..., None]


def dot(vec_A, vec_B):
    return np.sum(vec_A * vec_B, axis=-1)


def project_to_tangent(vecs, unit_normals):
    # Given (..., 3) vectors and normals, projects out any components of vecs
    # which lies in the direction of normals. Normals are assumed to be unit.
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots[..., None]


def face_normals(verts, faces, normalized=True):
    coords = verts[faces]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    raw_normal = np.cross(vec_A, vec_B)
    if normalized:
        return normalize(raw_normal)
    return raw_normal


def mesh_vertex_normals(verts, faces):
    face_n = face_normals(verts, faces)
    vertex_normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(
        vertex_normals, axis=-1, keepdims=True
    )

    return vertex_normals


def vertex_normals(verts, faces):
    normals = mesh_vertex_normals(verts, faces)

    # if any are NaN, wiggle slightly and recompute
    bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
    if bad_normals_mask.any():
        bbox = np.amax(verts, axis=0) - np.amin(verts, axis=0)
        scale = np.linalg.norm(bbox) * 1e-4
        wiggle = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5) * scale
        wiggle_verts = verts + bad_normals_mask * wiggle
        normals = mesh_vertex_normals(wiggle_verts, faces)

    # if still NaN assign random normals (probably means unreferenced verts in mesh)
    bad_normals_mask = np.isnan(normals).any(axis=1)
    if bad_normals_mask.any():
        normals[bad_normals_mask, :] = (np.random.RandomState(seed=777).rand(*verts.shape) - 0.5)[bad_normals_mask, :]
        normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]
    if np.any(np.isnan(normals)):
        raise ValueError("NaN normals :(")
    return normals


def build_tangent_frames(verts, faces, normals=None):
    """
    Define a local frame based on thee normal to get an approx of local manifold
    :param verts:
    :param faces:
    :param normals:
    :return:
    """
    V = verts.shape[0]
    dtype = verts.dtype

    if normals is None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

    # = find an orthogonal basis
    basis_cand1 = np.tile(np.array([1, 0, 0]).astype(dtype=dtype)[None, :], (V, 1))
    basis_cand2 = np.tile(np.array([0, 1, 0]).astype(dtype=dtype)[None, :], (V, 1))
    basis_1_far = np.abs(dot(vert_normals, basis_cand1)) < 0.9
    basisX = np.where(basis_1_far[:, None], basis_cand1, basis_cand2)
    basisX = project_to_tangent(basisX, vert_normals)
    basisX = normalize(basisX)
    basisY = np.cross(vert_normals, basisX)
    frames = np.stack((basisX, basisY, vert_normals), axis=-2)
    if np.any(np.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")
    return frames


def edge_tangent_vectors(verts, frames, edges):
    """
    Get tangent vector of edges in each local frame
    :param verts:
    :param frames:
    :param edges:
    :return:
    """
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basisX = frames[edges[0, :], 0, :]
    basisY = frames[edges[0, :], 1, :]
    compX = dot(edge_vecs, basisX)
    compY = dot(edge_vecs, basisY)
    edge_tangent = np.stack((compX, compY), axis=-1)
    return edge_tangent


def build_grad(verts, edges, edge_tangent_vectors):
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at vertices, produces a complex (vector value) at vertices giving the gradient.
    All values pointwise.
    - edges: (2, E)
    """
    # TODO find a way to do this in pure numpy?

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for iE in range(edges.shape[1]):
        tail_ind = edges[0, iE]
        tip_ind = edges[1, iE]
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
            jV = edges[1, iE]
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


def compute_operators(verts, faces, k_eig=128, normals=None, use_hmr_decomp=False):
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
    Note: for a generalized eigenvalue problem, the mass matrix matters! The eigenvectors are only orthonormal with respect to the mass matrix,
    like v^H M v, so the mass (given as the diagonal vector massvec) needs to be used in projections, etc.
    """
    dtype = verts.dtype
    eps = 1e-8
    # Build the scalar Laplacian
    L = pp3d.cotan_laplacian(verts, faces, denom_eps=1e-10)
    massvec = pp3d.vertex_areas(verts, faces)
    massvec += eps * np.mean(massvec)

    if np.isnan(L.data).any():
        raise RuntimeError("NaN Laplace matrix")
    if np.isnan(massvec).any():
        raise RuntimeError("NaN mass matrix")

    # Read off neighbors & rotations from the Laplacian
    L_coo = L.tocoo()
    inds_row = L_coo.row
    inds_col = L_coo.col

    # === Compute the eigenbasis
    if not use_hmr_decomp:
        # Prepare matrices
        L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * eps).tocsc()
        massvec_eigsh = massvec
        Mmat = scipy.sparse.diags(massvec_eigsh)
        eigs_sigma = eps

        failcount = 0
        while True:
            try:
                # We would be happy here to lower tol or maxiter since we don't need these to be super precise,
                # but for some reason those parameters seem to have no effect
                evals, evecs = sla.eigsh(
                    L_eigsh, k=k_eig, M=Mmat, sigma=eigs_sigma
                )

                # Clip off any eigenvalues that end up slightly negative due to numerical weirdness
                evals = np.clip(evals, a_min=0.0, a_max=float("inf"))

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
    else:
        evals, evecs, Mmat = hmr_decomp(verts=verts, faces=faces)

    # == Build gradient matrices
    # For meshes, we use the same edges as were used to build the Laplacian.
    frames = build_tangent_frames(verts, faces, normals=normals)
    edges = np.stack((inds_row, inds_col), axis=0)
    edge_vecs = edge_tangent_vectors(verts, frames, edges)
    grad_mat = build_grad(verts, edges, edge_vecs)

    # Split complex gradient in to two real sparse mats (torch doesn't like complex sparse matrices)
    gradX = np.real(grad_mat)
    gradY = np.imag(grad_mat)
    return frames, massvec, L, evals, evecs, gradX, gradY


def get_operators(npz_path, verts, faces, k_eig=128, normals=None, recompute=False, use_hmr_decomp=False):
    """
    We remove the hashing util and add a filename for the npz instead.
    """
    if not os.path.exists(npz_path) or recompute:
        frames, mass, L, evals, evecs, gradX, gradY = compute_operators(verts,
                                                                        faces,
                                                                        k_eig,
                                                                        normals=normals,
                                                                        use_hmr_decomp=use_hmr_decomp)
        np.savez(npz_path,
                 verts=verts,
                 faces=faces,
                 k_eig=k_eig,
                 mass=mass,
                 L_data=L.data,
                 L_indices=L.indices,
                 L_indptr=L.indptr,
                 L_shape=L.shape,
                 evals=evals,
                 evecs=evecs,
                 gradX_data=gradX.data,
                 gradX_indices=gradX.indices,
                 gradX_indptr=gradX.indptr,
                 gradX_shape=gradX.shape,
                 gradY_data=gradY.data,
                 gradY_indices=gradY.indices,
                 gradY_indptr=gradY.indptr,
                 gradY_shape=gradY.shape,
                 )


def load_operators(npzfile):
    """
    We remove the hashing util and add a filename for the npz instead.
    """
    if not isinstance(npzfile, np.lib.npyio.NpzFile):
        npzfile = np.load(npzfile, allow_pickle=True)
    mass = npzfile["mass"]
    L = diff_utils.read_sp_mat(npzfile, "L")
    evals = npzfile["evals"]
    evecs = npzfile["evecs"]
    gradX = diff_utils.read_sp_mat(npzfile, "gradX")
    gradY = diff_utils.read_sp_mat(npzfile, "gradY")
    return mass, L, evals, evecs, gradX, gradY


if __name__ == "__main__":
    import open3d as o3d

    ply_file = "../../data/example_files/example_mesh.ply"
    mesh = o3d.io.read_triangle_mesh(filename=ply_file)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    operator_file = "../../data/example_files/example_operator.npz"
    get_operators(operator_file, vertices, faces, k_eig=128, recompute=True, use_hmr_decomp=False)
    # get_operators(operator_file, vertices, faces, k_eig=128, recompute=True, use_hmr_decomp=True)
    operators = load_operators(operator_file)
