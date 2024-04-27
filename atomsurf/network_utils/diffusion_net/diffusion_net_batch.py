import torch
import torch.nn as nn
from .layers import LearnedTimeDiffusion, MiniMLP, SpatialGradientFeatures


class DiffusionNetBlockBatch(nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(self, C_width, mlp_hidden_dims, dropout=0.5, use_bn=True, init_time=2.0, init_std=2.0,
                 diffusion_method="spectral", with_gradient_features=True, with_gradient_rotations=True):
        super().__init__()

        # Specified dimensions
        self.C_width = C_width
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.C_width, method=diffusion_method, init_time=init_time, init_std=init_std)

        self.MLP_C = 2 * self.C_width

        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(
                self.C_width, with_gradient_rotations=self.with_gradient_rotations
            )
            self.MLP_C += self.C_width

        # MLPs
        self.mlp = MiniMLP([self.MLP_C] + list(self.mlp_hidden_dims) + [self.C_width],
                           dropout=self.dropout,
                           use_bn=use_bn)

        # todo: is this needed?
        # self.bn = nn.BatchNorm1d(C_width)

    # def forward(self, x_in, mass, L, evals, evecs, gradX, gradY):
    def forward(self, surfaces):
        x_in = [mini_surface.x for mini_surface in surfaces]
        mass = [mini_surface.mass for mini_surface in surfaces]
        L = [mini_surface.L for mini_surface in surfaces]
        evals = [mini_surface.evals for mini_surface in surfaces]
        evecs = [mini_surface.evecs for mini_surface in surfaces]
        gradX = [mini_surface.gradX for mini_surface in surfaces]
        gradY= [mini_surface.gradY for mini_surface in surfaces]

        x_in_batch = torch.cat(x_in, dim=0)
        split_sizes = [tensor.size(0) for tensor in x_in]

        # Manage dimensions
        B = len(x_in)
        assert x_in[0].shape[-1] == self.C_width, f"x_in has wrong last dimension {x_in[0]} != {self.C_width}"

        # Diffusion block
        x_diffuse = [self.diffusion(x.unsqueeze(0), L_, mass_, evals_, evecs_)[0]
                     for x, L_, mass_, evals_, evecs_ in zip(x_in, L, mass, evals, evecs)]

        x_diffuse_batch = torch.cat(x_diffuse, dim=0)

        # Compute gradient features, if using
        if self.with_gradient_features:

            # Compute gradients
            x_grads = []
            for b in range(B):
                # gradient after diffusion
                x_gradX = torch.mm(gradX[b], x_diffuse[b])
                x_gradY = torch.mm(gradY[b], x_diffuse[b])
                # x_gradX = torch.mm(gradX[b].to_torch_sparse_coo_tensor(), x_diffuse[b])
                # x_gradY = torch.mm(gradY[b].to_torch_sparse_coo_tensor(), x_diffuse[b])
                x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))

            x_grad_batch = torch.cat(x_grads, dim=0)

            # Evaluate gradient features
            x_grad_features_batch = self.gradient_features(x_grad_batch)

            # Stack inputs to mlp
            feature_combined = torch.cat((x_in_batch, x_diffuse_batch, x_grad_features_batch), dim=-1)
        else:
            # Stack inputs to mlp
            feature_combined = torch.cat((x_in_batch, x_diffuse_batch), dim=-1)

        # Apply the mlp
        x0_out_batch = self.mlp(feature_combined)

        # Skip connection
        x0_out_batch = x0_out_batch + x_in_batch

        # # apply batch norm # todo: is this needed?
        # x0_out_batch = self.bn(x0_out_batch)

        # Split batch back into list
        x0_out = torch.split(x0_out_batch, split_sizes, dim=0)

        return x0_out


class DiffusionNetBatch(nn.Module):
    def __init__(self, C_in, C_out, C_width=128, N_block=4, last_activation=None, mlp_hidden_dims=None, dropout=0.5,
                 with_gradient_features=True, with_gradient_rotations=True, use_bn=True, init_time=2.0, init_std=2.0):

        super().__init__()

        # # Store parameters

        # Basic parameters
        self.C_in = C_in
        self.C_out = C_out
        self.C_width = C_width
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation

        # MLP options
        if mlp_hidden_dims is None:
            mlp_hidden_dims = [C_width, C_width]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # Gradient features
        self.with_gradient_features = with_gradient_features
        self.with_gradient_rotations = with_gradient_rotations

        # # Set up the network

        # First and last affine layers
        self.first_lin = nn.Linear(C_in, C_width)
        self.last_lin = nn.Linear(C_width, C_out)

        # DiffusionNet blocks
        self.blocks = []
        for i_block in range(self.N_block):
            block = DiffusionNetBlockBatch(
                C_width=C_width,
                mlp_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                with_gradient_features=with_gradient_features,
                with_gradient_rotations=with_gradient_rotations,
                use_bn=use_bn,
                init_time=init_time,
                init_std=init_std
            )

            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(self, surface, graph=None):
        """
        all the inputs are in list format
        A forward pass on the DiffusionNet.
        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        Generally, our data layout it is [N,C] or [B,N,C].
        Call get_operators() to generate geometric quantities mass/L/evals/evecs/gradX/gradY. Note that depending on the options for the DiffusionNet,
        not all are strictly necessary.
        Parameters:
            x_in (tensor):      Input features, dimension [N,C] or [B,N,C]
            mass (tensor):      Mass vector, dimension [N] or [B,N]
            L (tensor):         Laplace matrix, sparse tensor with dimension [N,N] or [B,N,N]
            evals (tensor):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
            evecs (tensor):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
            gradX (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
            gradY (tensor):     Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """
        x_in, mass, L, evals, evecs, gradX, gradY = surface.x, surface.mass, surface.L, surface.evals, surface.evecs, surface.gradX, surface.gradY

        assert isinstance(x_in, list), "inputs to `DiffusionNetBatch` must be a lists"
        assert x_in[0].shape[-1] == self.C_in, f"DiffusionNet was constructed with C_in={self.C_in}, but x_in has last dim={x_in[0].shape[-1]}"

        mass = [m.unsqueeze(0) for m in mass]
        L = [ll.unsqueeze(0) for ll in L]
        evals = [e.unsqueeze(0) for e in evals]
        evecs = [e.unsqueeze(0) for e in evecs]

        # Apply the first linear layer
        x = [self.first_lin(y) for y in x_in]

        # Apply each of the blocks
        for b in self.blocks:
            x = b(x, mass, L, evals, evecs, gradX, gradY)

        # Apply the last linear layer
        x_out = [self.last_lin(y) for y in x]

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = [self.last_activation(y) for y in x_out]

        surface.y = x_out

        return surface
