import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class GNNRiskEncoder(nn.Module):
    """
    2-layer GCN. Message passing encodes systemic risk propagation.
    Output H in R^{N x d_latent}: latent factor matrix.
    """
    def __init__(self, d_in: int, d_hidden: int, d_latent: int):
        super().__init__()
        self.conv1 = GCNConv(d_in,     d_hidden)
        self.conv2 = GCNConv(d_hidden, d_latent)
        self.bn1   = nn.BatchNorm1d(d_hidden)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = self.bn1(F.relu(h))
        h = self.conv2(h, edge_index, edge_weight)
        h = F.normalize(h, p=2, dim=1)  # ClusterNet: project onto unit sphere
        return h   # (N, d_latent)


class DifferentiableGMV(nn.Module):
    """
    Differentiable long-only GMV layer via cvxpylayers.

    Takes H (N x d_latent) directly as parameter instead of Sigma,
    exploiting the factored structure:

      w^T Sigma w = w^T (H H^T + eps I) w = ||H^T w||^2 + eps ||w||^2

    This sum-of-squares form is DPP-compliant, avoiding the DPP issue
    of quad_form with a symmetric matrix parameter.

    Backprop: dL/dw* -> dL/dH via KKT implicit differentiation.
    """
    def __init__(self, n_assets: int, d_latent: int, epsilon: float = 1e-4):
        super().__init__()
        self.n   = n_assets
        w        = cp.Variable(n_assets)
        H_cp     = cp.Parameter((n_assets, d_latent))
        problem  = cp.Problem(
            cp.Minimize(
                0.5 * cp.sum_squares(H_cp.T @ w) + (epsilon / 2) * cp.sum_squares(w)
            ),
            [cp.sum(w) == 1, w >= 0]
        )
        self.layer = CvxpyLayer(problem, parameters=[H_cp], variables=[w])

    def forward(self, H):
        w_star, = self.layer(H)
        return w_star.float()   # cvxpylayers returns float64; cast back to float32


class GraphGMVModel(nn.Module):
    """
    End-to-end pipeline:
      X_t, A_t  ->  GNN  ->  H_t  ->  w*

    H_t flows directly into the GMV layer (no explicit Sigma_t construction).
    Sigma_t = H_t H_t^T + eps I is implicit inside the optimization objective.
    """
    def __init__(self, n_assets: int, d_in: int = 7,
                 d_hidden: int = 32, d_latent: int = 16, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.gnn     = GNNRiskEncoder(d_in, d_hidden, d_latent)
        self.gmv     = DifferentiableGMV(n_assets, d_latent, epsilon)

    def forward(self, x, edge_index, edge_weight=None):
        H      = self.gnn(x, edge_index, edge_weight)   # (N, d_latent)
        w_star = self.gmv(H)                             # (N,)
        return w_star, H

    def compute_sigma(self, H):
        """Recover explicit Sigma_t = H H^T + eps I for analysis."""
        N = H.shape[0]
        return H @ H.T + self.epsilon * torch.eye(N, device=H.device)
