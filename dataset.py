import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
from scipy import stats


def compute_node_features(window: np.ndarray) -> np.ndarray:
    """
    window  : (T, N) return matrix for the lookback period
    returns : (N, 7) standardized feature matrix

    Features per asset:
      0: mean return
      1: volatility (std)
      2: skewness
      3: excess kurtosis
      4: in-sample Sharpe (mean / std)
      5: total momentum (cumulative return over window)
      6: max drawdown
    """
    T, n  = window.shape
    feats = np.zeros((n, 7))
    for i in range(n):
        r            = window[:, i]
        feats[i, 0]  = np.mean(r)
        feats[i, 1]  = np.std(r)
        feats[i, 2]  = stats.skew(r)
        feats[i, 3]  = stats.kurtosis(r)
        feats[i, 4]  = np.mean(r) / (np.std(r) + 1e-8)
        feats[i, 5]  = np.prod(1 + r) - 1
        wealth       = np.cumprod(1 + r)
        peak         = np.maximum.accumulate(wealth)
        feats[i, 6]  = ((wealth - peak) / peak).min()

    mu  = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0,  keepdims=True) + 1e-8
    return (feats - mu) / std


def compute_adjacency(window: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    (N, N) thresholded adjacency: |Pearson correlation| >= threshold, no self-loops.
    Edges below threshold are zeroed out.
    Isolated nodes (all edges removed) get their single strongest edge restored.
    """
    corr = np.corrcoef(window.T)
    adj  = np.abs(corr)
    np.fill_diagonal(adj, 0.0)
    adj  = np.where(adj >= threshold, adj, 0.0)
    for i in range(adj.shape[0]):
        if adj[i].sum() == 0:
            tmp    = np.abs(corr[i]).copy()
            tmp[i] = 0.0
            j      = int(np.argmax(tmp))
            adj[i, j] = adj[j, i] = float(np.abs(corr[i, j]))
    return adj


def compute_sample_cov(window: np.ndarray) -> np.ndarray:
    """(N, N) sample covariance — used as Sigma_realized in losses."""
    return np.cov(window.T)


class PortfolioDataset(Dataset):
    """
    Pre-computes all samples upfront for fast training.

    Each sample corresponds to a rebalancing date t:
      window [t - lookback : t]       ->  x, edge_index, edge_weight, sigma_sample
      returns[t : t + holding_period] ->  r_next (compounded holding-period return)

    holding_period matches rebalance_freq so the loss reflects the actual
    return earned over the full holding period, not just one day.
    """
    def __init__(self, returns_arr: np.ndarray, indices, lookback: int,
                 holding_period: int = 1, corr_threshold: float = 0.3):
        self.samples = []
        for t in indices:
            window       = returns_arr[t - lookback : t]
            # Compound returns over the holding period: (1+r1)(1+r2)...(1+rT) - 1
            r_next       = np.prod(1 + returns_arr[t : t + holding_period], axis=0) - 1

            x            = compute_node_features(window)
            adj          = compute_adjacency(window, threshold=corr_threshold)
            sigma_sample = compute_sample_cov(window)

            adj_t                   = torch.FloatTensor(adj)
            edge_index, edge_weight = dense_to_sparse(adj_t)

            self.samples.append({
                'x'           : torch.FloatTensor(x),
                'edge_index'  : edge_index,
                'edge_weight' : edge_weight,
                'sigma_sample': torch.FloatTensor(sigma_sample),
                'r_next'      : torch.FloatTensor(r_next),
            })

    def __len__(self):          return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
