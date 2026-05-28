import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import dense_to_sparse
from scipy import stats


def compute_node_features(window: np.ndarray) -> np.ndarray:
    """
    window  : (T, N) return matrix for the lookback period  [variable T ok]
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

    sample_indices : list of (lb_start, t, h_end) integer position tuples
      window = returns_arr[lb_start : t]    → x, edge_index, edge_weight, sigma_sample
               (variable length ≈ lookback_years 거래일)
      r_next = prod(1 + returns_arr[t : h_end], axis=0) - 1
               (variable length ≈ horizon_months 거래일)

    캘린더 기반으로 생성된 인덱스를 받으므로 lookback/horizon 길이가
    샘플마다 수 거래일 내에서 달라질 수 있음 (완전히 허용).
    """
    def __init__(self, returns_arr: np.ndarray, sample_indices: list,
                 corr_threshold: float = 0.3):
        self.samples = []
        for (lb_start, t, h_end) in sample_indices:
            window       = returns_arr[lb_start : t]
            r_next       = np.prod(1 + returns_arr[t : h_end], axis=0) - 1

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
