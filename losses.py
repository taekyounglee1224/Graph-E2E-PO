import torch


def loss_variance(w_star, r_next, sigma_sample):
    """
    L1: Realized portfolio variance under the sample covariance.

    L1 = w*^T Sigma_sample w*
    grad = 2 * Sigma_sample @ w*

    sigma_sample is fixed (not the model's Sigma_t), preventing the model
    from gaming its own evaluation via self-referential covariance manipulation.
    """
    return w_star @ sigma_sample @ w_star


def loss_sharpe(w_star, r_next, sigma_sample, eps: float = 1e-8):
    """
    L2: Negative realized Sharpe ratio.

    L2 = - (w*^T r) / sqrt(w*^T Sigma_sample w* + eps)

    Consistent with GMV: GMV minimizes the denominator (portfolio vol),
    so penalizing -Sharpe aligns the optimization direction while also
    rewarding return — without discarding risk awareness.

    grad = -r/sigma_p + (w*^T r / sigma_p^3) * Sigma_sample @ w*
    """
    ret   = w_star @ r_next
    var_p = w_star @ sigma_sample @ w_star
    vol_p = torch.sqrt(var_p + eps)
    return -ret / vol_p
