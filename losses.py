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


def loss_mean_variance(w_star, r_next, sigma_sample, lam: float = 1.0):
    """
    L2: Mean-Variance (Markowitz) loss.

    L2 = w*^T Sigma_sample w* - lambda * w*^T r

    lambda controls the risk-return trade-off:
      lambda=0  → pure variance (= L1)
      lambda>0  → penalize variance, reward return

    Advantages over negative Sharpe:
      - No division → stable gradients (no exploding grad from small vol)
      - Linear in r → less prone to overfitting on return direction
      - lambda is an explicit, interpretable hyperparameter
    """
    var_p = w_star @ sigma_sample @ w_star
    ret   = w_star @ r_next
    return var_p - lam * ret
