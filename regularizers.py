"""
Regularizers for Müntz-Szász Networks
=====================================

The Müntz condition Σ 1/λ_k = ∞ ensures basis completeness in classical
approximation theory. For finite K exponents, we encourage configurations
that would satisfy this condition asymptotically through regularization.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def muntz_divergence(mu: torch.Tensor, lam: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the Müntz divergence sum.
    
    D(μ, λ) = Σ_{k=1}^{K_e} 1/μ_k + Σ_{k=1}^{K_o} 1/λ_k
    
    The Müntz-Szász theorem requires this sum to diverge (= ∞) for density.
    Larger values indicate exponent configurations closer to satisfying the
    Müntz condition.
    
    Parameters
    ----------
    mu : torch.Tensor
        Even exponents of shape (K_e,)
    lam : torch.Tensor
        Odd exponents of shape (K_o,)
    eps : float
        Small constant for numerical stability (default: 1e-12)
    
    Returns
    -------
    torch.Tensor
        Scalar divergence value
    
    Example
    -------
    >>> mu = torch.tensor([0.5, 1.0, 2.0])
    >>> lam = torch.tensor([0.3, 1.5])
    >>> D = muntz_divergence(mu, lam)
    >>> print(f"Divergence: {D.item():.2f}")  # Higher is better
    """
    return torch.sum(1.0 / (mu + eps)) + torch.sum(1.0 / (lam + eps))


def muntz_regularizer(
    mu: torch.Tensor,
    lam: torch.Tensor,
    C: float = 2.0,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Müntz divergence regularizer.
    
    R_Müntz(μ, λ) = ReLU(C - D(μ, λ))
    
    This regularizer is zero when the divergence exceeds threshold C,
    and positive otherwise, encouraging:
    1. Small exponents (to maximize D)
    2. Diverse exponents (spreading increases D vs concentration)
    
    Parameters
    ----------
    mu : torch.Tensor
        Even exponents of shape (K_e,)
    lam : torch.Tensor
        Odd exponents of shape (K_o,)
    C : float
        Threshold for divergence (default: 2.0)
    eps : float
        Numerical stability constant (default: 1e-12)
    
    Returns
    -------
    torch.Tensor
        Regularization penalty (0 if D >= C)
    
    Notes
    -----
    For K=6 exponents with C=2.0, the constraint D >= C requires an average
    exponent value below 3.0, which encourages at least one small exponent
    near zero while allowing others to capture higher-order behavior.
    
    Example
    -------
    >>> mu = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    >>> lam = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    >>> loss = muntz_regularizer(mu, lam, C=2.0)
    """
    D = muntz_divergence(mu, lam, eps=eps)
    return F.relu(C - D)


def l1_coefficient_regularizer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    L1 regularization on Müntz edge coefficients.
    
    Promotes sparsity in the coefficient vectors, encouraging the network
    to use fewer power terms.
    
    Parameters
    ----------
    a : torch.Tensor
        Even coefficients of shape (K_e,)
    b : torch.Tensor
        Odd coefficients of shape (K_o,)
    
    Returns
    -------
    torch.Tensor
        Sum of absolute values
    """
    return torch.sum(torch.abs(a)) + torch.sum(torch.abs(b))


def exponent_growth_penalty(
    mu: torch.Tensor,
    lam: torch.Tensor,
    max_target: float = 10.0
) -> torch.Tensor:
    """
    Penalty for exponents exceeding a maximum target.
    
    Helps prevent numerical instability from very large exponents.
    
    Parameters
    ----------
    mu : torch.Tensor
        Even exponents
    lam : torch.Tensor
        Odd exponents
    max_target : float
        Maximum desired exponent (default: 10.0)
    
    Returns
    -------
    torch.Tensor
        Quadratic penalty for excess
    """
    penalty_mu = torch.mean(F.relu(mu - max_target) ** 2)
    penalty_lam = torch.mean(F.relu(lam - max_target) ** 2)
    return penalty_mu + penalty_lam
