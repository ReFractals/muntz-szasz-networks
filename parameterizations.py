"""
Exponent Parameterizations for Müntz-Szász Networks
====================================================

This module provides numerically stable parameterizations for learning
exponents in MSN. Direct optimization of exponents in R_{>0} is unstable
because gradients can push exponents negative (undefined) or extremely 
large (overflow for |x| > 1).

Two parameterizations are provided:
1. Bounded: Maps raw parameters to (eps, p_max - eps) via sigmoid + sort
2. Cumsum: Uses softplus + cumsum to ensure ordering and positivity
"""

import torch
import torch.nn.functional as F


def make_ordered_bounded(raw: torch.Tensor, p_max: float = 4.0, eps: float = 1e-3) -> torch.Tensor:
    """
    Bounded exponent parameterization.
    
    Maps raw parameters to strictly positive, ordered exponents in (eps, p_max - eps).
    
    Parameters
    ----------
    raw : torch.Tensor
        Raw learnable parameters of shape (K,)
    p_max : float
        Maximum exponent value (default: 4.0)
    eps : float
        Margin from 0 and p_max (default: 1e-3)
    
    Returns
    -------
    torch.Tensor
        Ordered exponents in (eps, p_max - eps)
    
    Example
    -------
    >>> raw = torch.randn(6)
    >>> mu = make_ordered_bounded(raw, p_max=4.0)
    >>> assert (mu > 0).all() and (mu < 4.0).all()
    >>> assert (mu[1:] >= mu[:-1]).all()  # Ordered
    """
    u = torch.sigmoid(raw)
    u_sorted, _ = torch.sort(u)
    return eps + (p_max - 2 * eps) * u_sorted


def make_cumsum_softplus(raw: torch.Tensor, min_val: float = 0.05) -> torch.Tensor:
    """
    Cumsum-softplus exponent parameterization.
    
    Alternative parameterization using cumulative sum of softplus outputs.
    This naturally produces positive, increasing exponents without explicit
    sorting, which can be more stable in some optimization scenarios.
    
    Parameters
    ----------
    raw : torch.Tensor
        Raw learnable parameters of shape (K,)
    min_val : float
        Minimum increment between consecutive exponents (default: 0.05)
    
    Returns
    -------
    torch.Tensor
        Ordered, strictly positive exponents
    
    Example
    -------
    >>> raw = torch.randn(6)
    >>> mu = make_cumsum_softplus(raw, min_val=0.05)
    >>> assert (mu > 0).all()
    >>> assert (mu[1:] > mu[:-1]).all()  # Strictly increasing
    """
    increments = F.softplus(raw) + min_val
    return torch.cumsum(increments, dim=0)


def safe_pow_abs(x: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Numerically stable computation of |x|^p for potentially fractional p.
    
    Uses the identity |x|^p = exp(p * log(|x|)) with a small epsilon
    added to prevent log(0).
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (...,)
    p : torch.Tensor
        Exponent tensor of shape (K,)
    eps : float
        Small constant for numerical stability (default: 1e-12)
    
    Returns
    -------
    torch.Tensor
        Result of shape (..., K) where result[..., k] = |x|^{p[k]}
    
    Example
    -------
    >>> x = torch.linspace(0, 1, 100)
    >>> p = torch.tensor([0.5, 1.0, 2.0])
    >>> result = safe_pow_abs(x, p)  # shape: (100, 3)
    """
    ax = torch.abs(x) + eps
    # Broadcast: x[..., None] has shape (..., 1), p has shape (K,)
    # log(ax)[..., None] * p broadcasts to (..., K)
    return torch.exp(torch.log(ax)[..., None] * p)
