"""
Utility Functions for Müntz-Szász Networks
==========================================
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


def count_params(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    
    Returns
    -------
    int
        Number of trainable parameters
    
    Example
    -------
    >>> from msn import MSN
    >>> model = MSN(dims=[1, 8, 8, 1])
    >>> print(f"Parameters: {count_params(model):,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def msn_exponent_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all exponent parameters from an MSN model.
    
    Exponent parameters have names containing 'raw_even' or 'raw_odd'.
    
    Parameters
    ----------
    model : nn.Module
        MSN model
    
    Returns
    -------
    list
        List of exponent parameters
    
    Example
    -------
    >>> model = MSN(dims=[1, 8, 1])
    >>> exp_params = msn_exponent_params(model)
    >>> print(f"Exponent params: {len(exp_params)}")
    """
    exp_params = []
    for name, p in model.named_parameters():
        if "raw_even" in name or "raw_odd" in name:
            exp_params.append(p)
    return exp_params


def msn_coeff_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all coefficient (non-exponent) parameters from an MSN model.
    
    Parameters
    ----------
    model : nn.Module
        MSN model
    
    Returns
    -------
    list
        List of coefficient parameters
    """
    exp_ids = set(id(p) for p in msn_exponent_params(model))
    return [p for p in model.parameters() if id(p) not in exp_ids]


def dump_exponents(model: nn.Module, layer_idx: int = 0, out_idx: int = 0, in_idx: int = 0) -> Dict[str, Any]:
    """
    Extract exponents and coefficients from a specific edge in an MSN.
    
    Parameters
    ----------
    model : nn.Module
        MSN model
    layer_idx : int
        Layer index (default: 0)
    out_idx : int
        Output dimension index (default: 0)
    in_idx : int
        Input dimension index (default: 0)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'mu': even exponents (list)
        - 'lam': odd exponents (list)
        - 'a': even coefficients (list)
        - 'b': odd coefficients (list)
    
    Example
    -------
    >>> model = MSN(dims=[1, 8, 1])
    >>> # ... train model ...
    >>> exp_dict = dump_exponents(model, layer_idx=0)
    >>> print(f"Learned μ: {exp_dict['mu']}")
    """
    from .layers import MSN
    
    if not isinstance(model, MSN):
        raise TypeError("Model must be an MSN instance")
    
    edge = model.layers[layer_idx].edges[out_idx][in_idx]
    mu, lam = edge.exponents()
    
    return {
        "mu": mu.detach().cpu().tolist(),
        "lam": lam.detach().cpu().tolist(),
        "a": edge.a.detach().cpu().tolist(),
        "b": edge.b.detach().cpu().tolist(),
    }


def get_flat_exponents(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Get all exponents from an MSN as flat tensors.
    
    Parameters
    ----------
    model : nn.Module
        MSN model
    
    Returns
    -------
    dict
        Dictionary with 'mu' and 'lam' tensors containing all exponents
    """
    from .layers import MSN
    
    if not isinstance(model, MSN):
        raise TypeError("Model must be an MSN instance")
    
    all_mu = []
    all_lam = []
    
    for layer in model.layers:
        for edge in layer.iter_edges():
            mu, lam = edge.exponents()
            all_mu.append(mu.detach())
            all_lam.append(lam.detach())
    
    return {
        "mu": torch.cat(all_mu),
        "lam": torch.cat(all_lam),
    }


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping (default: 10)
    min_delta : float
        Minimum change to qualify as improvement (default: 1e-6)
    
    Example
    -------
    >>> early_stop = EarlyStopping(patience=10)
    >>> for epoch in range(1000):
    ...     val_loss = train_epoch()
    ...     if early_stop(val_loss):
    ...         print(f"Early stopping at epoch {epoch}")
    ...         break
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        
        self.counter += 1
        return self.counter >= self.patience
