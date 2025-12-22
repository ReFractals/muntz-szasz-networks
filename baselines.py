"""
Baseline Models
===============

Standard MLP baselines for comparison with MSN.
"""

import torch
import torch.nn as nn
from typing import Literal

from .utils import count_params


class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron baseline.
    
    Parameters
    ----------
    Din : int
        Input dimension (default: 1)
    H : int
        Hidden layer width (default: 64)
    Dout : int
        Output dimension (default: 1)
    depth : int
        Number of layers including output (default: 3)
    act : str
        Activation function: 'tanh' or 'silu' (default: 'tanh')
    
    Example
    -------
    >>> mlp = MLP(Din=1, H=64, Dout=1, depth=3)
    >>> x = torch.rand(100, 1)
    >>> y = mlp(x)  # shape: (100, 1)
    """
    
    def __init__(
        self,
        Din: int = 1,
        H: int = 64,
        Dout: int = 1,
        depth: int = 3,
        act: Literal["tanh", "silu"] = "tanh",
    ):
        super().__init__()
        
        act_layer = nn.Tanh if act == "tanh" else nn.SiLU
        
        layers = []
        d = Din
        for _ in range(depth - 1):
            layers.extend([nn.Linear(d, H), act_layer()])
            d = H
        layers.append(nn.Linear(d, Dout))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_param_matched_mlp(
    Din: int,
    Dout: int,
    target_params: int,
    depth: int = 3,
    act: str = "tanh",
    H_min: int = 4,
    H_max: int = 256,
) -> tuple:
    """
    Build an MLP with approximately the same number of parameters as a target.
    
    Searches for a hidden dimension H such that the resulting MLP has at least
    `target_params` parameters.
    
    Parameters
    ----------
    Din : int
        Input dimension
    Dout : int
        Output dimension
    target_params : int
        Target number of parameters to match
    depth : int
        Number of layers (default: 3)
    act : str
        Activation function (default: 'tanh')
    H_min : int
        Minimum hidden dimension to search (default: 4)
    H_max : int
        Maximum hidden dimension to search (default: 256)
    
    Returns
    -------
    model : MLP
        The constructed MLP
    params : int
        Actual number of parameters
    H : int
        Hidden dimension used
    
    Example
    -------
    >>> from msn import MSN
    >>> msn_model = MSN(dims=[1, 8, 1])
    >>> target = count_params(msn_model)
    >>> mlp, actual_params, H = build_param_matched_mlp(1, 1, target)
    >>> print(f"MSN: {target}, MLP: {actual_params} (H={H})")
    """
    for H in range(H_min, H_max + 1):
        model = MLP(Din=Din, H=H, Dout=Dout, depth=depth, act=act)
        params = count_params(model)
        if params >= target_params:
            return model, params, H
    
    # If we couldn't find a match, return the largest
    model = MLP(Din=Din, H=H_max, Dout=Dout, depth=depth, act=act)
    return model, count_params(model), H_max
