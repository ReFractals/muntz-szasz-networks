"""
Müntz-Szász Network Layers
==========================

This module implements the core building blocks of Müntz-Szász Networks:
- MuntzEdge: A single edge computing φ(x) = Σ_k a_k|x|^{μ_k} + Σ_k b_k sgn(x)|x|^{λ_k}
- MSNLayer: A layer connecting all input-output pairs via Müntz edges
- MSN: The full network stacking MSN layers with inter-layer nonlinearities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Literal

from .parameterizations import make_ordered_bounded, make_cumsum_softplus, safe_pow_abs


class MuntzEdge(nn.Module):
    """
    A Müntz edge computing a learnable power-law transformation.
    
    Each edge computes:
        φ(x) = Σ_{k=1}^{K_e} a_k |x|^{μ_k} + Σ_{k=1}^{K_o} b_k sgn(x)|x|^{λ_k}
    
    where:
    - μ_k are even exponents (for symmetric functions)
    - λ_k are odd exponents (for antisymmetric functions)
    - a_k, b_k are learnable coefficients
    
    Parameters
    ----------
    Ke : int
        Number of even exponents (default: 6)
    Ko : int
        Number of odd exponents (default: 6)
    share_exponents : dict, optional
        Dictionary with 'raw_even' and 'raw_odd' parameters to share
    init_scale : float
        Scale for coefficient initialization (default: 0.02)
    p_max_even : float
        Maximum value for even exponents (default: 4.0)
    p_max_odd : float
        Maximum value for odd exponents (default: 4.0)
    anchor_even : list, optional
        Fixed exponent values to include in even exponents
    anchor_odd : list, optional
        Fixed exponent values to include in odd exponents
    exponent_mode : str
        Parameterization mode: 'bounded' or 'cumsum' (default: 'bounded')
    
    Example
    -------
    >>> edge = MuntzEdge(Ke=6, Ko=6)
    >>> x = torch.randn(100)
    >>> y = edge(x)  # shape: (100,)
    """
    
    def __init__(
        self,
        Ke: int = 6,
        Ko: int = 6,
        share_exponents: Optional[Dict[str, nn.Parameter]] = None,
        init_scale: float = 0.02,
        p_max_even: float = 4.0,
        p_max_odd: float = 4.0,
        anchor_even: Optional[List[float]] = None,
        anchor_odd: Optional[List[float]] = None,
        exponent_mode: Literal["bounded", "cumsum"] = "bounded",
    ):
        super().__init__()
        self.Ke = Ke
        self.Ko = Ko
        self.p_max_even = p_max_even
        self.p_max_odd = p_max_odd
        self.anchor_even = anchor_even or []
        self.anchor_odd = anchor_odd or []
        self.exponent_mode = exponent_mode
        
        # Learnable coefficients
        self.a = nn.Parameter(init_scale * torch.randn(Ke))
        self.b = nn.Parameter(init_scale * torch.randn(Ko))
        
        # Exponents (either own parameters or shared)
        if share_exponents is None:
            self.raw_even = nn.Parameter(torch.randn(Ke))
            self.raw_odd = nn.Parameter(torch.randn(Ko))
        else:
            self.raw_even = share_exponents["raw_even"]
            self.raw_odd = share_exponents["raw_odd"]
    
    def exponents(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the current even and odd exponents.
        
        Returns
        -------
        mu : torch.Tensor
            Even exponents of shape (Ke,)
        lam : torch.Tensor
            Odd exponents of shape (Ko,)
        """
        if self.exponent_mode == "bounded":
            mu = make_ordered_bounded(self.raw_even, p_max=self.p_max_even)
            lam = make_ordered_bounded(self.raw_odd, p_max=self.p_max_odd)
        elif self.exponent_mode == "cumsum":
            mu = make_cumsum_softplus(self.raw_even, min_val=0.05)
            lam = make_cumsum_softplus(self.raw_odd, min_val=0.05)
        else:
            raise ValueError(f"exponent_mode must be 'bounded' or 'cumsum', got {self.exponent_mode}")
        
        # Merge with anchor exponents if specified
        if len(self.anchor_even) > 0:
            anchor = torch.tensor(self.anchor_even, device=mu.device, dtype=mu.dtype)
            mu_all = torch.cat([mu, anchor])
            mu_all, _ = torch.sort(mu_all)
            mu = mu_all[:self.Ke]
        
        if len(self.anchor_odd) > 0:
            anchor = torch.tensor(self.anchor_odd, device=lam.device, dtype=lam.dtype)
            lam_all = torch.cat([lam, anchor])
            lam_all, _ = torch.sort(lam_all)
            lam = lam_all[:self.Ko]
        
        return mu, lam
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing the Müntz edge output.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size,) or (batch_size, 1)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size,)
        """
        if x.dim() == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)
        
        mu, lam = self.exponents()
        
        # Even part: Σ_k a_k |x|^{μ_k}
        pe = safe_pow_abs(x, mu)  # (batch, Ke)
        even_part = pe @ self.a  # (batch,)
        
        # Odd part: Σ_k b_k sgn(x)|x|^{λ_k}
        po = safe_pow_abs(x, lam)  # (batch, Ko)
        s = torch.sign(x)[..., None]  # (batch, 1)
        odd_part = (s * po) @ self.b  # (batch,)
        
        return even_part + odd_part


class MSNLayer(nn.Module):
    """
    A Müntz-Szász Network layer.
    
    Connects all input dimensions to all output dimensions via Müntz edges:
        L(x)_j = Σ_{i=1}^{D_in} φ_{ij}(x_i) + c_j
    
    Parameters
    ----------
    Din : int
        Input dimension
    Dout : int
        Output dimension
    Ke : int
        Number of even exponents per edge (default: 6)
    Ko : int
        Number of odd exponents per edge (default: 6)
    share : str
        Exponent sharing mode: 'layer' (share across edges) or 'none' (default: 'layer')
    p_max_even : float
        Maximum even exponent value (default: 4.0)
    p_max_odd : float
        Maximum odd exponent value (default: 4.0)
    anchor_even : list, optional
        Fixed even exponent values
    anchor_odd : list, optional
        Fixed odd exponent values
    exponent_mode : str
        Parameterization: 'bounded' or 'cumsum' (default: 'bounded')
    
    Example
    -------
    >>> layer = MSNLayer(Din=2, Dout=4, Ke=6, Ko=6)
    >>> x = torch.randn(100, 2)
    >>> y = layer(x)  # shape: (100, 4)
    """
    
    def __init__(
        self,
        Din: int,
        Dout: int,
        Ke: int = 6,
        Ko: int = 6,
        share: Literal["layer", "none"] = "layer",
        p_max_even: float = 4.0,
        p_max_odd: float = 4.0,
        anchor_even: Optional[List[float]] = None,
        anchor_odd: Optional[List[float]] = None,
        exponent_mode: Literal["bounded", "cumsum"] = "bounded",
    ):
        super().__init__()
        self.Din = Din
        self.Dout = Dout
        
        # Setup exponent sharing
        share_exponents = None
        if share == "layer":
            self.raw_even = nn.Parameter(torch.randn(Ke))
            self.raw_odd = nn.Parameter(torch.randn(Ko))
            share_exponents = {"raw_even": self.raw_even, "raw_odd": self.raw_odd}
        
        # Create edge grid: Dout x Din
        self.edges = nn.ModuleList([
            nn.ModuleList([
                MuntzEdge(
                    Ke=Ke, Ko=Ko,
                    share_exponents=share_exponents,
                    p_max_even=p_max_even,
                    p_max_odd=p_max_odd,
                    anchor_even=anchor_even,
                    anchor_odd=anchor_odd,
                    exponent_mode=exponent_mode,
                )
                for _ in range(Din)
            ])
            for _ in range(Dout)
        ])
        
        self.bias = nn.Parameter(torch.zeros(Dout))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MSN layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, Din)
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch_size, Dout)
        """
        B = x.shape[0]
        out = torch.zeros(B, self.Dout, device=x.device, dtype=x.dtype)
        
        for j in range(self.Dout):
            s = 0.0
            for i in range(self.Din):
                s = s + self.edges[j][i](x[:, i])
            out[:, j] = s + self.bias[j]
        
        return out
    
    def iter_edges(self):
        """Iterate over all edges in this layer."""
        for row in self.edges:
            for edge in row:
                yield edge


class MSN(nn.Module):
    """
    Müntz-Szász Network.
    
    A neural network architecture where each edge computes a Müntz polynomial
    with learnable exponents. The network stacks MSN layers with inter-layer
    tanh nonlinearities.
    
    Parameters
    ----------
    dims : list of int
        Layer dimensions, e.g., [1, 8, 8, 1] for input->8->8->output
    Ke : int
        Number of even exponents per edge (default: 6)
    Ko : int
        Number of odd exponents per edge (default: 6)
    share : str
        Exponent sharing: 'layer' or 'none' (default: 'layer')
    p_max_even : float
        Maximum even exponent (default: 4.0)
    p_max_odd : float
        Maximum odd exponent (default: 4.0)
    anchor_even : list, optional
        Fixed even exponent values to include
    anchor_odd : list, optional
        Fixed odd exponent values to include
    exponent_mode : str
        Parameterization: 'bounded' or 'cumsum' (default: 'bounded')
    
    Example
    -------
    >>> # Create MSN for 1D function approximation
    >>> model = MSN(dims=[1, 8, 8, 1], Ke=6, Ko=6)
    >>> x = torch.rand(100, 1)
    >>> y = model(x)  # shape: (100, 1)
    >>> 
    >>> # Access learned exponents
    >>> mu, lam = model.layers[0].edges[0][0].exponents()
    """
    
    def __init__(
        self,
        dims: List[int],
        Ke: int = 6,
        Ko: int = 6,
        share: Literal["layer", "none"] = "layer",
        p_max_even: float = 4.0,
        p_max_odd: float = 4.0,
        anchor_even: Optional[List[float]] = None,
        anchor_odd: Optional[List[float]] = None,
        exponent_mode: Literal["bounded", "cumsum"] = "bounded",
    ):
        super().__init__()
        self.dims = dims
        
        self.layers = nn.ModuleList([
            MSNLayer(
                Din=a, Dout=b,
                Ke=Ke, Ko=Ko,
                share=share,
                p_max_even=p_max_even,
                p_max_odd=p_max_odd,
                anchor_even=anchor_even,
                anchor_odd=anchor_odd,
                exponent_mode=exponent_mode,
            )
            for a, b in zip(dims[:-1], dims[1:])
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MSN.
        
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch_size, dims[0])
        
        Returns
        -------
        torch.Tensor
            Output of shape (batch_size, dims[-1])
        """
        for k, layer in enumerate(self.layers):
            x = layer(x)
            # Apply tanh between layers (not after the last one)
            if k < len(self.layers) - 1:
                x = torch.tanh(x)
        return x
    
    def muntz_regularizer(self, C: float = 2.0) -> torch.Tensor:
        """
        Compute the Müntz divergence regularizer.
        
        Encourages exponents to satisfy (approximately) the Müntz condition
        Σ 1/λ_k = ∞ by penalizing configurations where Σ 1/μ_k + Σ 1/λ_k < C.
        
        Parameters
        ----------
        C : float
            Threshold for the divergence sum (default: 2.0)
        
        Returns
        -------
        torch.Tensor
            Regularization loss (0 if condition satisfied, positive otherwise)
        """
        total = 0.0
        for layer in self.layers:
            for edge in layer.iter_edges():
                mu, lam = edge.exponents()
                total = total + torch.sum(1.0 / (mu + 1e-12))
                total = total + torch.sum(1.0 / (lam + 1e-12))
        return F.relu(C - total)
    
    def l1_coeff_regularizer(self) -> torch.Tensor:
        """
        L1 regularization on coefficients for sparsity.
        
        Returns
        -------
        torch.Tensor
            Sum of absolute values of all coefficients
        """
        total = 0.0
        for layer in self.layers:
            for edge in layer.iter_edges():
                total = total + torch.sum(torch.abs(edge.a))
                total = total + torch.sum(torch.abs(edge.b))
        return total
    
    def exponent_growth_penalty(self, max_target: float = 10.0) -> torch.Tensor:
        """
        Penalty for exponents exceeding a maximum target.
        
        Parameters
        ----------
        max_target : float
            Maximum desired exponent value (default: 10.0)
        
        Returns
        -------
        torch.Tensor
            Penalty for exponents exceeding max_target
        """
        total = 0.0
        for layer in self.layers:
            for edge in layer.iter_edges():
                mu, lam = edge.exponents()
                total = total + torch.mean(F.relu(mu - max_target) ** 2)
                total = total + torch.mean(F.relu(lam - max_target) ** 2)
        return total
    
    def get_all_exponents(self) -> Dict[str, List[List[float]]]:
        """
        Get all learned exponents from the network.
        
        Returns
        -------
        dict
            Dictionary with 'mu' and 'lam' keys, each containing nested lists
            of exponent values per layer/edge
        """
        all_mu = []
        all_lam = []
        
        for layer in self.layers:
            layer_mu = []
            layer_lam = []
            for edge in layer.iter_edges():
                mu, lam = edge.exponents()
                layer_mu.append(mu.detach().cpu().tolist())
                layer_lam.append(lam.detach().cpu().tolist())
            all_mu.append(layer_mu)
            all_lam.append(layer_lam)
        
        return {"mu": all_mu, "lam": all_lam}
