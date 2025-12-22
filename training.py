"""
Training Utilities for Müntz-Szász Networks
============================================

MSN requires specialized training techniques:
1. Two-time-scale optimization (exponents evolve slower than coefficients)
2. Exponent warmup (freeze exponents initially)
3. Exponent gradient clipping (prevent instabilities)

This module provides utilities to implement these techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from .utils import msn_exponent_params, msn_coeff_params
from .layers import MSN


def make_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    lr_exp_mult: float = 0.02,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Create an optimizer with separate learning rates for exponents and coefficients.
    
    For MSN models, exponents are updated with a smaller learning rate
    (two-time-scale optimization) to ensure stable training.
    
    Parameters
    ----------
    model : nn.Module
        Model to optimize (MSN or standard nn.Module)
    lr : float
        Base learning rate (default: 1e-3)
    lr_exp_mult : float
        Multiplier for exponent learning rate (default: 0.02)
        Exponents use lr * lr_exp_mult
    weight_decay : float
        L2 regularization (default: 0.0)
    
    Returns
    -------
    torch.optim.Optimizer
        Adam optimizer with parameter groups
    
    Example
    -------
    >>> model = MSN(dims=[1, 8, 1])
    >>> optimizer = make_optimizer(model, lr=1e-3, lr_exp_mult=0.02)
    >>> # Coefficients: lr=1e-3, Exponents: lr=2e-5
    """
    if isinstance(model, MSN):
        exp_params = msn_exponent_params(model)
        coeff_params = msn_coeff_params(model)
        
        return torch.optim.Adam([
            {"params": coeff_params, "lr": lr, "weight_decay": weight_decay},
            {"params": exp_params, "lr": lr * lr_exp_mult, "weight_decay": 0.0},
        ])
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def clip_exponent_grads(model: nn.Module, clip_norm: float = 0.05) -> None:
    """
    Clip gradients of exponent parameters only.
    
    This prevents large gradient updates to exponents, which can cause
    training instabilities.
    
    Parameters
    ----------
    model : nn.Module
        MSN model
    clip_norm : float
        Maximum gradient norm (default: 0.05)
    
    Example
    -------
    >>> loss.backward()
    >>> clip_exponent_grads(model, clip_norm=0.05)
    >>> optimizer.step()
    """
    if not isinstance(model, MSN):
        return
    
    exp_params = msn_exponent_params(model)
    torch.nn.utils.clip_grad_norm_(exp_params, clip_norm)


def freeze_exponents(model: nn.Module) -> None:
    """
    Freeze exponent parameters (disable gradient updates).
    
    Useful during warmup phase when only coefficients should be trained.
    
    Parameters
    ----------
    model : nn.Module
        MSN model
    
    Example
    -------
    >>> model = MSN(dims=[1, 8, 1])
    >>> freeze_exponents(model)
    >>> # Train for warmup steps...
    >>> unfreeze_exponents(model)
    >>> # Continue training with exponent updates
    """
    if not isinstance(model, MSN):
        return
    
    for p in msn_exponent_params(model):
        p.requires_grad_(False)


def unfreeze_exponents(model: nn.Module) -> None:
    """
    Unfreeze exponent parameters (enable gradient updates).
    
    Parameters
    ----------
    model : nn.Module
        MSN model
    """
    if not isinstance(model, MSN):
        return
    
    for p in msn_exponent_params(model):
        p.requires_grad_(True)


class MSNTrainer:
    """
    Trainer class for Müntz-Szász Networks with all stabilization techniques.
    
    Implements:
    - Two-time-scale optimization
    - Exponent warmup
    - Exponent gradient clipping
    - Global gradient clipping
    - Optional Müntz regularization
    - Optional L1 coefficient regularization
    
    Parameters
    ----------
    model : MSN
        MSN model to train
    lr : float
        Base learning rate (default: 1e-3)
    lr_exp_mult : float
        Exponent learning rate multiplier (default: 0.02)
    warmup_steps : int
        Number of steps to freeze exponents (default: 500)
    exp_grad_clip : float
        Gradient clip norm for exponents (default: 0.05)
    global_grad_clip : float
        Global gradient clip norm (default: 1.0)
    beta_muntz : float
        Weight for Müntz regularizer (default: 1e-2)
    beta_l1 : float
        Weight for L1 coefficient regularizer (default: 1e-4)
    use_muntz : bool
        Whether to use Müntz regularization (default: True)
    use_l1 : bool
        Whether to use L1 regularization (default: True)
    device : str
        Device to train on (default: 'cuda' if available)
    
    Example
    -------
    >>> model = MSN(dims=[1, 8, 8, 1])
    >>> trainer = MSNTrainer(model, lr=1e-3, warmup_steps=500)
    >>> 
    >>> for step in range(6000):
    ...     loss = compute_loss(model, data)
    ...     metrics = trainer.step(loss)
    ...     if step % 500 == 0:
    ...         print(f"Step {step}: loss={metrics['loss']:.4f}")
    """
    
    def __init__(
        self,
        model: MSN,
        lr: float = 1e-3,
        lr_exp_mult: float = 0.02,
        warmup_steps: int = 500,
        exp_grad_clip: float = 0.05,
        global_grad_clip: float = 1.0,
        beta_muntz: float = 1e-2,
        beta_l1: float = 1e-4,
        use_muntz: bool = True,
        use_l1: bool = True,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model.to(device)
        self.device = device
        self.warmup_steps = warmup_steps
        self.exp_grad_clip = exp_grad_clip
        self.global_grad_clip = global_grad_clip
        self.beta_muntz = beta_muntz
        self.beta_l1 = beta_l1
        self.use_muntz = use_muntz
        self.use_l1 = use_l1
        
        self.optimizer = make_optimizer(model, lr=lr, lr_exp_mult=lr_exp_mult)
        self.step_count = 0
        self._warmup_done = False
        
        # Start with frozen exponents
        freeze_exponents(model)
    
    def step(self, loss: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Parameters
        ----------
        loss : torch.Tensor
            The base loss (e.g., MSE, PDE residual)
        
        Returns
        -------
        dict
            Dictionary with 'loss', 'max_grad', and optionally regularization terms
        """
        self.step_count += 1
        
        # Unfreeze exponents after warmup
        if self.step_count == self.warmup_steps + 1:
            unfreeze_exponents(self.model)
            self._warmup_done = True
        
        # Add regularization
        total_loss = loss
        metrics = {"base_loss": float(loss.detach().cpu())}
        
        if isinstance(self.model, MSN):
            if self.use_muntz:
                muntz_loss = self.model.muntz_regularizer(C=2.0)
                total_loss = total_loss + self.beta_muntz * muntz_loss
                metrics["muntz_reg"] = float(muntz_loss.detach().cpu())
            
            if self.use_l1:
                l1_loss = self.model.l1_coeff_regularizer()
                total_loss = total_loss + self.beta_l1 * l1_loss
                metrics["l1_reg"] = float(l1_loss.detach().cpu())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self._warmup_done:
            clip_exponent_grads(self.model, clip_norm=self.exp_grad_clip)
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.global_grad_clip)
        
        # Track max gradient
        max_grad = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                max_grad = max(max_grad, float(p.grad.abs().max().cpu()))
        metrics["max_grad"] = max_grad
        
        # Update
        self.optimizer.step()
        
        metrics["loss"] = float(total_loss.detach().cpu())
        metrics["step"] = self.step_count
        metrics["warmup_done"] = self._warmup_done
        
        return metrics
