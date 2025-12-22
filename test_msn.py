"""
Basic tests for Müntz-Szász Networks.

Run with: pytest tests/
"""

import torch
import pytest


def test_import():
    """Test that the package can be imported."""
    from msn import MSN, MuntzEdge, MSNLayer
    from msn import make_ordered_bounded, safe_pow_abs
    from msn import MSNTrainer, make_optimizer
    

def test_msn_forward():
    """Test MSN forward pass."""
    from msn import MSN
    
    model = MSN(dims=[1, 4, 1], Ke=4, Ko=4)
    x = torch.rand(10, 1)
    y = model(x)
    
    assert y.shape == (10, 1)
    assert not torch.isnan(y).any()


def test_msn_forward_multidim():
    """Test MSN with multiple input/output dimensions."""
    from msn import MSN
    
    model = MSN(dims=[3, 8, 8, 2], Ke=4, Ko=4)
    x = torch.rand(10, 3)
    y = model(x)
    
    assert y.shape == (10, 2)


def test_muntz_regularizer():
    """Test Müntz divergence regularizer."""
    from msn import MSN
    
    model = MSN(dims=[1, 4, 1], Ke=6, Ko=6)
    reg = model.muntz_regularizer(C=2.0)
    
    assert reg.shape == ()
    assert reg >= 0


def test_bounded_parameterization():
    """Test bounded exponent parameterization."""
    from msn.parameterizations import make_ordered_bounded
    
    raw = torch.randn(6)
    mu = make_ordered_bounded(raw, p_max=4.0, eps=1e-3)
    
    # Check positivity
    assert (mu > 0).all()
    assert (mu < 4.0).all()
    
    # Check ordering
    assert (mu[1:] >= mu[:-1]).all()


def test_safe_pow_abs():
    """Test safe power computation."""
    from msn.parameterizations import safe_pow_abs
    
    x = torch.linspace(-1, 1, 100)
    p = torch.tensor([0.5, 1.0, 2.0])
    
    result = safe_pow_abs(x, p)
    
    assert result.shape == (100, 3)
    assert not torch.isnan(result).any()
    assert (result >= 0).all()


def test_exponent_access():
    """Test accessing learned exponents."""
    from msn import MSN
    from msn.utils import dump_exponents
    
    model = MSN(dims=[1, 4, 1], Ke=6, Ko=6)
    exp_dict = dump_exponents(model)
    
    assert "mu" in exp_dict
    assert "lam" in exp_dict
    assert len(exp_dict["mu"]) == 6
    assert len(exp_dict["lam"]) == 6


def test_trainer():
    """Test MSNTrainer."""
    from msn import MSN, MSNTrainer
    
    model = MSN(dims=[1, 4, 1], Ke=4, Ko=4)
    trainer = MSNTrainer(model, lr=1e-3, warmup_steps=10, device="cpu")
    
    # Simulate a few training steps
    x = torch.rand(32, 1)
    y = torch.sqrt(x)
    
    for _ in range(20):
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        metrics = trainer.step(loss)
    
    assert "loss" in metrics
    assert metrics["warmup_done"]  # After 10 steps


def test_gradient_flow():
    """Test that gradients flow through exponents."""
    from msn import MSN
    
    model = MSN(dims=[1, 4, 1], Ke=4, Ko=4)
    x = torch.rand(10, 1)
    y = torch.sqrt(x)
    
    pred = model(x)
    loss = torch.mean((pred - y) ** 2)
    loss.backward()
    
    # Check that exponent parameters have gradients
    for name, p in model.named_parameters():
        if "raw_even" in name or "raw_odd" in name:
            assert p.grad is not None
            assert not torch.isnan(p.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
