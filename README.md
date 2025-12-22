# Müntz-Szász Networks (MSN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

<!-- arXiv badge will be added after submission -->

**Neural Architectures with Learnable Power-Law Bases**

<p align="center">
  <img src="figures/msn_architecture.png" width="800" alt="MSN Architecture">
</p>

## Overview

**Müntz-Szász Networks (MSN)** are a novel neural network architecture that replaces fixed activation functions with learnable fractional power bases. Unlike standard MLPs that use fixed activations (ReLU, tanh, sigmoid), MSN learns the *exponents* of power functions alongside the coefficients:

$$\phi(x) = \sum_k a_k |x|^{\mu_k} + \sum_k b_k \text{sgn}(x)|x|^{\lambda_k}$$

where $\{\mu_k, \lambda_k\}$ are **learned exponents**.

### Key Results

| Task | MSN Error | MLP Error | Improvement | Parameter Reduction |
|------|-----------|-----------|-------------|---------------------|
| √x (supervised) | 0.00224 | 0.01043 | **4.6×** | **10×** |
| Cusp singularity | 0.00500 | 0.02175 | **4.4×** | **3×** |
| Singular ODE (PINN) | 0.0529 | 0.1677 | **3.2×** | **5×** |

### Why MSN?

- **Singular functions**: MSN achieves **$\mathcal{O}(\delta^2)$ error** for power functions when the learned exponent is within $\delta$ of the true exponent. MLPs require $\mathcal{O}(\epsilon^{-1/\alpha})$ neurons.
- **Interpretability**: Learned exponents reveal solution structure (e.g., $\mu \approx 0.5$ for $\sqrt{x}$)
- **Physics applications**: Boundary layers, fracture mechanics, corner singularities

## Installation

### From source (recommended)

```bash
git clone https://github.com/ReFractals/muntz-szasz-networks.git
cd muntz-szasz-networks
pip install -e .
```

### Dependencies only

```bash
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy ≥ 1.20
- Matplotlib ≥ 3.5

## Quick Start

### Basic Usage

```python
import torch
from msn import MSN

# Create MSN for 1D function approximation
model = MSN(
    dims=[1, 8, 8, 1],  # Input -> 8 -> 8 -> Output
    Ke=6,               # 6 even exponents
    Ko=6,               # 6 odd exponents
    p_max_even=4.0,     # Max exponent value
    exponent_mode="bounded"  # Stable parameterization
)

# Forward pass
x = torch.rand(100, 1)
y = model(x)

# Access learned exponents
mu, lam = model.layers[0].edges[0][0].exponents()
print(f"Learned even exponents: {mu.tolist()}")
```

### Training with Stabilization

MSN requires special training techniques for stability:

```python
from msn import MSN, MSNTrainer

model = MSN(dims=[1, 8, 8, 1])
trainer = MSNTrainer(
    model,
    lr=1e-3,
    lr_exp_mult=0.02,      # Exponents learn 50× slower
    warmup_steps=500,       # Freeze exponents initially
    exp_grad_clip=0.05,     # Clip exponent gradients
    use_muntz=True,         # Müntz divergence regularizer
    use_l1=True,            # Coefficient sparsity
)

# Training loop
for step in range(6000):
    loss = compute_your_loss(model, data)
    metrics = trainer.step(loss)
    
    if step % 500 == 0:
        print(f"Step {step}: loss={metrics['loss']:.4f}")
```

### PINN Example (Singular ODE)

Solve $u'(x) = \frac{1}{2\sqrt{x}}$ with $u(0) = 0$ (solution: $u(x) = \sqrt{x}$):

```python
import torch
from msn import MSN, MSNTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MSN(dims=[1, 8, 8, 1], Ke=6, Ko=6, p_max_even=2.0).to(device)
trainer = MSNTrainer(model, lr=1e-3, warmup_steps=500)

def pinn_loss(model):
    # Collocation points (biased toward singularity at x=0)
    x = torch.rand(2048, 1, device=device) ** 2
    x.requires_grad_(True)
    
    u = model(x)
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    
    # PDE residual: u' - 1/(2√x) = 0
    rhs = 0.5 / torch.sqrt(x + 1e-12)
    pde_loss = torch.mean((u_x - rhs) ** 2)
    
    # Boundary condition: u(0) = 0
    x0 = torch.zeros(256, 1, device=device)
    bc_loss = torch.mean(model(x0) ** 2)
    
    return pde_loss + 200 * bc_loss

# Train
for step in range(6000):
    loss = pinn_loss(model)
    metrics = trainer.step(loss)
```

## Architecture Details

### Müntz Edge

The fundamental building block computes:

```
φ(x) = Σ_k a_k |x|^{μ_k} + Σ_k b_k sgn(x)|x|^{λ_k}
       \_____________/   \____________________/
         Even part           Odd part
```

- **Even exponents** ($\mu_k$): For symmetric functions
- **Odd exponents** ($\lambda_k$): For antisymmetric functions
- **Coefficients** ($a_k, b_k$): Standard learnable weights

### Exponent Parameterization

Direct optimization of exponents is unstable. We use a bounded parameterization:

$$\mu_k = \epsilon + (p_{\max} - 2\epsilon) \cdot \text{sort}(\sigma(\mathbf{r}))_k$$

This ensures:
- Strict positivity: $\mu_k \in (\epsilon, p_{\max} - \epsilon)$
- Ordering: $\mu_1 < \mu_2 < \cdots < \mu_K$
- Differentiability (almost everywhere)
- Bounded gradients: $\|\partial\mu/\partial r\|_\infty \leq p_{\max}/4$

### Müntz Regularizer

Encourages exponents to satisfy the Müntz condition:

$$\mathcal{R}_{\text{Müntz}} = \text{ReLU}(C - D(\mu, \lambda))$$

where $D(\mu, \lambda) = \sum_k 1/\mu_k + \sum_k 1/\lambda_k$.

## Repository Structure

```
muntz-szasz-networks/
├── msn/                    # Core package
│   ├── __init__.py        
│   ├── layers.py          # MuntzEdge, MSNLayer, MSN
│   ├── parameterizations.py  # Bounded exponent maps
│   ├── regularizers.py    # Müntz divergence regularizer
│   ├── training.py        # MSNTrainer with stabilization
│   ├── baselines.py       # MLP baseline
│   └── utils.py           # Utilities
├── experiments/
│   ├── supervised/        # √x, cusp, polynomial benchmarks
│   └── pinn/              # Singular ODE, boundary-layer BVP
├── notebooks/
│   └── MSN_Demo.ipynb     # Interactive demonstration
├── figures/               # Paper figures
├── requirements.txt
├── setup.py
└── README.md
```

## Reproducing Paper Results

### Supervised Regression

```bash
cd experiments/supervised
python run_all.py --seeds 0 1 2
```

### PINN Benchmarks

```bash
cd experiments/pinn
python run_sqrt_ode.py --seeds 0 1 2
python run_boundary_layer.py --eps 0.05 0.02 --seeds 0 1 2
```

### Generating Figures

```bash
python scripts/generate_figures.py
```

## Theoretical Background

MSN is grounded in the **Müntz-Szász theorem** (1914):

> The span of $\{1, x^{\lambda_1}, x^{\lambda_2}, \ldots\}$ with $0 < \lambda_1 < \lambda_2 < \cdots$ is dense in $C[0,1]$ if and only if $\sum_{k=1}^\infty 1/\lambda_k = \infty$.

**Key theoretical results:**

1. **Universal Approximation** (Theorem 3): MSN can approximate any continuous function on compact domains.

2. **Approximation Rate** (Theorem 4): For $f(x) = |x|^\alpha$:
   - If $\mu_j = \alpha$: error is **exactly zero**
   - If $\min_k |\mu_k - \alpha| = \delta$: error is $\mathcal{O}(\delta^2)$
   
   Compare to ReLU MLPs: $\mathcal{O}(\epsilon^{-1/2})$ neurons for error $\epsilon$.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nguessan2024muntz,
  title={Müntz-Szász Networks: Neural Architectures with Learnable Power-Law Bases},
  author={N'guessan, Gnankan Landry Regis},
  journal={arXiv preprint},
  year={2024},
  note={Code: https://github.com/ReFractals/muntz-szasz-networks}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- African Institute for Mathematical Sciences (AIMS)
- Nelson Mandela African Institution of Science and Technology (NM-AIST)

## Contact

- **Author**: Gnankan Landry Regis N'guessan
- **Email**: rnguessan@aimsric.org
- **Affiliation**: Axiom Research Group / AIMS RIC / NM-AIST
