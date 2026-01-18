# Multilevel Monte Carlo Unbiased Gradient Estimation for Deep Latent Variable Models

This repository implements and compares several unbiased gradient estimators for deep latent variable models, reproducing the first numerical experiment from [Shi & Cornish (2021)](http://proceedings.mlr.press/v130/shi21d.html). The project demonstrates the application of Multilevel Monte Carlo (MLMC) methods to obtain unbiased estimates of gradients, which are crucial for stochastic gradient descent optimization in latent variable models.

## What This Demonstrates

- **Unbiased gradient estimation** using Multilevel Monte Carlo (MLMC) techniques for latent variable models
- **Comparison of four estimators**: ML-SS (Single Sample), ML-RR (Russian Roulette), SUMO (Stochastically Unbiased Marginalization Objective), and IWAE (Importance Weighted Autoencoder)
- **Bias-variance analysis** of gradient estimators across different computational costs
- **Stochastic gradient descent** performance using unbiased vs. biased gradient estimates
- **Implementation** of Russian Roulette and Single Sample estimators for infinite series truncation
- **Numerical validation** on a tractable Gaussian latent variable model with known analytical solutions

## Key Results

- **Unbiased estimators** (ML-SS, ML-RR, SUMO) provide accurate gradient estimates with controlled variance
- **IWAE shows bias** that decreases with sample size but remains present at finite computational budgets
- **MLMC methods** achieve lower bias-squared compared to IWAE at similar computational costs
- **Gradient estimation accuracy** improves near the true parameter value for all methods
- See `reports/` for detailed analysis and visualizations
- See `notebooks/MLMC_Main.ipynb` for complete experimental results

## Repository Layout

```
mlmc_unbiased_gradient_project/
├── src/                    # Source code
│   └── Code.py            # Main implementation of estimators
├── notebooks/              # Jupyter notebooks
│   └── MLMC_Main.ipynb    # Main analysis notebook
├── reports/                # Reports and documentation
├── tests/                  # Unit tests (optional)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md              # This file
```

## Quickstart

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mlmc_unbiased_gradient_project

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Start Jupyter notebook
jupyter notebook

# Open notebooks/MLMC_Main.ipynb and run all cells
# This will:
# 1. Generate synthetic data from a Gaussian latent variable model
# 2. Compare likelihood estimates across all four methods
# 3. Analyze bias and variance of gradient estimators
# 4. Evaluate SGD performance with different gradient estimators
```

### Minimal Example

```python
import sys
import os
sys.path.append('src')
import Code
import numpy as np

# Generate synthetic data
theta_true = np.random.normal(0, 1)
x, _ = Code.joint_probability(theta_true, dim=20)

# Set up encoder parameters
dim = 20
A = 0.5 * np.eye(dim)
b = 0.5 * theta_true * np.ones(dim)
noised_A, noised_b = Code.noised_params(A, b, dim=dim)

# Estimate likelihood using SUMO
r = 0.6
n_simulations = 10
likelihood_estimate = Code.log_likelihood_SUMO(
    r, theta_true, x, noised_A, noised_b, n_simulations
)
print(f"Estimated log-likelihood: {likelihood_estimate}")
```

## Method Overview

This project implements unbiased gradient estimation methods for latent variable models where the marginal likelihood is intractable. The key challenge is that standard importance-weighted estimators (like IWAE) are biased, which can lead to suboptimal parameter estimates.

### Methods Implemented

1. **ML-SS (Multilevel Single Sample)**: Uses a single random truncation level with appropriate weighting
2. **ML-RR (Multilevel Russian Roulette)**: Applies Russian Roulette estimator to the multilevel decomposition
3. **SUMO**: Stochastically Unbiased Marginalization Objective using Russian Roulette on incremental differences
4. **IWAE**: Importance Weighted Autoencoder (biased baseline for comparison)

All methods use importance sampling with an encoder network `q_φ(z|x)` to approximate the intractable posterior `p_θ(z|x)`.

### Mathematical Framework

The marginal likelihood is:
```
log p_θ(x) = log ∫ p_θ(x,z) dz
```

The MLMC methods decompose this as:
```
I_∞ = E[I_0] + Σ E[Δ_k]
```

where `Δ_k` represents the difference between estimates at different sample sizes, enabling unbiased estimation through random truncation.

## Notes and Limitations

- **Computational cost**: Unbiased estimators require random truncation, leading to variable computational costs per iteration
- **Variance control**: The choice of geometric distribution parameter `r` affects the variance-bias tradeoff
- **Model assumptions**: Current implementation uses a Gaussian latent variable model for tractability; extension to deep models requires additional implementation
- **Encoder quality**: Performance depends on how well the encoder `q_φ(z|x)` approximates the true posterior
- **Numerical stability**: Some estimators may encounter numerical issues (e.g., log(0)) in edge cases

## References

- **Shi, Y., & Cornish, R. (2021)**: [On Multilevel Monte Carlo Unbiased Gradient Estimation for Deep Latent Variable Models](http://proceedings.mlr.press/v130/shi21d.html). ICML 2021.

- **Rainforth, T., et al. (2018)**: [Tighter Variational Bounds are Not Necessarily Better](https://arxiv.org/abs/1802.04537). ICML 2018.

- **Blanchet, J. H., & Glynn, P. W. (2015)**: [Unbiased Monte Carlo for optimization and functions of expectations via multi-level randomization](https://web.stanford.edu/~glynn/papers/2015/BlanchetG15.html).

- **Luo, Y., et al. (2020)**: [SUMO: Unbiased estimation of log marginal probability for latent variable models](https://arxiv.org/abs/2004.00353). ICML 2020.

- **Burda, Y., Grosse, R., & Salakhutdinov, R. (2016)**: [Importance Weighted Autoencoders](https://arxiv.org/pdf/1509.00519). ICLR 2016.

## Authors

- Tom Rossa
- Axel Pinçon  
- Naïl Khelifa

## License

Distributed under the MIT License. See `LICENSE` for more information.
