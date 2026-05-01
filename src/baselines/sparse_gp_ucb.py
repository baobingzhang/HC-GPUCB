"""Sparse GP-UCB baseline using inducing-point GP regression."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.kernels import InducingPointKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

from src.baselines.vanilla_ucb import beta_iwazaki


class _SparseGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood, m: int = 100):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        d = train_x.shape[-1]
        # Sample initial inducing points uniformly from training inputs (or random subset)
        n = train_x.shape[0]
        n_ind = min(m, max(2, n))
        idx = torch.randperm(n)[:n_ind]
        inducing = train_x[idx].clone()
        base = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
        self.covar_module = InducingPointKernel(base, inducing_points=inducing, likelihood=likelihood)

    def forward(self, x):
        from gpytorch.distributions import MultivariateNormal
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


@dataclass
class SparseUCBResult:
    X: Tensor
    Y: Tensor
    cumulative_regret: Tensor
    simple_regret: Tensor
    per_step_time: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)


def run_sparse_gp_ucb(
    objective: Callable[[Tensor], Tensor],
    bounds: Tensor,
    T: int,
    f_star: float,
    *,
    n_init: int = 5,
    delta: float = 0.05,
    noise_std: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    num_restarts: int = 8,
    raw_samples: int = 256,
    n_inducing: int = 100,
) -> SparseUCBResult:
    """Sparse GP-UCB: a sparse-inducing-point variational GP under the GP-UCB acquisition.

    Falls back to a regular SingleTaskGP for very small datasets where inducing points
    are not yet useful.
    """
    torch.manual_seed(seed)
    bounds = bounds.to(device=device, dtype=dtype)
    d = bounds.shape[-1]
    lo, hi = bounds[0], bounds[1]

    X_init = lo + (hi - lo) * torch.rand(n_init, d, device=device, dtype=dtype)
    Y_init = objective(X_init).unsqueeze(-1)
    if noise_std > 0:
        Y_init = Y_init + noise_std * torch.randn_like(Y_init)
    X = X_init.clone()
    Y = Y_init.clone()
    f_vals = list(objective(X_init).tolist())
    per_step_time: list[float] = []
    betas: list[float] = []

    for t in range(n_init + 1, T + 1):
        t_start = time.time()
        try:
            if X.shape[0] <= n_inducing:
                gp = SingleTaskGP(X, Y).to(device=device, dtype=dtype)
            else:
                lik = GaussianLikelihood().to(device=device, dtype=dtype)
                gp = _SparseGP(X, Y.squeeze(-1), lik, m=n_inducing).to(device=device, dtype=dtype)
                gp.likelihood = lik
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
        except Exception:
            gp = SingleTaskGP(X, Y).to(device=device, dtype=dtype)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

        beta_t = beta_iwazaki(t, d, delta)
        betas.append(beta_t)
        ucb = UpperConfidenceBound(gp, beta=beta_t)
        try:
            cand, _ = optimize_acqf(ucb, bounds=bounds, q=1,
                                    num_restarts=num_restarts, raw_samples=raw_samples)
        except Exception:
            cand = lo + (hi - lo) * torch.rand(1, d, device=device, dtype=dtype)
        f_t = objective(cand).item()
        y_t = f_t + (noise_std * torch.randn(1).item() if noise_std > 0 else 0.0)
        X = torch.cat([X, cand], dim=0)
        Y = torch.cat([Y, torch.tensor([[y_t]], device=device, dtype=dtype)], dim=0)
        f_vals.append(f_t)
        per_step_time.append(time.time() - t_start)

    f_vec = torch.tensor(f_vals, device=device, dtype=dtype)
    inst_regret = f_star - f_vec
    cum = torch.cumsum(inst_regret, dim=0)
    running_best = torch.cummax(f_vec, dim=0).values
    simple = f_star - running_best
    return SparseUCBResult(X=X, Y=Y, cumulative_regret=cum, simple_regret=simple,
                           per_step_time=per_step_time, betas=betas)
