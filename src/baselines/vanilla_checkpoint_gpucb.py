"""
Vanilla GP-UCB with logarithmic-checkpoint hyperparameter refits.

This baseline isolates the contribution of *deferred hyperparameter optimization*
from the contribution of *incremental Cholesky amortization* in our Rank-1 GP-UCB:

  - Vanilla GP-UCB:               full Cholesky + full hyperfit per step.
  - Vanilla-checkpoint GP-UCB:    full Cholesky per step, hyperfit only at log checkpoints.
  - Rank-1 GP-UCB:                hyperfit at log checkpoints + incremental conditioning.

Comparing Vanilla-fullfit vs. Vanilla-checkpoint reveals how much of Rank-1's wall-clock
saving comes from skipping fit_gpytorch_mll, vs. how much comes from BoTorch's lazy-tensor
caching in condition_on_observations.

Implementation: at each step we always rebuild SingleTaskGP from scratch (full Cholesky),
but we only run fit_gpytorch_mll at t in {2, 4, 8, 16, ...}; otherwise we copy the
hyperparameter state_dict from the previous step's GP.
"""

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
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.baselines.vanilla_ucb import beta_iwazaki


@dataclass
class CheckpointResult:
    X: Tensor
    Y: Tensor
    cumulative_regret: Tensor
    simple_regret: Tensor
    per_step_time: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)
    n_full_fits: int = 0


def _is_checkpoint(t: int) -> bool:
    return (t & (t - 1) == 0) and t >= 2


def run_vanilla_checkpoint_gpucb(
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
    num_restarts: int = 10,
    raw_samples: int = 256,
) -> CheckpointResult:
    """Vanilla GP-UCB with hyperparameter refits restricted to logarithmic checkpoints."""
    torch.manual_seed(seed)
    device = torch.device(device)
    bounds = bounds.to(device=device, dtype=dtype)
    d = bounds.shape[-1]
    lo, hi = bounds[0], bounds[1]

    X_init = lo + (hi - lo) * torch.rand(n_init, d, device=device, dtype=dtype)
    Y_init = objective(X_init).unsqueeze(-1)
    if noise_std > 0:
        Y_init = Y_init + noise_std * torch.randn_like(Y_init)
    X = X_init.clone()
    Y = Y_init.clone()
    f_vals: list[float] = list(objective(X_init).tolist())
    per_step_time: list[float] = []
    betas: list[float] = []
    n_full = 0

    # initial fit: always full
    gp = SingleTaskGP(X, Y).to(device=device, dtype=dtype)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    n_full += 1
    next_checkpoint = 2 * X.shape[0]
    cached_state = gp.state_dict()

    for t in range(n_init + 1, T + 1):
        t_start = time.time()

        # rebuild fresh SingleTaskGP every step (full Cholesky), but only refit hyperparams
        # at log checkpoints; otherwise inherit from cached_state.
        gp = SingleTaskGP(X, Y).to(device=device, dtype=dtype)
        do_refit = X.shape[0] >= next_checkpoint
        if do_refit:
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            n_full += 1
            next_checkpoint = 2 * X.shape[0]
            cached_state = gp.state_dict()
        else:
            try:
                gp.load_state_dict(cached_state, strict=False)
            except Exception:
                # mismatch (e.g. different model size attribute) — fall back to fresh fit
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                n_full += 1
                cached_state = gp.state_dict()

        beta_t = beta_iwazaki(t, d, delta)
        betas.append(beta_t)
        ucb = UpperConfidenceBound(gp, beta=beta_t)
        cand, _ = optimize_acqf(
            ucb, bounds=bounds, q=1,
            num_restarts=num_restarts, raw_samples=raw_samples,
        )
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

    return CheckpointResult(
        X=X, Y=Y,
        cumulative_regret=cum,
        simple_regret=simple,
        per_step_time=per_step_time,
        betas=betas,
        n_full_fits=n_full,
    )
