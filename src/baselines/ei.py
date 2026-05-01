"""Expected Improvement (EI) baseline using BoTorch's qExpectedImprovement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


@dataclass
class EIResult:
    X: Tensor
    Y: Tensor
    cumulative_regret: Tensor
    simple_regret: Tensor
    per_step_time: list[float] = field(default_factory=list)


def run_ei(
    objective: Callable[[Tensor], Tensor],
    bounds: Tensor,
    T: int,
    f_star: float,
    *,
    n_init: int = 5,
    noise_std: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    num_restarts: int = 8,
    raw_samples: int = 256,
) -> EIResult:
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
    per_step_time = []

    for t in range(n_init + 1, T + 1):
        t_start = time.time()
        gp = SingleTaskGP(X, Y).to(device=device, dtype=dtype)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        ei = ExpectedImprovement(gp, best_f=Y.max().item())
        cand, _ = optimize_acqf(ei, bounds=bounds, q=1,
                                num_restarts=num_restarts,
                                raw_samples=raw_samples)
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
    return EIResult(X=X, Y=Y, cumulative_regret=cum, simple_regret=simple,
                    per_step_time=per_step_time)
