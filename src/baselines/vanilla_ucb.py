"""
Vanilla GP-UCB (Srinivas et al. 2010), our baseline.

Following Iwazaki 2025 (NeurIPS) Algorithm 1:
    x_t = argmax_{x in X} mu(x; X_{t-1}, y_{t-1}) + beta_t^{1/2} * sigma(x; X_{t-1})

beta_t schedule from Iwazaki Eq. (6):
    beta_t = 2 ln(2 t^2 pi^2 / (3 delta)) + 2 d ln(t^2 d b sqrt(ln(4 d a / delta)))
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


@dataclass
class GPUCBResult:
    X: Tensor                    # (T, d) all queried points
    Y: Tensor                    # (T, 1) all observations
    cumulative_regret: Tensor    # (T,) running sum of (f* - f(x_t))
    simple_regret: Tensor        # (T,) f* - max_{i<=t} f(x_i)
    per_step_time: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)


def beta_iwazaki(t: int, d: int, delta: float, a: float = 1.0, b: float = 1.0) -> float:
    """beta_t schedule from Iwazaki 2025, Eq. (6).

    a, b are constants from Lemma 1 (Lipschitz of sample path); default 1.0 is fine
    for synthetic stationary kernels on bounded domains.
    """
    term1 = 2.0 * math.log(2.0 * (t**2) * (math.pi**2) / (3.0 * delta))
    inner = math.sqrt(math.log(4.0 * d * a / delta))
    term2 = 2.0 * d * math.log((t**2) * d * b * inner)
    return max(term1 + term2, 1e-3)


def run_vanilla_gpucb(
    objective: Callable[[Tensor], Tensor],
    bounds: Tensor,           # (2, d), row 0 = lower, row 1 = upper
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
) -> GPUCBResult:
    """Run vanilla GP-UCB for T steps and return regret trajectory.

    Parameters
    ----------
    objective : f: (q, d) -> (q,) tensor; we will treat this as the noiseless f.
    bounds    : (2, d) lower/upper bounds of search domain.
    T         : total queries (initial + sequential).
    f_star    : known maximum of objective (for regret computation).
    n_init    : random Sobol-like initial design size before BO loop.
    delta     : confidence parameter for beta_t.
    noise_std : observation noise std (added to objective output).
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    bounds = bounds.to(device=device, dtype=dtype)
    d = bounds.shape[-1]

    # ---- random init ----
    lo, hi = bounds[0], bounds[1]
    X_init = lo + (hi - lo) * torch.rand(n_init, d, device=device, dtype=dtype)
    Y_init = objective(X_init).unsqueeze(-1)
    if noise_std > 0:
        Y_init = Y_init + noise_std * torch.randn_like(Y_init)

    X = X_init.clone()
    Y = Y_init.clone()

    f_vals = []  # noiseless f(x_t) sequence for regret
    f_init = objective(X_init)
    f_vals.extend(f_init.tolist())
    betas: list[float] = []
    per_step_time: list[float] = []

    # ---- sequential BO loop ----
    for t in range(n_init + 1, T + 1):
        t_start = time.time()
        # Fit GP
        gp = SingleTaskGP(X, Y).to(device=device, dtype=dtype)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # UCB with Iwazaki beta_t (note BoTorch UCB uses sqrt(beta) internally as 'beta')
        beta_t = beta_iwazaki(t, d, delta)
        betas.append(beta_t)
        ucb = UpperConfidenceBound(gp, beta=beta_t)

        cand, _ = optimize_acqf(
            ucb,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        f_t = objective(cand).item()
        y_t = f_t + (noise_std * torch.randn(1).item() if noise_std > 0 else 0.0)
        X = torch.cat([X, cand], dim=0)
        Y = torch.cat([Y, torch.tensor([[y_t]], device=device, dtype=dtype)], dim=0)
        f_vals.append(f_t)
        per_step_time.append(time.time() - t_start)

    # ---- regret ----
    f_vec = torch.tensor(f_vals, device=device, dtype=dtype)
    inst_regret = f_star - f_vec
    cum = torch.cumsum(inst_regret, dim=0)
    running_best = torch.cummax(f_vec, dim=0).values
    simple = f_star - running_best

    return GPUCBResult(
        X=X,
        Y=Y,
        cumulative_regret=cum,
        simple_regret=simple,
        per_step_time=per_step_time,
        betas=betas,
    )
