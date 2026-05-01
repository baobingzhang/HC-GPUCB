"""
HC-GPUCB: Hybrid Concentration-aware GP-UCB.

Phase 1 (t = 1..T_0): standard GP-UCB on full domain X.
Switch:               x_hat_star = argmax mu(.; X_{T_0}, y_{T_0});
                      X_local = ball(R; x_hat_star) cap X.
Phase 2 (t > T_0):    GP-UCB on X_local using only data in X_local.

Two switching modes:
  * "fixed":    T_0 = ceil(c_0 * sqrt(T))
  * "adaptive": switch at first t > t_min when last `window` points are
                rho-concentrated within `radius`.

Local radius schedule (from Theorem 1 of our paper):
  R(T_0) = c_R * T_0^(-1/4) * sqrt(beta_{T_0})
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
from torch import Tensor

from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.baselines.vanilla_ucb import beta_iwazaki
from src.concentration_test import is_concentrated
from src.local_domain import make_box_around, project_to_ball, points_in_ball


@dataclass
class HCGPUCBResult:
    X: Tensor
    Y: Tensor
    cumulative_regret: Tensor
    simple_regret: Tensor
    switch_step: int               # t at which Phase 2 began
    local_center: Tensor | None    # x_hat_star at switch
    local_radius: float | None
    per_step_time: list[float] = field(default_factory=list)  # wall-clock per BO iteration
    betas: list[float] = field(default_factory=list)


def run_hc_gpucb(
    objective: Callable[[Tensor], Tensor],
    bounds: Tensor,
    T: int,
    f_star: float,
    *,
    mode: Literal["fixed", "adaptive"] = "fixed",
    c_0: float = 3.0,                # T_0 = ceil(c_0 * sqrt(T)); 3 gives a sensible Phase-1 length even for small T
    c_R: float = 2.0,
    min_phase1: int = 20,            # never let Phase-1 be shorter than this (small-T safeguard)
    radius_slack: float = 0.5,       # multiplicative widening: R *= (1 + radius_slack)
    # adaptive trigger params
    window: int = 10,
    rho: float = 0.6,
    trigger_radius_frac: float = 0.20,  # radius (relative to domain size) to test concentration
    t_min_frac: float = 0.25,        # earliest fraction of T at which to allow trigger
    # common
    n_init: int = 5,
    delta: float = 0.05,
    noise_std: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    num_restarts: int = 8,
    raw_samples: int = 256,
) -> HCGPUCBResult:
    torch.manual_seed(seed)
    device = torch.device(device)
    bounds = bounds.to(device=device, dtype=dtype)
    d = bounds.shape[-1]
    domain_diam = (bounds[1] - bounds[0]).norm().item()
    lo, hi = bounds[0], bounds[1]

    # ---- random init ----
    X_init = lo + (hi - lo) * torch.rand(n_init, d, device=device, dtype=dtype)
    Y_init = objective(X_init).unsqueeze(-1)
    if noise_std > 0:
        Y_init = Y_init + noise_std * torch.randn_like(Y_init)
    X = X_init.clone()
    Y = Y_init.clone()
    f_vals: list[float] = list(objective(X_init).tolist())
    betas: list[float] = []
    per_step_time: list[float] = []

    # ---- precompute fixed-mode switch step ----
    # Phase-1 length must be long enough for the post-Phase-1 incumbent to be a
    # reliable estimate of x*. We require:
    #   1) at least min_phase1 BO iterations (data-density safeguard for low T);
    #   2) at least 5*d iterations (per-dimension exploration);
    #   3) the canonical c_0 sqrt(T) target.
    # Then cap so Phase 2 still has at least 30% of the budget.
    if mode == "fixed":
        T_0_canonical = int(math.ceil(c_0 * math.sqrt(T)))
        T_0 = max(n_init + min_phase1, n_init + 5 * d, T_0_canonical)
        T_0 = min(T_0, max(n_init + min_phase1, int(0.7 * T)))
    else:
        T_0 = T  # placeholder; will be set when trigger fires

    switched = False
    local_center: Tensor | None = None
    local_radius: float | None = None
    local_box: Tensor | None = None
    actual_switch_step = T

    for t in range(n_init + 1, T + 1):
        t_start = time.time()

        # decide active domain & active data
        if not switched and t >= T_0 and mode == "fixed":
            switched = True
        if not switched and mode == "adaptive" and t >= max(int(t_min_frac * T), n_init + 2):
            # cheap concentration test using current incumbent
            current_best_idx = int(torch.argmax(Y.squeeze(-1)).item())
            x_hat = X[current_best_idx]
            tr_radius = trigger_radius_frac * domain_diam
            if is_concentrated(X, x_hat, window=window, rho=rho, radius=tr_radius):
                switched = True

        if switched and local_center is None:
            # Use the argmax of observed Y as the local center. This avoids the
            # extra GP fit + acquisition optimization that posterior-mean argmax
            # would require, at the cost of slightly noisier centering. In the
            # noiseless setting the two coincide; in noisy settings the radius
            # slack `alpha` (default 0.5) absorbs the imprecision.
            current_best_idx = int(torch.argmax(Y.squeeze(-1)).item())
            local_center = X[current_best_idx].detach().clone()

            beta_at_switch = beta_iwazaki(t, d, delta)
            r_theory = c_R * (t ** (-0.25)) * math.sqrt(beta_at_switch)
            r_widen = r_theory * (1.0 + radius_slack)
            r = max(r_widen, 0.05 * domain_diam)
            r = min(r, 0.5 * domain_diam)
            local_radius = float(r)
            local_box = make_box_around(local_center, local_radius, bounds)
            actual_switch_step = t

        # ---- choose data subset ----
        if switched:
            mask = points_in_ball(X, local_center, local_radius)
            if mask.sum() < 2:
                # fallback to last observed point + nearest neighbor
                topk = max(2, int(min(5, X.shape[0])))
                dists = (X - local_center).norm(dim=-1)
                _, idx = torch.topk(-dists, topk)
                X_use = X[idx]
                Y_use = Y[idx]
            else:
                X_use = X[mask]
                Y_use = Y[mask]
            active_bounds = local_box
        else:
            X_use = X
            Y_use = Y
            active_bounds = bounds

        # ---- fit GP & optimize UCB on active domain ----
        gp = SingleTaskGP(X_use, Y_use).to(device=device, dtype=dtype)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        beta_t = beta_iwazaki(t, d, delta)
        betas.append(beta_t)
        ucb = UpperConfidenceBound(gp, beta=beta_t)

        cand, _ = optimize_acqf(
            ucb,
            bounds=active_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        if switched and local_center is not None:
            cand = project_to_ball(cand, local_center, local_radius)

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

    return HCGPUCBResult(
        X=X,
        Y=Y,
        cumulative_regret=cum,
        simple_regret=simple,
        switch_step=actual_switch_step,
        local_center=local_center,
        local_radius=local_radius,
        per_step_time=per_step_time,
        betas=betas,
    )
