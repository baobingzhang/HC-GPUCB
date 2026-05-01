"""
HC-GPUCB-Rank1: HC-GPUCB with rank-1 inference (logarithmic-checkpoint hyperparameter
refits + BoTorch incremental conditioning) integrated into Phase 2.

Phase 1 (t = 1..T_0): standard GP-UCB on full domain X (full-Cholesky each step).
Switch:               x_hat_star = argmax y_i; X_local = ball(R; x_hat_star) cap X.
Phase 2 (t > T_0):    GP-UCB on X_local with rank-1 amortization:
                        - hyperparameter refit only at logarithmic checkpoints
                        - condition_on_observations on local-ball points between checkpoints
                        - chain_cap=128 to bound memory growth

This combines the structural advantage of HC-GPUCB (smaller local data + smaller search
domain) with the engineering advantage of Rank-1 (per-step O(L^2) instead of O(L^3)).
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
class HCRank1Result:
    X: Tensor
    Y: Tensor
    cumulative_regret: Tensor
    simple_regret: Tensor
    switch_step: int
    local_center: Tensor | None
    local_radius: float | None
    per_step_time: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)
    n_full_fits: int = 0
    n_rank1_steps: int = 0


def _is_log_checkpoint(local_t: int) -> bool:
    """Refit hyperparameters at local_t = 2, 4, 8, 16, ... (powers of 2 within Phase 2)."""
    return (local_t & (local_t - 1) == 0) and local_t >= 2


def run_hc_gpucb_rank1(
    objective: Callable[[Tensor], Tensor],
    bounds: Tensor,
    T: int,
    f_star: float,
    *,
    mode: Literal["fixed", "adaptive"] = "fixed",
    c_0: float = 3.0,
    c_R: float = 2.0,
    min_phase1: int = 20,
    radius_slack: float = 0.5,
    window: int = 10,
    rho: float = 0.6,
    trigger_radius_frac: float = 0.20,
    t_min_frac: float = 0.25,
    n_init: int = 5,
    delta: float = 0.05,
    noise_std: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
    num_restarts: int = 8,
    raw_samples: int = 256,
    chain_cap: int = 128,
) -> HCRank1Result:
    torch.manual_seed(seed)
    device = torch.device(device)
    bounds = bounds.to(device=device, dtype=dtype)
    d = bounds.shape[-1]
    domain_diam = (bounds[1] - bounds[0]).norm().item()
    lo, hi = bounds[0], bounds[1]

    # init
    X_init = lo + (hi - lo) * torch.rand(n_init, d, device=device, dtype=dtype)
    Y_init = objective(X_init).unsqueeze(-1)
    if noise_std > 0:
        Y_init = Y_init + noise_std * torch.randn_like(Y_init)
    X = X_init.clone()
    Y = Y_init.clone()
    f_vals: list[float] = list(objective(X_init).tolist())
    betas: list[float] = []
    per_step_time: list[float] = []
    n_full = 0
    n_rank1 = 0

    if mode == "fixed":
        T_0_canonical = int(math.ceil(c_0 * math.sqrt(T)))
        T_0 = max(n_init + min_phase1, n_init + 5 * d, T_0_canonical)
        T_0 = min(T_0, max(n_init + min_phase1, int(0.7 * T)))
    else:
        T_0 = T

    switched = False
    local_center: Tensor | None = None
    local_radius: float | None = None
    local_box: Tensor | None = None
    actual_switch_step = T

    # Rank-1 state for Phase 2 (initialised at switch).
    gp_p2: SingleTaskGP | None = None
    p2_X_size: int = 0           # last fitted size of X_use during Phase 2
    chain_len: int = 0
    local_t_p2: int = 0          # iterations elapsed since switch (for log checkpoints)

    for t in range(n_init + 1, T + 1):
        t_start = time.time()

        # decide if we switch at this step
        if not switched and t >= T_0 and mode == "fixed":
            switched = True
        if not switched and mode == "adaptive" and t >= max(int(t_min_frac * T), n_init + 2):
            current_best_idx = int(torch.argmax(Y.squeeze(-1)).item())
            x_hat = X[current_best_idx]
            tr_radius = trigger_radius_frac * domain_diam
            if is_concentrated(X, x_hat, window=window, rho=rho, radius=tr_radius):
                switched = True

        if switched and local_center is None:
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

        # choose data subset
        if switched:
            mask = points_in_ball(X, local_center, local_radius)
            if mask.sum() < 2:
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

        # ---- fit GP ----
        if not switched:
            # Phase 1: full Cholesky each step (faithful Iwazaki). T_0 ~ sqrt(T) is small.
            gp = SingleTaskGP(X_use, Y_use).to(device=device, dtype=dtype)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            n_full += 1
        else:
            # Phase 2: rank-1 amortization.
            local_t_p2 += 1
            current_size = X_use.shape[0]

            # decide whether to do a full refit:
            # - first iteration after switch (initialise gp_p2)
            # - hit a log checkpoint within Phase 2
            # - chain_len exceeded chain_cap
            # - X_use shrunk (mask removed older points -- shouldn't happen but defensive)
            do_refit = (
                gp_p2 is None
                or _is_log_checkpoint(local_t_p2)
                or chain_len >= chain_cap
                or current_size < p2_X_size  # data identity changed
            )

            if do_refit:
                gp = SingleTaskGP(X_use, Y_use).to(device=device, dtype=dtype)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                gp_p2 = gp
                p2_X_size = current_size
                chain_len = 0
                n_full += 1
            else:
                # Incremental: condition on the new local point(s) since last refit.
                # We may have added several points since last fit; condition on the
                # ones that lie in the ball (i.e. the last `current_size - p2_X_size` points
                # in X_use).
                added = current_size - p2_X_size
                if added > 0:
                    new_X = X_use[-added:]
                    new_Y = Y_use[-added:]
                    try:
                        gp = gp_p2.condition_on_observations(new_X, new_Y)
                        gp_p2 = gp
                        p2_X_size = current_size
                        chain_len += added
                        n_rank1 += 1
                    except Exception:
                        # fallback: full refit
                        gp = SingleTaskGP(X_use, Y_use).to(device=device, dtype=dtype)
                        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                        fit_gpytorch_mll(mll)
                        gp_p2 = gp
                        p2_X_size = current_size
                        chain_len = 0
                        n_full += 1
                else:
                    # No new local point this step (last candidate fell outside ball
                    # via projection). Reuse previous gp_p2.
                    gp = gp_p2

        beta_t = beta_iwazaki(t, d, delta)
        betas.append(beta_t)
        ucb = UpperConfidenceBound(gp, beta=beta_t)
        cand, _ = optimize_acqf(
            ucb, bounds=active_bounds, q=1,
            num_restarts=num_restarts, raw_samples=raw_samples,
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

    return HCRank1Result(
        X=X, Y=Y,
        cumulative_regret=cum,
        simple_regret=simple,
        switch_step=actual_switch_step,
        local_center=local_center,
        local_radius=local_radius,
        per_step_time=per_step_time,
        betas=betas,
        n_full_fits=n_full,
        n_rank1_steps=n_rank1,
    )
