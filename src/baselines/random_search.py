"""Random search baseline (uniform sampling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class RandomResult:
    X: Tensor
    Y: Tensor
    cumulative_regret: Tensor
    simple_regret: Tensor


def run_random_search(
    objective: Callable[[Tensor], Tensor],
    bounds: Tensor,
    T: int,
    f_star: float,
    *,
    seed: int = 0,
    noise_std: float = 0.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.double,
) -> RandomResult:
    torch.manual_seed(seed)
    bounds = bounds.to(device=device, dtype=dtype)
    d = bounds.shape[-1]
    lo, hi = bounds[0], bounds[1]
    X = lo + (hi - lo) * torch.rand(T, d, device=device, dtype=dtype)
    f_vals = objective(X)  # noiseless f for regret computation
    Y_noisy = f_vals + (noise_std * torch.randn_like(f_vals) if noise_std > 0 else 0.0)
    Y = Y_noisy.unsqueeze(-1)
    inst_regret = f_star - f_vals
    cum = torch.cumsum(inst_regret, dim=0)
    running_best = torch.cummax(f_vals, dim=0).values
    simple = f_star - running_best
    return RandomResult(X=X, Y=Y, cumulative_regret=cum, simple_regret=simple)
