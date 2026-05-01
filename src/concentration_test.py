"""Concentration trigger for adaptive Phase-1 -> Phase-2 switching.

Idea: at time t, look at the most recent Delta selections X_{t-Delta+1}, ..., X_t.
If at least rho fraction of them lie within radius `r` of the current incumbent
x_hat^*, then declare the algorithm has entered the concentration regime and
trigger Phase 2 switching.
"""

from __future__ import annotations

import torch
from torch import Tensor


def concentration_score(
    X_recent: Tensor, x_hat_star: Tensor, radius: float
) -> float:
    """Fraction of points in `X_recent` (k, d) that lie within `radius` of x_hat_star."""
    if X_recent.numel() == 0:
        return 0.0
    dists = (X_recent - x_hat_star).norm(dim=-1)
    inside = (dists <= radius).float().mean()
    return float(inside.item())


def is_concentrated(
    X_history: Tensor,
    x_hat_star: Tensor,
    *,
    window: int,
    rho: float,
    radius: float,
) -> bool:
    """Test if last `window` points have >= rho fraction inside radius around x_hat_star."""
    if X_history.shape[0] < window:
        return False
    recent = X_history[-window:]
    score = concentration_score(recent, x_hat_star, radius)
    return score >= rho
