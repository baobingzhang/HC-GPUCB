"""Local domain (L2 ball) operations for HC-GPUCB Phase 2."""

from __future__ import annotations

import torch
from torch import Tensor


def make_box_around(
    center: Tensor, radius: float, bounds: Tensor
) -> Tensor:
    """Return (2, d) box-bounds = [center - r, center + r] intersected with `bounds`.

    The L2 ball is approximated by its bounding box. This is what BoTorch's
    optimize_acqf accepts. The actual L2 constraint is enforced by post-projection
    of the optimum (acq_optimum_in_ball below).
    """
    lo = (center - radius).maximum(bounds[0])
    hi = (center + radius).minimum(bounds[1])
    return torch.stack([lo, hi])


def project_to_ball(x: Tensor, center: Tensor, radius: float) -> Tensor:
    """Project `x` (n, d) onto the L2 ball of given radius around `center`."""
    diff = x - center
    norm = diff.norm(dim=-1, keepdim=True)
    factor = torch.minimum(torch.ones_like(norm), radius / (norm + 1e-12))
    return center + diff * factor


def points_in_ball(X: Tensor, center: Tensor, radius: float) -> Tensor:
    """Return boolean mask of shape (n,): which rows of X are inside L2 ball."""
    return (X - center).norm(dim=-1) <= radius


def expand_radius_if_boundary(
    cand: Tensor, center: Tensor, radius: float, slack: float = 1e-3
) -> bool:
    """Return True iff candidate sits at the ball boundary (signaling we should
    consider expanding the local radius). `slack` is the distance tolerance."""
    return ((cand - center).norm(dim=-1) >= radius - slack).any().item()
