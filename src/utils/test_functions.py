"""
Standard BO benchmark functions, formulated as MAXIMIZATION problems
(we negate canonical minimization functions so that f_star is the maximum).

All functions take a Tensor of shape (n, d) and return Tensor of shape (n,).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Branin (d=2). Canonical minimum is 0.397887; we negate so f_star=-0.397887.
# Domain: x1 in [-5, 10], x2 in [0, 15].
# -----------------------------------------------------------------------------
class Branin:
    dim = 2
    f_star = -0.397887  # max of -Branin

    def bounds(self, dtype=torch.double, device="cpu") -> Tensor:
        return torch.tensor([[-5.0, 0.0], [10.0, 15.0]], dtype=dtype, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        a = 1.0
        b = 5.1 / (4.0 * math.pi**2)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)
        x1 = x[..., 0]
        x2 = x[..., 1]
        val = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * torch.cos(x1) + s
        return -val  # we maximize


# -----------------------------------------------------------------------------
# Hartmann-3 (d=3). Canonical minimum is -3.86278; we negate so f_star=3.86278.
# Domain: [0, 1]^3.
# -----------------------------------------------------------------------------
class Hartmann3:
    dim = 3
    f_star = 3.86278

    _alpha = (1.0, 1.2, 3.0, 3.2)
    _A = (
        (3.0, 10.0, 30.0),
        (0.1, 10.0, 35.0),
        (3.0, 10.0, 30.0),
        (0.1, 10.0, 35.0),
    )
    _P = (
        (0.3689, 0.1170, 0.2673),
        (0.4699, 0.4387, 0.7470),
        (0.1091, 0.8732, 0.5547),
        (0.03815, 0.5743, 0.8828),
    )

    def bounds(self, dtype=torch.double, device="cpu") -> Tensor:
        return torch.stack([
            torch.zeros(3, dtype=dtype, device=device),
            torch.ones(3, dtype=dtype, device=device),
        ])

    def __call__(self, x: Tensor) -> Tensor:
        alpha = torch.tensor(self._alpha, dtype=x.dtype, device=x.device)
        A = torch.tensor(self._A, dtype=x.dtype, device=x.device)
        P = torch.tensor(self._P, dtype=x.dtype, device=x.device)
        # x: (n, 3). Compute -sum_i alpha_i exp(-sum_j A_ij (x_j - P_ij)^2)
        # broadcast: (n, 1, 3) - (1, 4, 3) = (n, 4, 3)
        diff = x.unsqueeze(-2) - P.unsqueeze(0)
        inner = -(A.unsqueeze(0) * diff**2).sum(dim=-1)
        out = -(alpha.unsqueeze(0) * inner.exp()).sum(dim=-1)
        return -out  # negate to maximize


# -----------------------------------------------------------------------------
# Hartmann-6 (d=6). Canonical min -3.32237; f_star=3.32237 after negation.
# Domain: [0, 1]^6.
# -----------------------------------------------------------------------------
class Hartmann6:
    dim = 6
    f_star = 3.32237

    _alpha = (1.0, 1.2, 3.0, 3.2)
    _A = (
        (10.0, 3.0, 17.0, 3.5, 1.7, 8.0),
        (0.05, 10.0, 17.0, 0.1, 8.0, 14.0),
        (3.0, 3.5, 1.7, 10.0, 17.0, 8.0),
        (17.0, 8.0, 0.05, 10.0, 0.1, 14.0),
    )
    _P = (
        (0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886),
        (0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991),
        (0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650),
        (0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381),
    )

    def bounds(self, dtype=torch.double, device="cpu") -> Tensor:
        return torch.stack([
            torch.zeros(6, dtype=dtype, device=device),
            torch.ones(6, dtype=dtype, device=device),
        ])

    def __call__(self, x: Tensor) -> Tensor:
        alpha = torch.tensor(self._alpha, dtype=x.dtype, device=x.device)
        A = torch.tensor(self._A, dtype=x.dtype, device=x.device)
        P = torch.tensor(self._P, dtype=x.dtype, device=x.device)
        diff = x.unsqueeze(-2) - P.unsqueeze(0)
        inner = -(A.unsqueeze(0) * diff**2).sum(dim=-1)
        out = -(alpha.unsqueeze(0) * inner.exp()).sum(dim=-1)
        return -out


# -----------------------------------------------------------------------------
# Ackley (any d). Canonical min 0; f_star=0 after negation.
# Domain: [-32.768, 32.768]^d, but we use [-5, 5]^d for tractability in BO.
# -----------------------------------------------------------------------------
class Ackley:
    f_star = 0.0

    def __init__(self, dim: int = 5, lo: float = -5.0, hi: float = 5.0):
        self.dim = dim
        self._lo = lo
        self._hi = hi

    def bounds(self, dtype=torch.double, device="cpu") -> Tensor:
        return torch.stack([
            torch.full((self.dim,), self._lo, dtype=dtype, device=device),
            torch.full((self.dim,), self._hi, dtype=dtype, device=device),
        ])

    def __call__(self, x: Tensor) -> Tensor:
        a, b, c = 20.0, 0.2, 2 * math.pi
        d = x.shape[-1]
        s1 = (x**2).sum(dim=-1) / d
        s2 = torch.cos(c * x).sum(dim=-1) / d
        val = -a * torch.exp(-b * torch.sqrt(s1)) - torch.exp(s2) + a + math.e
        return -val


class Levy:
    """Levy function on [-10, 10]^d. Global maximum is f_star = 0 at x = (1, 1, ..., 1).

    Standard formula minimizes a sum-of-sines; we negate so that BO's argmax
    aligns with the function's argmin at (1, ..., 1).
    """
    f_star = 0.0

    def __init__(self, dim: int = 10, lo: float = -10.0, hi: float = 10.0):
        self.dim = dim
        self._lo = lo
        self._hi = hi

    def bounds(self, dtype=torch.double, device="cpu") -> Tensor:
        return torch.stack([
            torch.full((self.dim,), self._lo, dtype=dtype, device=device),
            torch.full((self.dim,), self._hi, dtype=dtype, device=device),
        ])

    def __call__(self, x: Tensor) -> Tensor:
        w = 1.0 + (x - 1.0) / 4.0
        term1 = torch.sin(math.pi * w[..., 0]) ** 2
        # middle sum: i = 0..d-2
        wi = w[..., :-1]
        s_mid = ((wi - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * wi + 1.0) ** 2)).sum(dim=-1)
        wd = w[..., -1]
        term3 = (wd - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * wd) ** 2)
        val = term1 + s_mid + term3
        return -val  # negate so we maximize


REGISTRY = {
    "branin": Branin,
    "hartmann3": Hartmann3,
    "hartmann6": Hartmann6,
    "ackley5": lambda: Ackley(dim=5),
    "levy10": lambda: Levy(dim=10),
    "levy20": lambda: Levy(dim=20),
}


def get_function(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown test function {name!r}; choices: {list(REGISTRY)}")
    return REGISTRY[name]()
