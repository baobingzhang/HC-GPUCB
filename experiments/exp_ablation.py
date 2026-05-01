"""
Ablation: HC-GPUCB sensitivity to (c_0, c_R) on Hartmann-3 at T=200.

c_0 controls Phase-1 length: T_0 = ceil(c_0 * sqrt(T))
c_R controls local radius : R = c_R * T_0^{-1/4} * sqrt(beta)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="botorch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="botorch")

from src.hc_gpucb import run_hc_gpucb
from src.utils.test_functions import get_function


C0_GRID = [1.0, 3.0, 6.0]
CR_GRID = [1.0, 2.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", type=str, default="hartmann3")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "results" / "raw" / "exp_ablation"))
    ap.add_argument("--array-id", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # decode array id
    array_id = args.array_id
    if array_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if array_id is None:
        # run all configs in single process
        configs = [(c0, cR, s) for c0 in C0_GRID for cR in CR_GRID for s in range(3)]
    else:
        # decode (c0_idx, cR_idx, seed)
        seeds_per = 3
        nc = len(C0_GRID)
        nr = len(CR_GRID)
        n_per_c = nr * seeds_per
        c0_idx = array_id // n_per_c
        rem = array_id % n_per_c
        cR_idx = rem // seeds_per
        seed = rem % seeds_per
        configs = [(C0_GRID[c0_idx], CR_GRID[cR_idx], seed)]

    fn = get_function(args.func)
    bounds = fn.bounds()
    f_star = fn.f_star

    def obj(x):
        return fn(x.to(torch.double))

    for (c0, cR, seed) in configs:
        print(f"[abl] c0={c0} cR={cR} seed={seed} T={args.T}")
        t0 = time.time()
        res = run_hc_gpucb(
            obj, bounds, T=args.T, f_star=f_star,
            mode="fixed", c_0=c0, c_R=cR, seed=seed,
        )
        elapsed = time.time() - t0

        out_dir = out_root / f"c0={c0}_cR={cR}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"seed{seed}.pkl"
        payload = {
            "func": args.func,
            "method": "hc_fixed",
            "c0": c0, "cR": cR, "seed": seed, "T": args.T,
            "f_star": f_star,
            "X": res.X.cpu().numpy(),
            "Y": res.Y.cpu().numpy().squeeze(-1),
            "cumulative_regret": res.cumulative_regret.cpu().numpy(),
            "simple_regret": res.simple_regret.cpu().numpy(),
            "elapsed_total": elapsed,
            "switch_step": res.switch_step,
            "local_radius": res.local_radius,
        }
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  saved -> {out_path}  ({elapsed:.1f}s)  switch={res.switch_step}")


if __name__ == "__main__":
    main()
