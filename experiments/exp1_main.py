"""
Main experiment runner for HC-GPUCB.

Runs ONE (function, method, seed) configuration and pickles the result.
Designed to be array-job-friendly via env var SLURM_ARRAY_TASK_ID OR
direct CLI args --func --method --seed.

Output: results/raw/exp1/<func>/<method>/seed<seed>.pkl
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

# Suppress benign BoTorch warnings about domain scaling and lengthscale priors
warnings.filterwarnings("ignore", category=UserWarning, module="botorch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="botorch")

# add project root to sys.path so `import src...` works in batch jobs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.ei import run_ei
from src.baselines.random_search import run_random_search
from src.baselines.rank1_gp_ucb import run_rank1_gpucb
from src.baselines.sparse_gp_ucb import run_sparse_gp_ucb
from src.baselines.thompson_sampling import run_thompson_sampling
from src.baselines.turbo_lite import run_turbo_lite, run_turbo_m
from src.baselines.vanilla_ucb import run_vanilla_gpucb
from src.baselines.vanilla_checkpoint_gpucb import run_vanilla_checkpoint_gpucb
from src.hc_gpucb import run_hc_gpucb
from src.hc_gpucb_rank1 import run_hc_gpucb_rank1
from src.utils.test_functions import REGISTRY, get_function


METHODS = ["random", "gpucb", "vanilla_chk", "rank1_gpucb", "ei", "turbo", "turbo_m", "ts",
           "sparse_ucb", "hc_fixed", "hc_adaptive", "hc_fixed_rank1", "hc_adaptive_rank1"]
FUNCS = ["branin", "hartmann3", "hartmann6", "ackley5"]


def run_one(func_name: str, method: str, seed: int, T: int, out_root: Path,
            noise_std: float = 0.0) -> Path:
    print(f"[exp1] func={func_name} method={method} seed={seed} T={T} noise={noise_std}")
    fn = get_function(func_name)
    bounds = fn.bounds()
    f_star = fn.f_star

    # objective wrapped: ensure tensor input, return tensor
    def obj(x):
        return fn(x.to(torch.double))

    kw = dict(noise_std=noise_std)

    t0 = time.time()
    if method == "random":
        res = run_random_search(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "gpucb":
        res = run_vanilla_gpucb(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "rank1_gpucb":
        res = run_rank1_gpucb(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "vanilla_chk":
        res = run_vanilla_checkpoint_gpucb(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "hc_fixed_rank1":
        res = run_hc_gpucb_rank1(obj, bounds, T=T, f_star=f_star, mode="fixed", seed=seed, **kw)
    elif method == "hc_adaptive_rank1":
        res = run_hc_gpucb_rank1(obj, bounds, T=T, f_star=f_star, mode="adaptive", seed=seed, **kw)
    elif method == "ei":
        res = run_ei(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "turbo":
        res = run_turbo_lite(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "turbo_m":
        res = run_turbo_m(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "ts":
        res = run_thompson_sampling(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "sparse_ucb":
        res = run_sparse_gp_ucb(obj, bounds, T=T, f_star=f_star, seed=seed, **kw)
    elif method == "hc_fixed":
        res = run_hc_gpucb(obj, bounds, T=T, f_star=f_star, mode="fixed", seed=seed, **kw)
    elif method == "hc_adaptive":
        res = run_hc_gpucb(obj, bounds, T=T, f_star=f_star, mode="adaptive", seed=seed, **kw)
    else:
        raise ValueError(f"unknown method: {method}")

    elapsed = time.time() - t0

    out_dir = out_root / func_name / method
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{seed}.pkl"

    # serialize a lean dict (avoid pickling GPyTorch model objects)
    payload = {
        "func": func_name,
        "method": method,
        "seed": seed,
        "T": T,
        "f_star": f_star,
        "X": res.X.cpu().numpy(),
        "Y": res.Y.cpu().numpy().squeeze(-1),
        "cumulative_regret": res.cumulative_regret.cpu().numpy(),
        "simple_regret": res.simple_regret.cpu().numpy(),
        "elapsed_total": elapsed,
    }
    if hasattr(res, "per_step_time"):
        payload["per_step_time"] = list(res.per_step_time)
    if hasattr(res, "switch_step"):
        payload["switch_step"] = res.switch_step
    if hasattr(res, "local_radius"):
        payload["local_radius"] = res.local_radius
    if hasattr(res, "betas"):
        payload["betas"] = list(res.betas)
    if hasattr(res, "n_full_fits"):
        payload["n_full_fits"] = res.n_full_fits
    if hasattr(res, "n_rank1_steps"):
        payload["n_rank1_steps"] = res.n_rank1_steps

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[exp1] saved -> {out_path}  (elapsed={elapsed:.1f}s)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", type=str, default=None)
    ap.add_argument("--method", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--noise", type=float, default=0.0,
                    help="Observation noise std (passed as noise_std to method runners).")
    ap.add_argument("--out", type=str, default=str(PROJECT_ROOT / "results" / "raw" / "exp1"))
    ap.add_argument("--array-id", type=int, default=None,
                    help="If set, decode (func, method, seed) from this index. "
                         "Total = len(FUNCS) * len(METHODS) * n_seeds.")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Manual --func/--method/--seed override SLURM_ARRAY_TASK_ID. The implicit
    # env-var fallback is ONLY used when no manual triple is given.
    manual_triple = args.func is not None and args.method is not None and args.seed is not None
    array_id = args.array_id
    if array_id is None and not manual_triple and "SLURM_ARRAY_TASK_ID" in os.environ:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if array_id is not None and not manual_triple:
        # decode: array_id -> (func_idx, method_idx, seed)
        n_methods = len(METHODS)
        n_seeds = args.n_seeds
        n_per_func = n_methods * n_seeds
        func_idx = array_id // n_per_func
        rem = array_id % n_per_func
        method_idx = rem // n_seeds
        seed = rem % n_seeds
        func_name = FUNCS[func_idx]
        method = METHODS[method_idx]
        run_one(func_name, method, seed, args.T, out_root, noise_std=args.noise)
    else:
        if args.func is None or args.method is None or args.seed is None:
            raise SystemExit("--func, --method, --seed required (or set --array-id)")
        run_one(args.func, args.method, args.seed, args.T, out_root, noise_std=args.noise)


if __name__ == "__main__":
    main()
