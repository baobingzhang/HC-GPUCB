# HC-GPUCB

**Hybrid Concentration-aware Gaussian Process Upper Confidence Bound** for Bayesian optimization with tight regret guarantees.

> **Accepted** in *IEEE Transactions on Artificial Intelligence* (2026).

Reference implementation for the paper *"HC-GPUCB: Hybrid Concentration-Aware Gaussian Process Upper Confidence Bound for Bayesian Optimization with Tight Regret Guarantees"*, Baobing Zhang and Wanxin Sui, *IEEE Transactions on Artificial Intelligence*, 2026.

---

## What's in this repo

Two algorithms and the engineering ingredients behind them:

| Algorithm | File | Theory | Wall-clock |
|---|---|---|---|
| **HC-GPUCB** (fixed schedule) | `src/hc_gpucb.py` | $\widetilde O(\sqrt T)$ Bayesian regret (Theorem 2) | constant-factor over vanilla |
| **HC-GPUCB** (adaptive trigger) | `src/hc_gpucb.py` | same | small wall-clock benefit; degenerates to vanilla on rare-event paths |
| **Rank-1 GP-UCB** | `src/baselines/rank1_gp_ucb.py` | Iwazaki regret bound applies directly | **3.5–7.1×** at $T{=}500$, **16.3×** at $T{=}1000$ on Hartmann-3 |
| **HC-GPUCB-Rank1** (combined) | `src/hc_gpucb_rank1.py` | $\widetilde O(\sqrt T)$ regret + rank-1 inference | scales to $T{=}3000$ where vanilla GP-UCB is computationally infeasible |

Plus the baselines used in the paper (vanilla GP-UCB, EI, Thompson Sampling, Sparse-GP-UCB, Vanilla-Checkpoint GP-UCB).

---

## Repo layout

```
src/
  hc_gpucb.py             # Algorithm 1 (fixed + adaptive)
  hc_gpucb_rank1.py       # HC-GPUCB with rank-1 Phase-2
  local_domain.py         # local-ball construction, floored-regime logic
  concentration_test.py   # adaptive trigger (Eq. 7)
  baselines/
    rank1_gp_ucb.py             # logarithmic-checkpoint hyperfit + incremental conditioning
    vanilla_checkpoint_gpucb.py # decoupled ablation: hyperfit-only
    vanilla_ucb.py              # standard GP-UCB with Iwazaki β_t schedule
    sparse_gp_ucb.py            # inducing-point variant
    ei.py / thompson_sampling.py / random_search.py
  utils/
    test_functions.py     # Branin / Hartmann-3 / Hartmann-6 / Ackley / Levy-d
experiments/
  exp1_main.py            # entry point: --func --method --T --seed
  analyze_exp1.py         # generates main_table_T500.tex + figures
  analyze_T2k3k.py        # T=2000/3000 results table
  analyze_phaseD.py       # decoupled, hc-rank1, noise, highdim tables
  analyze_compute_fair.py # cumulative regret + iters/1000s
  stat_tests.py           # Wilcoxon + TOST equivalence
scripts/
  run_exp_T500.slurm              # main T=500 grid (4 funcs × 8 methods × 10 seeds)
  run_speedup_vs_T.slurm          # exclusive single-node, controlled experiment
  run_hc_rank1.slurm              # HC-GPUCB-Rank1 at T={500,1000,2000,3000}
  run_T2000_T3000.slurm           # large-budget runs
```

---

## Quick start

```bash
# 1. Environment
conda create -n hcgpucb python=3.10
conda activate hcgpucb
pip install -r requirements.txt

# 2. Run a single experiment (T=500 on Hartmann-3 with HC-GPUCB-adaptive)
python -m experiments.exp1_main --func hartmann3 --method hc_adaptive --T 500 --seed 0

# 3. Reproduce a paper table
#    Step 3a: full T=500 grid (10 seeds × 4 funcs × 8 methods)
sbatch scripts/run_exp_T500.slurm
#    Step 3b: regenerate Table III + Figure 3
python experiments/analyze_exp1.py --raw results/raw/exp_T500 --tag T500
```

Each run writes a pickle to `results/raw/<exp_name>/<func>/<method>/seed<k>.pkl` containing `simple_regret`, `cumulative_regret`, `elapsed_total`, `X`, `Y`, `f_star`. The analysis scripts read these pickles and emit `paper/tables/*.tex` + `results/figures/*.png` consumed by the paper.

---

## Method-by-method headline numbers

(All on a shared CPU cluster, BoTorch 0.16.1 / PyTorch 2.5.1 / GPyTorch.)

**At $T = 500$ (Table III), regret-guaranteed family on Hartmann-3:**
- Vanilla GP-UCB: regret $0.001$, time $608.4$ s
- Rank-1 GP-UCB: regret $0.000$, time $103.4$ s — **5.9×** speedup
- HC-GPUCB-adaptive: regret $0.005$, time $303.9$ s

**At $T = 1000$ on Hartmann-3 (controlled, exclusive single-node, Figure 1):**
- Vanilla GP-UCB: $4678$ s
- Rank-1 GP-UCB: $289$ s — **16.3×** speedup

**At $T = 3000$ on Hartmann-6:**
- Vanilla GP-UCB: infeasible by extrapolation ($\sim 10.6$ h, $T^{1.83}$ fit)
- HC-GPUCB-Rank1 (adaptive): $89$ min/seed (Table IV)

---

## Citation

If you use this code or build on the ideas in your own work, please cite:

```bibtex
@article{zhang2026hcgpucb,
  author  = {Zhang, Baobing and Sui, Wanxin},
  title   = {{HC-GPUCB}: Hybrid Concentration-Aware {Gaussian} Process Upper Confidence Bound for {Bayesian} Optimization with Tight Regret Guarantees},
  journal = {IEEE Transactions on Artificial Intelligence},
  year    = {2026},
  note    = {Accepted; DOI to be added upon publication.}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
