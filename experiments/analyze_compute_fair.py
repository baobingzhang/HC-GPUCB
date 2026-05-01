"""
Compute-fair diagnostic table for §V item 4.

For each method on each function at T=500:
  - Cumulative regret R_T = Σ instantaneous regret (the metric Iwazaki's
    Theorem 2 bounds directly).
  - Wall-clock per query (s/iter) — implementation efficiency.
  - "Equivalent iterations under fixed 1000-s budget" — a Pareto reframing:
    given a fixed compute budget, how many BO iterations can each method
    afford?

Output: paper/tables/compute_fair_T500.tex
"""

from __future__ import annotations
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TBL = PROJECT_ROOT / "paper" / "tables"
TBL.mkdir(parents=True, exist_ok=True)
RAW = PROJECT_ROOT / "results" / "raw" / "exp_T500"

GUARANTEED_FAMILY = {"gpucb", "rank1_gpucb", "sparse_ucb", "hc_fixed", "hc_adaptive"}
TABLE_ORDER = ["gpucb", "rank1_gpucb", "sparse_ucb", "hc_fixed", "hc_adaptive",
               "random", "ei", "ts"]
METHOD_LABEL = {
    "random": "Random", "gpucb": r"GP-UCB (Iwazaki '25)",
    "rank1_gpucb": "Rank-1 GP-UCB", "ei": "EI",
    "ts": "Thompson Sampling", "sparse_ucb": "Sparse GP-UCB",
    "hc_fixed": "HC-GPUCB (fixed)", "hc_adaptive": "HC-GPUCB (adaptive)",
}
FUNC_LABEL = {"branin": "Branin", "hartmann3": "Hartmann-3",
              "hartmann6": "Hartmann-6", "ackley5": "Ackley-5"}
FUNC_ORDER = ["branin", "hartmann3", "hartmann6", "ackley5"]
FIXED_BUDGET_S = 1000  # wall-clock cap for the equivalent-iterations column


def load(d: Path):
    if not d.exists():
        return []
    runs = []
    for p in sorted(d.glob("seed*.pkl")):
        with open(p, "rb") as f:
            runs.append(pickle.load(f))
    return runs


def main():
    data = defaultdict(dict)
    for f in FUNC_ORDER:
        for m in TABLE_ORDER:
            runs = load(RAW / f / m)
            if runs:
                data[f][m] = runs

    # We aggregate cumulative regret and time, then derive equivalent-iterations.
    cum = {}; tpiq = {}
    for f in FUNC_ORDER:
        for m in TABLE_ORDER:
            runs = data.get(f, {}).get(m, [])
            if not runs:
                continue
            cs = np.array([r["cumulative_regret"][-1] for r in runs])
            ts = np.array([r["elapsed_total"] for r in runs])
            # T is the budget (= len(simple_regret)) which should be 500
            T_iters = float(np.mean([len(r["simple_regret"]) for r in runs]))
            cum[(f, m)] = (float(cs.mean()), float(cs.std(ddof=1) / np.sqrt(len(cs))))
            tpiq[(f, m)] = float(ts.mean()) / T_iters  # seconds per iter

    lines = [r"\begin{tabular}{l|cc|cc|cc|cc}", r"\toprule"]
    head = " & ".join([f"\\multicolumn{{2}}{{c|}}{{{FUNC_LABEL[f]}}}" for f in FUNC_ORDER])
    lines.append(f"Method ($T{{=}}500$) & {head} \\\\")
    lines.append(" & " + " & ".join([r"$R_T$ & iters/$1$ks"] * len(FUNC_ORDER)) + r" \\")
    lines.append(r"\midrule")
    n_cols = 1 + 2 * len(FUNC_ORDER)
    lines.append(r"\multicolumn{" + str(n_cols) +
                 r"}{l}{\textit{Regret-guaranteed family}} \\")
    last_g = True
    for m in TABLE_ORDER:
        if not any((f, m) in cum for f in FUNC_ORDER):
            continue
        is_g = m in GUARANTEED_FAMILY
        if last_g and not is_g:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(n_cols) +
                         r"}{l}{\textit{Heuristic-reference family}} \\")
        last_g = is_g
        cells = [METHOD_LABEL[m]]
        for f in FUNC_ORDER:
            if (f, m) not in cum:
                cells.extend(["--", "--"]); continue
            cm, cs = cum[(f, m)]
            iters_per_1ks = FIXED_BUDGET_S / tpiq[(f, m)] if tpiq[(f, m)] > 0 else float("inf")
            cells.append(f"${cm:.0f}{{\\scriptstyle\\,\\pm\\,}}{cs:.0f}$")
            if iters_per_1ks == float("inf") or iters_per_1ks > 1e6:
                cells.append(r"$>10^6$")
            elif iters_per_1ks >= 1e4:
                cells.append(f"${iters_per_1ks/1e3:.0f}$k")
            else:
                cells.append(f"${iters_per_1ks:.0f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = TBL / "compute_fair_T500.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
