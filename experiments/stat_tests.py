"""
Paired Wilcoxon signed-rank tests: HC-adaptive vs each baseline,
per (function, budget). Holm-Bonferroni correction (uniformly more
powerful than plain Bonferroni) is the default; --correction bonferroni
selects classical Bonferroni.

Usage:
    python experiments/stat_tests.py --raw results/raw/exp_T80 --tag T80
    python experiments/stat_tests.py --raw results/raw/exp_T500 --tag T500
    python experiments/stat_tests.py --raw results/raw/exp_T500 --tag T500 --correction bonferroni
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FUNC_LABEL = {
    "branin": "Branin (d=2)",
    "hartmann3": "Hartmann-3 (d=3)",
    "hartmann6": "Hartmann-6 (d=6)",
    "ackley5": "Ackley (d=5)",
}
FUNC_ORDER = ["branin", "hartmann3", "hartmann6", "ackley5"]

METHOD_LABEL = {
    "random": "Random",
    "gpucb": "GP-UCB",
    "rank1_gpucb": "Rank-1 GP-UCB",
    "ei": "EI",
    "ts": "Thompson Sampling",
    "sparse_ucb": "Sparse GP-UCB",
    "hc_fixed": "HC-GPUCB (fixed)",
}
BASELINES = ["random", "gpucb", "rank1_gpucb", "ei", "ts", "sparse_ucb", "hc_fixed"]
TARGET = "hc_adaptive"


def load_all(raw: Path) -> dict:
    data: dict = defaultdict(lambda: defaultdict(dict))  # data[func][method][seed] -> regret
    for func_dir in raw.iterdir():
        if not func_dir.is_dir():
            continue
        for method_dir in func_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for f in sorted(method_dir.glob("seed*.pkl")):
                with open(f, "rb") as fh:
                    p = pickle.load(fh)
                seed = p["seed"]
                data[func_dir.name][method_dir.name][seed] = float(p["simple_regret"][-1])
    return data


def paired_wilcoxon(target_by_seed: dict, base_by_seed: dict) -> tuple[float, float, int]:
    """Paired Wilcoxon on common seeds. Returns (statistic, p_two_sided, n_pairs)."""
    common = sorted(set(target_by_seed) & set(base_by_seed))
    if len(common) < 2:
        return float("nan"), float("nan"), len(common)
    diffs = np.array([target_by_seed[s] - base_by_seed[s] for s in common])
    if np.allclose(diffs, 0):
        return 0.0, 1.0, len(common)
    try:
        stat, p = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        return float(stat), float(p), len(common)
    except ValueError:
        return float("nan"), float("nan"), len(common)


def paired_tost(target_by_seed: dict, base_by_seed: dict, margin_rel: float = 0.10) -> tuple[float, int]:
    """Paired TOST (two one-sided tests) for equivalence at relative margin
    `margin_rel` (default 10%). Equivalence margin is `margin_rel * mean(base)`.

    Returns (p_max, n) where p_max = max(p_lower, p_upper). Equivalence
    is concluded at level alpha iff p_max < alpha. NaN if n<2 or margin
    not meaningful (base mean ≈ 0).

    H0: |target - base| >= margin (i.e., NOT equivalent)
    H1: |target - base| <  margin
    """
    common = sorted(set(target_by_seed) & set(base_by_seed))
    n = len(common)
    if n < 2:
        return float("nan"), n
    target = np.array([target_by_seed[s] for s in common])
    base = np.array([base_by_seed[s] for s in common])
    base_mean = float(np.mean(np.abs(base)))
    if base_mean < 1e-9:
        return float("nan"), n  # baseline ~0, equivalence trivially satisfied or undefined
    margin = margin_rel * base_mean
    diffs = target - base
    # Lower test: H0_L: diff <= -margin vs H1_L: diff > -margin
    # Upper test: H0_U: diff >=  margin vs H1_U: diff <  margin
    try:
        _, p_low = wilcoxon(diffs + margin, alternative="greater", zero_method="wilcox")
        _, p_up = wilcoxon(diffs - margin, alternative="less", zero_method="wilcox")
        return float(max(p_low, p_up)), n
    except ValueError:
        return float("nan"), n


def fmt_p(p: float, sig: bool) -> str:
    if np.isnan(p):
        return "--"
    star = r"$^{*}$" if sig else ""
    if p < 1e-3:
        return f"$<$0.001{star}"
    return f"{p:.3f}{star}"


def holm_significance(p_values: list[float], alpha: float) -> list[bool]:
    """Holm-Bonferroni step-down. Returns boolean significance per p-value
    (in original order). Holm is uniformly more powerful than Bonferroni
    while controlling family-wise error rate at alpha."""
    valid = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
    if not valid:
        return [False] * len(p_values)
    valid.sort(key=lambda x: x[1])
    n = len(valid)
    sig = [False] * len(p_values)
    for rank, (orig_idx, p) in enumerate(valid):
        threshold = alpha / (n - rank)
        if p < threshold:
            sig[orig_idx] = True
        else:
            # once a p fails, all subsequent (larger) p's also fail
            break
    return sig


def bonferroni_significance(p_values: list[float], alpha: float) -> list[bool]:
    n = sum(1 for p in p_values if not np.isnan(p))
    threshold = alpha / max(n, 1)
    return [(not np.isnan(p)) and p < threshold for p in p_values]


def bh_fdr_significance(p_values: list[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg FDR control. Less strict than FWER methods;
    standard in ML benchmark tables. Returns boolean significance per
    p-value in original order, controlling false discovery rate at alpha."""
    valid = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
    if not valid:
        return [False] * len(p_values)
    valid.sort(key=lambda x: x[1])
    n = len(valid)
    sig = [False] * len(p_values)
    max_k = -1
    for k, (orig_idx, p) in enumerate(valid, start=1):
        threshold = k * alpha / n
        if p <= threshold:
            max_k = k
    for k, (orig_idx, p) in enumerate(valid, start=1):
        if k <= max_k:
            sig[orig_idx] = True
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--correction", choices=("bh", "holm", "bonferroni"), default="bh",
                    help="bh = Benjamini-Hochberg FDR (default, standard in ML benchmark tables); "
                         "holm = Holm-Bonferroni FWER; bonferroni = classical Bonferroni FWER")
    ap.add_argument("--tost", action="store_true",
                    help="Also run TOST (two one-sided tests) for equivalence at "
                         "10%% relative margin and emit a separate equivalence table.")
    ap.add_argument("--tost-margin", type=float, default=0.10,
                    help="Relative equivalence margin for TOST (default 0.10 = 10%% of baseline mean)")
    args = ap.parse_args()

    data = load_all(Path(args.raw))
    if not data:
        print("ERROR: no data in", args.raw)
        return

    funcs_present = [f for f in FUNC_ORDER if f in data]
    rows = []  # (func, base_method, statistic, p, n)
    for fname in funcs_present:
        if TARGET not in data[fname]:
            continue
        target = data[fname][TARGET]
        for base in BASELINES:
            if base not in data[fname]:
                continue
            stat, p, n = paired_wilcoxon(target, data[fname][base])
            rows.append((fname, base, stat, p, n))

    n_comparisons = len(rows)
    p_values = [r[3] for r in rows]
    if args.correction == "bh":
        sig_flags = bh_fdr_significance(p_values, args.alpha)
        corr_label = f"Benjamini-Hochberg FDR at alpha={args.alpha} ({n_comparisons} comparisons)"
    elif args.correction == "holm":
        sig_flags = holm_significance(p_values, args.alpha)
        corr_label = f"Holm-Bonferroni FWER at alpha={args.alpha} ({n_comparisons} comparisons)"
    else:
        sig_flags = bonferroni_significance(p_values, args.alpha)
        corr_label = f"Bonferroni FWER at alpha/{n_comparisons}={args.alpha/max(n_comparisons,1):.5f}"
    print(corr_label + "\n")

    print(f"{'function':<14} {'baseline':<18} {'n':>3} {'W':>8} {'p':>8}  {'sig?':<5}")
    for (f, b, s, p, n), sig in zip(rows, sig_flags):
        s_str = "--" if np.isnan(s) else f"{s:.1f}"
        p_str = "--" if np.isnan(p) else f"{p:.3g}"
        print(f"{f:<14} {b:<18} {n:>3} {s_str:>8} {p_str:>8}  {'*' if sig else '':<5}")

    # ---- LaTeX table ----
    out_dir = PROJECT_ROOT / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"stat_tests_{args.tag}.tex"

    # Group by function: rows = baselines, columns = funcs (regret diff + p-value)
    bases_seen = sorted({b for _, b, _, _, _ in rows}, key=BASELINES.index)
    lines = []
    n_funcs = len(funcs_present)
    col_spec = "l" + "c" * n_funcs
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    head = "Baseline & " + " & ".join(FUNC_LABEL[f].split(" (")[0] for f in funcs_present) + r" \\"
    lines.append(head)
    lines.append(r"\midrule")
    sig_lookup = {(r[0], r[1]): s for r, s in zip(rows, sig_flags)}
    for base in bases_seen:
        cells = [METHOD_LABEL.get(base, base)]
        for f in funcs_present:
            entry = next((r for r in rows if r[0] == f and r[1] == base), None)
            if entry is None:
                cells.append("--")
            else:
                cells.append(fmt_p(entry[3], sig_lookup.get((entry[0], entry[1]), False)))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out.write_text("\n".join(lines))
    print(f"\nsaved {out}")
    print(corr_label)

    # ---- TOST equivalence table ----
    if args.tost:
        print(f"\n=== TOST equivalence at {args.tost_margin*100:.0f}% relative margin ===")
        tost_rows = []
        for fname in funcs_present:
            if TARGET not in data[fname]:
                continue
            target = data[fname][TARGET]
            for base in BASELINES:
                if base not in data[fname]:
                    continue
                p_tost, n_pairs = paired_tost(target, data[fname][base], args.tost_margin)
                tost_rows.append((fname, base, p_tost, n_pairs))
                eq = "EQ" if (not np.isnan(p_tost)) and p_tost < args.alpha else ""
                p_str = "--" if np.isnan(p_tost) else f"{p_tost:.3f}"
                print(f"  {fname:<12} {base:<14} n={n_pairs:>2} p_TOST={p_str:>6}  {eq}")

        margin_pct = int(round(args.tost_margin * 100))
        margin_tag = "" if margin_pct == 25 else f"_m{margin_pct}"
        out_tost = out_dir / f"tost_tests_{args.tag}{margin_tag}.tex"
        tlines = []
        tlines.append(r"\begin{tabular}{" + col_spec + "}")
        tlines.append(r"\toprule")
        head_tost = ("Baseline & "
                     + " & ".join(FUNC_LABEL[f].split(" (")[0] for f in funcs_present) + r" \\")
        tlines.append(head_tost)
        tlines.append(r"\midrule")
        for base in bases_seen:
            cells = [METHOD_LABEL.get(base, base)]
            for f in funcs_present:
                entry = next((r for r in tost_rows if r[0] == f and r[1] == base), None)
                if entry is None or np.isnan(entry[2]):
                    cells.append("--")
                else:
                    p = entry[2]
                    eq_star = r"$^{\dagger}$" if p < args.alpha else ""
                    if p < 1e-3:
                        cells.append(f"$<$0.001{eq_star}")
                    else:
                        cells.append(f"{p:.3f}{eq_star}")
            tlines.append(" & ".join(cells) + r" \\")
        tlines.append(r"\bottomrule")
        tlines.append(r"\end{tabular}")
        out_tost.write_text("\n".join(tlines))
        print(f"\nsaved {out_tost}")
        print(f"$^\\dagger$ = TOST equivalence at {args.tost_margin*100:.0f}% margin (p < {args.alpha})")


if __name__ == "__main__":
    main()
