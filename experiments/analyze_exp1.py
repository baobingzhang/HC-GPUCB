"""
Analyze exp1 results: load all pickles, generate figures and LaTeX tables.

Outputs:
  results/figures/regret_curves.png      (regret vs t per function)
  results/figures/walltime_compare.png   (cumulative wall-clock per function)
  results/figures/switch_distribution.png (when does HC-GPUCB switch?)
  results/figures/speedup_bars.png        (avg speedup vs vanilla)
  paper/tables/main_table.tex             (final-T regret + final wall-clock)
  paper/tables/speedup_table.tex          (per-function speedup)
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--raw", type=str, default=str(PROJECT_ROOT / "results" / "raw" / "exp_T80"))
_ap.add_argument("--tag", type=str, default="T80",
                 help="suffix appended to figure/table filenames")
_args, _ = _ap.parse_known_args()

RAW = Path(_args.raw)
TAG = _args.tag
FIG = PROJECT_ROOT / "results" / "figures"
TBL = PROJECT_ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)

METHOD_LABEL = {
    "random": "Random",
    "gpucb": "GP-UCB (Iwazaki '25)",
    "rank1_gpucb": "Rank-1 GP-UCB",
    "ei": "EI",
    "ts": "Thompson Sampling",
    "sparse_ucb": "Sparse GP-UCB",
    "hc_fixed": "HC-GPUCB (fixed)",
    "hc_adaptive": "HC-GPUCB (adaptive)",
}
METHOD_COLOR = {
    "random": "#888888",
    "gpucb": "#2E86AB",
    "rank1_gpucb": "#3CB44B",
    "ei": "#06A77D",
    "ts": "#1F968B",
    "sparse_ucb": "#9467BD",
    "hc_fixed": "#E63946",
    "hc_adaptive": "#F77F00",
}
METHOD_ORDER = ["random", "gpucb", "rank1_gpucb", "ei",
                "ts", "sparse_ucb", "hc_fixed", "hc_adaptive"]
# Regret-guaranteed family (Bayesian-regret-bounded under Iwazaki conditions);
# all other methods are heuristic baselines shown for reference. Used by
# make_main_table() to place a \midrule divider between groups.
GUARANTEED_FAMILY = {"gpucb", "rank1_gpucb", "sparse_ucb", "hc_fixed", "hc_adaptive"}
TABLE_ORDER = ["gpucb", "rank1_gpucb", "sparse_ucb", "hc_fixed", "hc_adaptive",
               "random", "ei", "ts"]
FUNC_LABEL = {"branin": "Branin (d=2)", "hartmann3": "Hartmann-3 (d=3)",
              "hartmann6": "Hartmann-6 (d=6)", "ackley5": "Ackley (d=5)"}
FUNC_ORDER = ["branin", "hartmann3", "hartmann6", "ackley5"]


def _math_bold(s: str) -> str:
    """Bold a math-mode entry: ``$X$`` -> ``$\\boldsymbol{X}$``. Falls back to
    \\textbf for non-math text. Plain ``\\textbf{$...$}`` does NOT bold the
    content because text-mode bold cannot penetrate the math-mode boundary.
    """
    if s.startswith("$") and s.endswith("$"):
        return "$\\boldsymbol{" + s[1:-1] + "}$"
    return r"\textbf{" + s + "}"


def _math_underline(s: str) -> str:
    """Underline a math-mode entry. ``\\underline{$X$}`` works in text mode
    (text-mode underline wraps the math as a box) and is left as-is, but for
    symmetry we also offer this helper using math-mode \\underline."""
    if s.startswith("$") and s.endswith("$"):
        return "$\\underline{" + s[1:-1] + "}$"
    return r"\underline{" + s + "}"


def load_all() -> dict:
    """Return data[func][method] -> list of payload dicts (one per seed)."""
    data: dict = defaultdict(lambda: defaultdict(list))
    for func_dir in RAW.iterdir():
        if not func_dir.is_dir():
            continue
        for method_dir in func_dir.iterdir():
            if not method_dir.is_dir():
                continue
            for f in sorted(method_dir.glob("seed*.pkl")):
                with open(f, "rb") as fh:
                    data[func_dir.name][method_dir.name].append(pickle.load(fh))
    return data


def aggregate(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack equal-length curves; return mean and std error of mean."""
    arr = np.stack(curves, axis=0)
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem


def fig_regret_curves(data):
    nfuncs = len(FUNC_ORDER)
    fig, axes = plt.subplots(1, nfuncs, figsize=(4.0 * nfuncs, 3.5), sharex=False)
    if nfuncs == 1:
        axes = [axes]
    for ax, fname in zip(axes, FUNC_ORDER):
        if fname not in data:
            ax.set_visible(False); continue
        for m in METHOD_ORDER:
            runs = data[fname].get(m, [])
            if not runs:
                continue
            curves = [r["simple_regret"] for r in runs]
            min_len = min(len(c) for c in curves)
            curves = [c[:min_len] for c in curves]
            mean, sem = aggregate(curves)
            t = np.arange(1, len(mean) + 1)
            ax.plot(t, mean, label=METHOD_LABEL[m], color=METHOD_COLOR[m], lw=1.7)
            ax.fill_between(t, mean - sem, mean + sem, color=METHOD_COLOR[m], alpha=0.18)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration $t$")
        ax.set_ylabel(r"Simple regret $f^* - \max_{i \leq t} f(x_i)$")
        ax.set_title(FUNC_LABEL[fname])
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG / f"regret_curves_{TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"regret_curves_{TAG}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_walltime(data):
    nfuncs = len(FUNC_ORDER)
    fig, axes = plt.subplots(1, nfuncs, figsize=(4.0 * nfuncs, 3.3), sharex=False)
    if nfuncs == 1:
        axes = [axes]
    for ax, fname in zip(axes, FUNC_ORDER):
        if fname not in data:
            ax.set_visible(False); continue
        # only methods with per_step_time
        for m in ["gpucb", "rank1_gpucb", "hc_fixed", "hc_adaptive"]:
            runs = data[fname].get(m, [])
            if not runs:
                continue
            curves = []
            for r in runs:
                pst = r.get("per_step_time", [])
                if not pst:
                    continue
                curves.append(np.cumsum(np.asarray(pst)))
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            curves = [c[:min_len] for c in curves]
            mean, sem = aggregate(curves)
            t = np.arange(1, len(mean) + 1)
            ax.plot(t, mean, label=METHOD_LABEL[m], color=METHOD_COLOR[m], lw=1.7)
            ax.fill_between(t, mean - sem, mean + sem, color=METHOD_COLOR[m], alpha=0.18)
        ax.set_xlabel("BO iteration")
        ax.set_ylabel("Cumulative wall-clock (s)")
        ax.set_title(FUNC_LABEL[fname])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = FIG / f"walltime_compare_{TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"walltime_compare_{TAG}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_switch_distribution(data):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m in ["hc_fixed", "hc_adaptive"]:
        steps = []
        for fname in FUNC_ORDER:
            for r in data.get(fname, {}).get(m, []):
                if r.get("switch_step") is not None:
                    steps.append(r["switch_step"] / max(1, r["T"]))
        if steps:
            ax.hist(steps, bins=15, alpha=0.55, label=METHOD_LABEL[m], color=METHOD_COLOR[m])
    ax.set_xlabel(r"Phase switch step $T_0 / T$")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Phase-1 $\\to$ Phase-2 switching")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = FIG / f"switch_distribution_{TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"switch_distribution_{TAG}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_speedup_bars(data):
    """Bar chart: speedup of HC-GPUCB / rank-1 variants over vanilla GP-UCB, per function."""
    funcs_present = [f for f in FUNC_ORDER if f in data]
    x = np.arange(len(funcs_present))
    methods_to_plot = ["rank1_gpucb", "hc_fixed", "hc_adaptive"]
    width = 0.8 / len(methods_to_plot)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for i, m in enumerate(methods_to_plot):
        speedups = []
        for fname in funcs_present:
            base = data[fname].get("gpucb", [])
            ours = data[fname].get(m, [])
            if not base or not ours:
                speedups.append(0); continue
            base_t = np.mean([r["elapsed_total"] for r in base])
            ours_t = np.mean([r["elapsed_total"] for r in ours])
            speedups.append(base_t / max(ours_t, 1e-6))
        offset = (i - (len(methods_to_plot) - 1) / 2) * width
        bars = ax.bar(x + offset, speedups, width=width,
                      label=METHOD_LABEL[m], color=METHOD_COLOR[m], zorder=2)
        for xi, sv in zip(x + offset, speedups):
            ax.annotate(f"{sv:.2f}×", xy=(xi, sv), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, fontweight="bold", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.axhline(1.0, color="gray", linestyle="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_LABEL[f] for f in funcs_present], rotation=10)
    ax.set_ylabel("Wall-clock speedup over GP-UCB")
    ax.set_title("HC-GPUCB compute-time speedup")
    # Add 25% headroom above the tallest bar so annotations never collide with legend
    cur_top = ax.get_ylim()[1]
    ax.set_ylim(0, max(cur_top, 2.2))
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    out = FIG / f"speedup_bars_{TAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"speedup_bars_{TAG}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def _rank_emph(values, formatter, lower_is_better=True):
    """Wrap each cell with \\textbf{} for the best value and \\underline{} for the
    second-best (lowest if lower_is_better, highest otherwise). Returns a list of
    LaTeX strings aligned with `values`. NaN/None entries are formatted as ``--''
    and never receive emphasis."""
    finite = [(i, float(v)) for i, v in enumerate(values)
              if v is not None and (isinstance(v, (int, float)) and not np.isnan(v))]
    if not finite:
        return [formatter(v) for v in values]
    sorted_finite = sorted(finite, key=lambda kv: kv[1] if lower_is_better else -kv[1])
    best_idx = sorted_finite[0][0] if len(sorted_finite) >= 1 else None
    second_idx = sorted_finite[1][0] if len(sorted_finite) >= 2 else None
    out = []
    for i, v in enumerate(values):
        s = formatter(v) if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "--"
        if i == best_idx:
            out.append(_math_bold(s))
        elif i == second_idx:
            out.append(_math_underline(s))
        else:
            out.append(s)
    return out


def make_main_table(data) -> str:
    """Final simple-regret and total wall-clock per (function, method).

    Per function, bold the method with the lowest mean regret and underline the
    second-lowest; same for wall-clock. Random and methods with no runs are
    excluded from the ranking (Random is excluded because it is a non-adaptive
    sanity-check baseline; including it would dilute the bold/underline signal).
    """
    funcs_present = [f for f in FUNC_ORDER if f in data]
    methods_with_data = [m for m in METHOD_ORDER if any(data[f].get(m) for f in funcs_present)]
    # Random is shown but excluded from the emphasis ranking (it is a non-adaptive
    # sanity baseline; ranking against it would dilute the bold/underline signal).
    rank_methods = [m for m in methods_with_data if m != "random"]
    raw_regret = {}  # (f, m) -> (mean, sem)
    raw_time = {}    # (f, m) -> mean
    for f in funcs_present:
        for m in methods_with_data:
            runs = data[f].get(m, [])
            if not runs:
                continue
            finals = np.array([r["simple_regret"][-1] for r in runs])
            times = np.array([r["elapsed_total"] for r in runs])
            raw_regret[(f, m)] = (float(finals.mean()),
                                  float(finals.std(ddof=1) / np.sqrt(len(finals))) if len(finals) > 1 else 0.0)
            raw_time[(f, m)] = float(times.mean())

    # per-func best/second-best (only over rank_methods, i.e. excluding Random)
    emph_regret = {}
    emph_time = {}
    for f in funcs_present:
        rg = sorted([(m, raw_regret[(f, m)][0]) for m in rank_methods if (f, m) in raw_regret],
                    key=lambda kv: kv[1])
        emph_regret[f] = (rg[0][0] if rg else None, rg[1][0] if len(rg) > 1 else None)
        tt = sorted([(m, raw_time[(f, m)]) for m in rank_methods if (f, m) in raw_time],
                    key=lambda kv: kv[1])
        emph_time[f] = (tt[0][0] if tt else None, tt[1][0] if len(tt) > 1 else None)

    lines = []
    lines.append(r"\begin{tabular}{l|cc|cc|cc|cc}")
    lines.append(r"\toprule")
    head = " & ".join([f"\\multicolumn{{2}}{{c|}}{{{FUNC_LABEL[f]}}}" for f in funcs_present])
    lines.append(f"Method & {head} \\\\")
    lines.append(" & " + " & ".join(["regret & time (s)"] * len(funcs_present)) + r" \\")
    lines.append(r"\midrule")
    n_cols = 1 + 2 * len(funcs_present)
    lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{Regret-guaranteed family (Bayesian-regret bound under Iwazaki conditions)}} \\")
    last_was_guaranteed = True
    for idx, m in enumerate(TABLE_ORDER):
        if not any(data[f].get(m) for f in funcs_present):
            continue
        # Insert divider when crossing from guaranteed family into reference heuristics.
        is_guaranteed = m in GUARANTEED_FAMILY
        if last_was_guaranteed and not is_guaranteed:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{Heuristic-reference family (no Bayesian regret guarantee; shown for context)}} \\")
        last_was_guaranteed = is_guaranteed
        cells = [METHOD_LABEL[m]]
        for fname in funcs_present:
            r_data = raw_regret.get((fname, m))
            t_data = raw_time.get((fname, m))
            if r_data is None or t_data is None:
                cells.append("--"); cells.append("--"); continue
            r_mean, r_sem = r_data
            r_str = f"${r_mean:.3f} \\pm {r_sem:.3f}$"
            t_str = f"${t_data:.1f}$"
            best_m, second_m = emph_regret[fname]
            if m == best_m:
                r_str = _math_bold(r_str)
            elif m == second_m:
                r_str = _math_underline(r_str)
            best_t, second_t = emph_time[fname]
            if m == best_t:
                t_str = _math_bold(t_str)
            elif m == second_t:
                t_str = _math_underline(t_str)
            cells.append(r_str); cells.append(t_str)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    data = load_all()
    if not data:
        print("WARNING: no data found in", RAW)
        return
    n_total = sum(len(rs) for f in data.values() for rs in f.values())
    print(f"loaded {n_total} runs across {len(data)} functions")

    fig_regret_curves(data)
    fig_walltime(data)
    fig_switch_distribution(data)
    fig_speedup_bars(data)

    tex = make_main_table(data)
    out = TBL / f"main_table_{TAG}.tex"
    out.write_text(tex)
    print(f"saved {out}")

    # also dump a quick summary
    print("\n=== Summary ===")
    for fname in FUNC_ORDER:
        if fname not in data:
            continue
        print(f"\n{FUNC_LABEL[fname]}:")
        for m in METHOD_ORDER:
            runs = data[fname].get(m, [])
            if not runs:
                continue
            finals = np.array([r["simple_regret"][-1] for r in runs])
            times = np.array([r["elapsed_total"] for r in runs])
            switches = [r.get("switch_step") for r in runs if r.get("switch_step") is not None]
            sw_str = f" switch~{np.mean(switches):.0f}" if switches else ""
            print(f"  {METHOD_LABEL[m]:<26} regret={finals.mean():.3f}±{finals.std(ddof=1)/np.sqrt(len(finals)):.3f}  "
                  f"time={times.mean():6.1f}s ({len(runs)} seeds){sw_str}")


if __name__ == "__main__":
    main()
