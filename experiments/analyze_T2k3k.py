"""
Aggregate the T=2000 + T=3000 large-budget runs (Phase B2 revised).

Generates:
  paper/tables/main_table_T2k3k.tex   — final regret + wall-clock per (func, T, method)
  results/figures/regret_curves_T2k3k.png
  results/figures/walltime_T2k3k.png
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
RAW = PROJECT_ROOT / "results" / "raw" / "exp_T2k3k"
FIG = PROJECT_ROOT / "results" / "figures"
TBL = PROJECT_ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)

FUNC_ORDER = ["branin", "hartmann3", "hartmann6"]
FUNC_LABEL = {
    "branin": "Branin (d=2)",
    "hartmann3": "Hartmann-3 (d=3)",
    "hartmann6": "Hartmann-6 (d=6)",
}
METHOD_ORDER = ["rank1_gpucb", "hc_fixed", "hc_adaptive"]
METHOD_LABEL = {
    "rank1_gpucb": "Rank-1 GP-UCB",
    "hc_fixed": "HC-GPUCB (fixed)",
    "hc_adaptive": "HC-GPUCB (adaptive)",
}
METHOD_COLOR = {
    "rank1_gpucb": "#3CB44B",
    "hc_fixed": "#E63946",
    "hc_adaptive": "#F77F00",
}
T_LIST = [2000, 3000]


def load() -> dict:
    """data[T][func][method] -> list of payloads."""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if not RAW.exists():
        return data
    for tdir in RAW.iterdir():
        if not tdir.is_dir() or not tdir.name.startswith("T"):
            continue
        try:
            T = int(tdir.name[1:])
        except ValueError:
            continue
        for fdir in tdir.iterdir():
            if not fdir.is_dir():
                continue
            for mdir in fdir.iterdir():
                if not mdir.is_dir():
                    continue
                for f in sorted(mdir.glob("seed*.pkl")):
                    with open(f, "rb") as fh:
                        data[T][fdir.name][mdir.name].append(pickle.load(fh))
    return data


def make_table(data) -> str:
    Ts_present = [T for T in T_LIST if T in data]
    cols = " | ".join(["cc"] * len(Ts_present))
    lines = [r"\begin{tabular}{l|" + cols + "}", r"\toprule"]
    head = "Method & " + " & ".join(
        f"\\multicolumn{{2}}{{c{'|' if T != Ts_present[-1] else ''}}}{{$T={T}$}}"
        for T in Ts_present
    ) + r" \\"
    lines.append(head)
    lines.append(" & " + " & ".join(["regret & time (s)"] * len(Ts_present)) + r" \\")
    lines.append(r"\midrule")
    funcs = [f for f in FUNC_ORDER if any(f in data[T] for T in Ts_present)]
    for fname in funcs:
        lines.append(r"\multicolumn{" + str(1 + 2 * len(Ts_present)) + r"}{l}{\textit{" + FUNC_LABEL[fname] + r"}} \\")
        # collect raw means per T per method (this function only)
        raw_reg = {}  # (T, m) -> (mean, sem)
        raw_t = {}    # (T, m) -> mean
        for T in Ts_present:
            for m in METHOD_ORDER:
                runs = data.get(T, {}).get(fname, {}).get(m, [])
                if not runs:
                    continue
                regs = np.array([r["simple_regret"][-1] for r in runs])
                times = np.array([r["elapsed_total"] for r in runs])
                raw_reg[(T, m)] = (float(regs.mean()),
                                   float(regs.std(ddof=1) / np.sqrt(len(regs))) if len(regs) > 1 else 0.0)
                raw_t[(T, m)] = float(times.mean())
        # per-T best/second-best
        emph_r = {}; emph_t = {}
        for T in Ts_present:
            rg = sorted([(m, raw_reg[(T, m)][0]) for m in METHOD_ORDER if (T, m) in raw_reg],
                        key=lambda kv: kv[1])
            emph_r[T] = (rg[0][0] if rg else None, rg[1][0] if len(rg) > 1 else None)
            tt = sorted([(m, raw_t[(T, m)]) for m in METHOD_ORDER if (T, m) in raw_t],
                        key=lambda kv: kv[1])
            emph_t[T] = (tt[0][0] if tt else None, tt[1][0] if len(tt) > 1 else None)

        for m in METHOD_ORDER:
            if not any((T, m) in raw_reg for T in Ts_present):
                continue
            cells = [METHOD_LABEL[m]]
            for T in Ts_present:
                if (T, m) not in raw_reg:
                    cells.extend(["--", "--"]); continue
                rm, rs = raw_reg[(T, m)]
                tm = raw_t[(T, m)]
                r_str = f"${rm:.3f}\\pm{rs:.3f}$"
                t_str = f"${tm:.0f}$"
                br, sr = emph_r[T]
                if m == br: r_str = "$\\boldsymbol{" + r_str[1:-1] + "}$"
                elif m == sr: r_str = "$\\underline{" + r_str[1:-1] + "}$"
                bt, st = emph_t[T]
                if m == bt: t_str = "$\\boldsymbol{" + t_str[1:-1] + "}$"
                elif m == st: t_str = "$\\underline{" + t_str[1:-1] + "}$"
                cells.append(r_str); cells.append(t_str)
            lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def fig_walltime(data):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=False)
    for ax, fname in zip(axes, FUNC_ORDER):
        for m in METHOD_ORDER:
            for T in T_LIST:
                runs = data[T].get(fname, {}).get(m, [])
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
                arr = np.stack([c[:min_len] for c in curves])
                mean = arr.mean(axis=0)
                t = np.arange(1, len(mean) + 1)
                style = "-" if T == T_LIST[-1] else "--"
                ax.plot(t, mean, style, lw=1.6, color=METHOD_COLOR[m],
                        label=f"{METHOD_LABEL[m]} (T={T})" if T == T_LIST[0] else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("BO iteration $t$")
        ax.set_ylabel("Cumulative wall-clock (s)")
        ax.set_title(FUNC_LABEL[fname])
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out = FIG / "walltime_T2k3k.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "walltime_T2k3k.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    data = load()
    if not data:
        print("WARNING: no data in", RAW)
        return
    n_total = sum(len(rs) for T in data.values() for f in T.values() for rs in f.values())
    print(f"loaded {n_total} runs across budgets {sorted(data.keys())}")

    # summary
    for T in sorted(data.keys()):
        print(f"\n=== T={T} ===")
        for fname in FUNC_ORDER:
            if fname not in data[T]:
                continue
            print(f"  {FUNC_LABEL[fname]}:")
            for m in METHOD_ORDER:
                runs = data[T][fname].get(m, [])
                if not runs:
                    continue
                regs = np.array([r["simple_regret"][-1] for r in runs])
                times = np.array([r["elapsed_total"] for r in runs])
                print(f"    {METHOD_LABEL[m]:<22} regret={regs.mean():.4f}±{regs.std(ddof=1)/np.sqrt(len(regs)) if len(regs)>1 else 0:.4f}  "
                      f"time={times.mean():6.0f}s ({len(runs)} seeds)")

    tex = make_table(data)
    out = TBL / "main_table_T2k3k.tex"
    out.write_text(tex)
    print(f"\nsaved {out}")

    fig_walltime(data)


if __name__ == "__main__":
    main()
