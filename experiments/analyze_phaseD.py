"""
Analyze D2/D3/D5 raw experiment data and emit the LaTeX tables that the
main paper §V-J/K/L/M sections \\input.

Outputs:
  paper/tables/decoupled_T500.tex     (D2)
  paper/tables/hc_rank1_T2k3k.tex     (D3)
  paper/tables/noise_T200.tex         (D5.2)
  paper/tables/highdim_T300.tex       (D5.3)
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TBL = PROJECT_ROOT / "paper" / "tables"
TBL.mkdir(parents=True, exist_ok=True)


def _load(d: Path) -> tuple[list[float], list[float]]:
    """Return (regrets, times) lists for a method directory."""
    if not d.exists():
        return [], []
    regs, times = [], []
    for f in sorted(d.glob("seed*.pkl")):
        with open(f, "rb") as fh:
            p = pickle.load(fh)
        regs.append(float(p["simple_regret"][-1]))
        times.append(float(p["elapsed_total"]))
    return regs, times


def _stat(values: list[float]) -> str:
    if not values:
        return r"\textit{TBD}"
    arr = np.array(values)
    if len(arr) == 1:
        return f"${arr.mean():.3f}$"
    return f"${arr.mean():.3f}{{\\scriptstyle\\,\\pm\\,}}{arr.std(ddof=1)/np.sqrt(len(arr)):.3f}$"


def _time_stat(values: list[float]) -> str:
    if not values:
        return r"\textit{TBD}"
    return f"${np.mean(values):.0f}$"


def _emph(s: str, kind: str) -> str:
    """kind in {'bold','underline'}. For ``$X$`` math strings, use math-mode
    \\boldsymbol/\\underline (which actually bolds digits, unlike text-mode
    \\textbf which cannot penetrate math mode). For non-math strings (e.g.
    \\textit{TBD}), fall back to text-mode wrappers."""
    if s.startswith("$") and s.endswith("$"):
        inner = s[1:-1]
        if kind == "bold": return "$\\boldsymbol{" + inner + "}$"
        return "$\\underline{" + inner + "}$"
    if kind == "bold": return r"\textbf{" + s + "}"
    return r"\underline{" + s + "}"


def emit_decoupled():
    """D2 + Hartmann-3 T=500 — vanilla / vanilla_chk / rank1 wall-clock.
    Bold the lowest wall-clock (best) and underline the second-lowest. Same for
    speedup (highest = best, second-highest = underline)."""
    base = PROJECT_ROOT / "results" / "raw"
    cells = {
        "Vanilla GP-UCB": _load(base / "exp_speedup_pinned" / "T500" / "hartmann3" / "gpucb"),
        "Vanilla-Checkpoint GP-UCB": _load(base / "exp_decoupled" / "T500" / "hartmann3" / "vanilla_chk"),
        "Rank-1 GP-UCB": _load(base / "exp_speedup_pinned" / "T500" / "hartmann3" / "rank1_gpucb"),
    }
    vanilla_t = cells["Vanilla GP-UCB"][1]
    vanilla_mean = float(np.mean(vanilla_t)) if vanilla_t else None
    sources = {
        "Vanilla GP-UCB": "---",
        "Vanilla-Checkpoint GP-UCB": "(a)",
        "Rank-1 GP-UCB": "(a) + (b)",
    }
    rows = []  # (name, t_mean_or_None, speedup_or_None)
    for name, (_, times) in cells.items():
        if not times:
            rows.append((name, None, None))
        else:
            t_mean = float(np.mean(times))
            sp = vanilla_mean / t_mean if vanilla_mean else None
            rows.append((name, t_mean, sp))

    # Best / second-best on wall-clock (lower = better) and speedup (higher = better),
    # excluding the Vanilla-vs-vanilla self-comparison from the speedup ranking.
    rk_t = sorted([(n, t) for n, t, _ in rows if t is not None], key=lambda kv: kv[1])
    best_t = rk_t[0][0] if rk_t else None
    second_t = rk_t[1][0] if len(rk_t) > 1 else None
    rk_sp = sorted([(n, s) for n, _, s in rows if s is not None and n != "Vanilla GP-UCB"], key=lambda kv: -kv[1])
    best_sp = rk_sp[0][0] if rk_sp else None
    second_sp = rk_sp[1][0] if len(rk_sp) > 1 else None

    lines = [r"\begin{tabular}{lccc}", r"\toprule"]
    lines.append(r"Method & Wall-clock (s) & Speedup & Saving source \\")
    lines.append(r"\midrule")
    for name, t_mean, sp in rows:
        if t_mean is None:
            t_str = r"\textit{TBD}"
            sp_str = r"\textit{TBD}"
        else:
            t_str = f"${t_mean:.0f}$"
            if name == best_t: t_str = "$\\boldsymbol{" + t_str[1:-1] + "}$"
            elif name == second_t: t_str = "$\\underline{" + t_str[1:-1] + "}$"
            if sp is None:
                sp_str = "---"
            else:
                sp_str = f"${sp:.2f}\\times$"
                if name == best_sp: sp_str = "$\\boldsymbol{" + sp_str[1:-1] + "}$"
                elif name == second_sp: sp_str = "$\\underline{" + sp_str[1:-1] + "}$"
        lines.append(f"{name} & {t_str} & {sp_str} & {sources[name]} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = TBL / "decoupled_T500.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


def emit_hc_rank1():
    """D3: HC-GPUCB-Rank1 T={500,1000,2000,3000} × 3 funcs × 2 modes.
    Bold lower mean regret / lower time per (func, T) cell pair; only 2 methods so
    no underline (would be redundant — second is the only other row)."""
    base = PROJECT_ROOT / "results" / "raw" / "exp_D3_hc_rank1"
    Ts = [500, 1000, 2000, 3000]
    funcs = [("branin", "Branin (d=2)"),
             ("hartmann3", "Hartmann-3 (d=3)"),
             ("hartmann6", "Hartmann-6 (d=6)")]
    methods = [("hc_fixed_rank1", "HC-GPUCB-Rank1 (fixed)"),
               ("hc_adaptive_rank1", "HC-GPUCB-Rank1 (adaptive)")]
    raw_r = {}; raw_t = {}
    for fkey, _ in funcs:
        for T in Ts:
            for mkey, _ in methods:
                regs, times = _load(base / f"T{T}" / fkey / mkey)
                if regs: raw_r[(fkey, T, mkey)] = regs
                if times: raw_t[(fkey, T, mkey)] = times
    emph_r = {}; emph_t = {}
    for fkey, _ in funcs:
        for T in Ts:
            rk = sorted([(m, float(np.mean(raw_r[(fkey, T, m)]))) for m, _ in methods if (fkey, T, m) in raw_r],
                        key=lambda kv: kv[1])
            emph_r[(fkey, T)] = rk[0][0] if rk else None
            tk = sorted([(m, float(np.mean(raw_t[(fkey, T, m)]))) for m, _ in methods if (fkey, T, m) in raw_t],
                        key=lambda kv: kv[1])
            emph_t[(fkey, T)] = tk[0][0] if tk else None

    lines = [r"\begin{tabular}{l|cc|cc|cc|cc}", r"\toprule"]
    head = " & ".join([f"\\multicolumn{{2}}{{c{'|' if T != Ts[-1] else ''}}}{{$T={T}$}}" for T in Ts])
    lines.append(f" & {head} \\\\")
    lines.append("Method & " + " & ".join(["regret & time (s)"] * len(Ts)) + r" \\")
    lines.append(r"\midrule")
    for fkey, flabel in funcs:
        lines.append(r"\multicolumn{" + str(1 + 2 * len(Ts)) + r"}{l}{\textit{" + flabel + r"}} \\")
        for mkey, mlabel in methods:
            cells = [mlabel]
            for T in Ts:
                r_str = _stat(raw_r.get((fkey, T, mkey), []))
                t_str = _time_stat(raw_t.get((fkey, T, mkey), []))
                if mkey == emph_r.get((fkey, T)): r_str = _emph(r_str, "bold")
                if mkey == emph_t.get((fkey, T)): t_str = _emph(t_str, "bold")
                cells.append(r_str); cells.append(t_str)
            lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = TBL / "hc_rank1_T2k3k.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


def emit_noise():
    """D5.2: noise sweep at T=200, σ²∈{0.01, 0.1} on hartmann3/hartmann6.
    Bold lowest mean regret per (func, sigma) column; underline second-lowest."""
    base = PROJECT_ROOT / "results" / "raw" / "exp_D5_noise"
    funcs = [("hartmann3", "Hartmann-3 (d=3)"),
             ("hartmann6", "Hartmann-6 (d=6)")]
    sigma_dirs = [("sigma01", r"$\sigma^2{=}0.01$"), ("sigma1", r"$\sigma^2{=}0.1$")]
    methods = [("gpucb", "GP-UCB (Iwazaki '25)"),
               ("rank1_gpucb", "Rank-1 GP-UCB"),
               ("hc_adaptive", "HC-GPUCB (adaptive)"),
               ("ei", "EI")]
    # collect raw regrets per (col_idx, mkey)
    columns = [(fkey, sigdir) for fkey, _ in funcs for sigdir, _ in sigma_dirs]
    raw = {}  # (col, mkey) -> regs list
    for ci, (fkey, sigdir) in enumerate(columns):
        for mkey, _ in methods:
            regs, _ = _load(base / sigdir / fkey / mkey)
            if regs:
                raw[(ci, mkey)] = regs
    # best/second per column
    emph = {}
    for ci in range(len(columns)):
        ranked = sorted([(m, float(np.mean(raw[(ci, m)]))) for m, _ in methods if (ci, m) in raw],
                        key=lambda kv: kv[1])
        emph[ci] = (ranked[0][0] if ranked else None, ranked[1][0] if len(ranked) > 1 else None)

    lines = [r"\begin{tabular}{l|cc|cc}", r"\toprule"]
    h1 = " & ".join([f"\\multicolumn{{2}}{{c{'|' if i==0 else ''}}}{{{flabel}}}" for i, (_, flabel) in enumerate(funcs)])
    lines.append(f" & {h1} \\\\")
    h2 = "Method ($T=200$) & " + " & ".join([f"{s} regret" for fk, fl in funcs for sk, s in sigma_dirs]) + r" \\"
    lines.append(h2)
    lines.append(r"\midrule")
    for mkey, mlabel in methods:
        cells = [mlabel]
        for ci in range(len(columns)):
            if (ci, mkey) not in raw:
                cells.append(r"\textit{TBD}"); continue
            regs = raw[(ci, mkey)]
            s = _stat(regs)
            best, second = emph[ci]
            if mkey == best: s = _emph(s, "bold")
            elif mkey == second: s = _emph(s, "underline")
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = TBL / "noise_T200.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


def emit_highdim():
    """D5.3: Levy-10/20 at T=300.
    Bold lowest mean regret AND lowest time per func; underline second-lowest."""
    base = PROJECT_ROOT / "results" / "raw" / "exp_D5_highdim"
    funcs = [("levy10", "Levy-10 (d=10)"),
             ("levy20", "Levy-20 (d=20)")]
    methods = [("gpucb", "GP-UCB (Iwazaki '25)"),
               ("rank1_gpucb", "Rank-1 GP-UCB"),
               ("hc_adaptive", "HC-GPUCB (adaptive)"),
               ("hc_adaptive_rank1", "HC-GPUCB-Rank1 (adaptive)")]
    raw_r = {}  # (fkey, mkey) -> regs list
    raw_t = {}  # (fkey, mkey) -> times list
    for fkey, _ in funcs:
        for mkey, _ in methods:
            regs, times = _load(base / fkey / mkey)
            if regs:
                raw_r[(fkey, mkey)] = regs
            if times:
                raw_t[(fkey, mkey)] = times
    # Tie-aware emphasis: ALL methods within `tol` of the min get bold; the
    # next strictly-larger value's tie group gets underline. Rounding to the
    # display precision (3 decimals for regret, integer for time) avoids
    # arbitrary-looking single-method bolds when the underlying floats happen
    # to differ in the 4th decimal but show as identical in the table.
    def _tied_emph(items, prec_round):
        """items: list of (method_key, float_value). Returns (bold_set, second_set)."""
        if not items:
            return set(), set()
        rounded = [(m, round(v, prec_round)) for m, v in items]
        rounded.sort(key=lambda kv: kv[1])
        best_v = rounded[0][1]
        bold = {m for m, v in rounded if v == best_v}
        second_candidates = [(m, v) for m, v in rounded if v > best_v]
        if not second_candidates:
            return bold, set()
        second_v = second_candidates[0][1]
        second = {m for m, v in second_candidates if v == second_v}
        return bold, second

    emph_r = {}; emph_t = {}
    for fkey, _ in funcs:
        regret_items = [(m, float(np.mean(raw_r[(fkey, m)]))) for m, _ in methods if (fkey, m) in raw_r]
        emph_r[fkey] = _tied_emph(regret_items, 3)  # round to 3 decimals (matches display)
        time_items = [(m, float(np.mean(raw_t[(fkey, m)]))) for m, _ in methods if (fkey, m) in raw_t]
        emph_t[fkey] = _tied_emph(time_items, 0)    # integer seconds

    lines = [r"\begin{tabular}{l|cc|cc}", r"\toprule"]
    h1 = " & ".join([f"\\multicolumn{{2}}{{c{'|' if i==0 else ''}}}{{{fl}}}" for i, (_, fl) in enumerate(funcs)])
    lines.append(f" & {h1} \\\\")
    lines.append("Method ($T=300$) & regret & time (s) & regret & time (s) \\\\")
    lines.append(r"\midrule")
    for mkey, mlabel in methods:
        cells = [mlabel]
        for fkey, _ in funcs:
            r_str = _stat(raw_r.get((fkey, mkey), []))
            t_str = _time_stat(raw_t.get((fkey, mkey), []))
            br_set, sr_set = emph_r[fkey]
            if mkey in br_set: r_str = _emph(r_str, "bold")
            elif mkey in sr_set: r_str = _emph(r_str, "underline")
            bt_set, st_set = emph_t[fkey]
            if mkey in bt_set: t_str = _emph(t_str, "bold")
            elif mkey in st_set: t_str = _emph(t_str, "underline")
            cells.append(r_str); cells.append(t_str)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = TBL / "highdim_T300.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


def main():
    print("=== D2 decoupled ===")
    emit_decoupled()
    print("\n=== D3 hc_rank1 ===")
    emit_hc_rank1()
    print("\n=== D5.2 noise ===")
    emit_noise()
    print("\n=== D5.3 highdim ===")
    emit_highdim()


if __name__ == "__main__":
    main()
