"""Heatmap of (c_0, c_R) vs final regret and wall-clock for HC-GPUCB on Hartmann-3."""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "results" / "raw" / "exp_ablation"
FIG = PROJECT_ROOT / "results" / "figures"
TBL = PROJECT_ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TBL.mkdir(parents=True, exist_ok=True)


def main():
    if not RAW.exists():
        print("no ablation data yet"); return
    data = defaultdict(list)
    for d in RAW.iterdir():
        if not d.is_dir():
            continue
        for f in d.glob("seed*.pkl"):
            with open(f, "rb") as fh:
                p = pickle.load(fh)
            data[(p["c0"], p["cR"])].append(p)
    if not data:
        print("ablation directory empty"); return

    c0_vals = sorted(set(k[0] for k in data))
    cR_vals = sorted(set(k[1] for k in data))
    regret = np.full((len(c0_vals), len(cR_vals)), np.nan)
    times = np.full_like(regret, np.nan)

    for i, c0 in enumerate(c0_vals):
        for j, cR in enumerate(cR_vals):
            runs = data.get((c0, cR), [])
            if not runs:
                continue
            regret[i, j] = np.mean([r["simple_regret"][-1] for r in runs])
            times[i, j] = np.mean([r["elapsed_total"] for r in runs])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, mat, title, cmap, fmt in [
        (axes[0], regret, "Final simple regret (lower=better)", "Reds", ".3f"),
        (axes[1], times, "Wall-clock (s, lower=better)", "Blues", ".0f"),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(cR_vals)))
        ax.set_xticklabels([f"{v:g}" for v in cR_vals])
        ax.set_yticks(range(len(c0_vals)))
        ax.set_yticklabels([f"{v:g}" for v in c0_vals])
        ax.set_xlabel("$c_R$ (local radius constant)")
        ax.set_ylabel("$c_0$ (Phase-1 length constant)")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    txt = "--"
                else:
                    txt = f"{v:{fmt}}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if (im.norm(v) > 0.5 if not np.isnan(v) else False) else "black",
                        fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("HC-GPUCB ablation: $(c_0, c_R)$ on Hartmann-3, T=200, 3 seeds")
    fig.tight_layout()
    fig.savefig(FIG / "ablation_heatmap.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "ablation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / 'ablation_heatmap.png'}")

    # LaTeX table
    lines = []
    lines.append(r"\begin{tabular}{c|" + "c" * len(cR_vals) + r"|" + "c" * len(cR_vals) + r"}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{" + str(len(cR_vals)) + r"}{c|}{Final regret} & \multicolumn{" + str(len(cR_vals)) + r"}{c}{Wall-clock (s)} \\")
    lines.append(r"$c_0\backslash c_R$ & " + " & ".join([f"${v:g}$" for v in cR_vals]) + " & " + " & ".join([f"${v:g}$" for v in cR_vals]) + r" \\")
    lines.append(r"\midrule")
    for i, c0 in enumerate(c0_vals):
        row = [f"${c0:g}$"]
        for j in range(len(cR_vals)):
            v = regret[i, j]
            row.append("--" if np.isnan(v) else f"${v:.3f}$")
        for j in range(len(cR_vals)):
            v = times[i, j]
            row.append("--" if np.isnan(v) else f"${v:.0f}$")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    out = TBL / "ablation_table.tex"
    out.write_text("\n".join(lines))
    print(f"saved {out}")

    print("\n=== Summary ===")
    for i, c0 in enumerate(c0_vals):
        for j, cR in enumerate(cR_vals):
            print(f"  c0={c0:g}, cR={cR:g}:  regret={regret[i,j]:.4f}  time={times[i,j]:.1f}s")


if __name__ == "__main__":
    main()
