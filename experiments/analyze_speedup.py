"""
Plot wall-clock speedup of HC-GPUCB vs vanilla GP-UCB as a function of T,
optionally including Rank-1 GP-UCB as a third curve.

Reads pickles from results/raw/exp_speedup/T{200,500,1000}/... by default,
or from a custom directory via --raw.
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
_ap.add_argument("--raw", type=str, default=str(PROJECT_ROOT / "results" / "raw" / "exp_speedup"),
                 help="Directory holding T<N>/hartmann3/<method>/*.pkl")
_ap.add_argument("--include-T80", action="store_true",
                 help="Also pull T=80 data from exp_T80/hartmann3.")
_ap.add_argument("--tag", type=str, default="",
                 help="Suffix appended to figure filenames (e.g. _pinned).")
_args, _ = _ap.parse_known_args()
RAW = Path(_args.raw)
TAG = _args.tag
FIG = PROJECT_ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    data = defaultdict(lambda: defaultdict(list))  # data[T][method] -> list of payloads

    if _args.include_T80:
        for f in (PROJECT_ROOT / "results" / "raw" / "exp_T80" / "hartmann3").rglob("*.pkl"):
            method = f.parent.name
            if method in ("gpucb", "hc_adaptive"):
                with open(f, "rb") as fh:
                    payload = pickle.load(fh)
                data[80][method].append(payload)

    if RAW.exists():
        for tdir in RAW.iterdir():
            if not tdir.is_dir() or not tdir.name.startswith("T"):
                continue
            try:
                T = int(tdir.name[1:])
            except ValueError:
                continue
            hm3 = tdir / "hartmann3"
            if not hm3.exists():
                continue
            for method_dir in hm3.iterdir():
                method = method_dir.name
                if method not in ("gpucb", "rank1_gpucb", "hc_adaptive"):
                    continue
                for f in method_dir.glob("*.pkl"):
                    with open(f, "rb") as fh:
                        data[T][method].append(pickle.load(fh))

    Ts_with_data = sorted([T for T in data if data[T].get("gpucb") and data[T].get("hc_adaptive")])
    print(f"Speedup data available at T = {Ts_with_data}")
    if not Ts_with_data:
        print("ERROR: no T values have both methods")
        return

    targets = [m for m in ("rank1_gpucb", "hc_adaptive") if any(data[T].get(m) for T in Ts_with_data)]
    target_label = {
        "rank1_gpucb": "Rank-1 GP-UCB",
        "hc_adaptive": "HC-GPUCB (adaptive)",
    }
    target_color = {
        "rank1_gpucb": "#3CB44B",
        "hc_adaptive": "#E63946",
    }

    walltimes = {"gpucb": []}
    for m in targets:
        walltimes[m] = []
    speedups = {m: [] for m in targets}
    speedup_sem = {m: [] for m in targets}

    for T in Ts_with_data:
        bt = np.array([r["elapsed_total"] for r in data[T]["gpucb"]])
        walltimes["gpucb"].append(bt.mean())
        line = f"T={T:>4}  gpucb={bt.mean():7.1f}s"
        for m in targets:
            mt = np.array([r["elapsed_total"] for r in data[T].get(m, [])])
            if mt.size == 0:
                walltimes[m].append(np.nan)
                speedups[m].append(np.nan); speedup_sem[m].append(np.nan)
                continue
            walltimes[m].append(mt.mean())
            ratios = bt[:, None] / mt[None, :]
            speedups[m].append(ratios.mean())
            speedup_sem[m].append(ratios.std(ddof=1) / np.sqrt(ratios.size) if ratios.size > 1 else 0.0)
            line += f"  {m}={mt.mean():7.1f}s ({speedups[m][-1]:.2f}x)"
        print(line)

    # ---- Plot 1: speedup vs T ----
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for m in targets:
        sp = np.array(speedups[m])
        se = np.array(speedup_sem[m])
        ok = ~np.isnan(sp)
        ax.errorbar(np.array(Ts_with_data)[ok], sp[ok], yerr=se[ok], marker="o",
                    lw=2, capsize=4, color=target_color[m], label=target_label[m])
    ax.axhline(1.0, color="gray", linestyle=":", lw=1, label="parity")
    ax.set_xscale("log")
    ax.set_xlabel("Budget $T$")
    ax.set_ylabel("Wall-clock speedup over GP-UCB")
    ax.set_title("Speedup vs.\\ vanilla GP-UCB across budgets (Hartmann-3, $d=3$)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"speedup_vs_T{TAG}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"speedup_vs_T{TAG}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / f'speedup_vs_T{TAG}.png'}")

    # ---- Plot 2: absolute wall-clock vs T ----
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(Ts_with_data, walltimes["gpucb"], "o-", lw=2, color="#2E86AB", label="GP-UCB (Iwazaki '25)")
    markers = {"rank1_gpucb": "s-", "hc_adaptive": "^-"}
    for m in targets:
        wt = np.array(walltimes[m])
        ok = ~np.isnan(wt)
        ax.plot(np.array(Ts_with_data)[ok], wt[ok], markers[m], lw=2,
                color=target_color[m], label=target_label[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Budget $T$")
    ax.set_ylabel("Total wall-clock (s)")
    ax.set_title("Wall-clock cost vs budget (Hartmann-3, $d=3$)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"walltime_vs_T{TAG}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"walltime_vs_T{TAG}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / f'walltime_vs_T{TAG}.png'}")


if __name__ == "__main__":
    main()
