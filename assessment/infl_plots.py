"""Plots for the inflation-corpus tests (style as assessment/make_plots.py)"""
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PLOTS = os.environ.get("PLOTS", os.path.join(os.path.dirname(HERE), "Plots"))
OLD = os.path.join(PLOTS, "Old")
C = {"katz_norm": "#2a78d6", "kn_mkn": "#eb6834", "kn_tuned": "#1baf7a", "kn_fixed": "#6250d6"}
LAB = {"katz_norm": "Katz (normalised)", "kn_mkn": "KN (Chen-Goodman D)", "kn_tuned": "KN (CV-tuned D)",
       "kn_fixed": "KN (fixed D = 0.8)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def save(fig, name):
    os.makedirs(OLD, exist_ok=True)
    path = os.path.join(PLOTS, name)
    if os.path.isfile(path):
        shutil.copy(path, os.path.join(OLD, name))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def curves(g, ref="katz_norm"):
    """mean held-out NLL and paired gain vs the (normalised) Katz model per n (equations finite
    under all models)"""
    ns = sorted(g.n.unique())
    mean, gain = {}, {}
    for n in ns:
        piv = g[g.n == n].pivot(index="eq", columns="model", values="logp")
        fin = np.isfinite(piv).all(axis=1)
        for m in list(C) + ["katz"]:
            v = -piv.loc[fin, m]
            mean.setdefault(m, []).append((v.mean(), v.std(ddof=1) / np.sqrt(len(v))))
            if m in C and m != ref:
                d = piv.loc[fin, m] - piv.loc[fin, ref]
                gain.setdefault(m, []).append((d.mean(), d.std(ddof=1) / np.sqrt(len(d))))
    return ns, {m: np.array(v) for m, v in mean.items()}, {m: np.array(v) for m, v in gain.items()}


def within():
    df = pd.read_csv(os.path.join(RES, "infl_within.csv"))
    df = df[df.cv == "fam"]
    titles = {"A": "set A (45 potentials)", "B": "set B (58)", "M": "full list (67)"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for j, c in enumerate(("A", "B", "M")):
        ns, mean, gain = curves(df[(df.corpus == c) & (df.rep == "root")])
        nb, mb, _ = curves(df[(df.corpus == c) & (df.rep == "base")])
        ax = axes[0, j]
        ax.plot(nb, mb["katz"][:, 0], color=C["katz_norm"], ls="--", lw=1.5,
                label="Katz (unnormalised), original representation")
        for m in C:
            v = mean[m]
            ax.errorbar(ns, v[:, 0], v[:, 1], color=C[m], marker="o", ms=4, lw=2,
                        capsize=2, label=LAB[m] + ", root context")
        ax.text(0.02, 0.97, titles[c], transform=ax.transAxes, va="top")
        ax = axes[1, j]
        ax.axhline(0, color="0.5", lw=1)
        for m, v in gain.items():
            ax.errorbar(ns, v[:, 0], v[:, 1], color=C[m], marker="o", ms=4, lw=2, capsize=2)
        ax.set_xlabel("n (context length)")
        ax.set_xticks(ns)
    axes[0, 0].set_ylabel("held-out NLL per potential [nats]")
    axes[1, 0].set_ylabel("NLL(Katz norm.) - NLL(KN) [nats]")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    for row in axes:
        lo = min(a.get_ylim()[0] for a in row)
        hi = max(a.get_ylim()[1] for a in row)
        for a in row:
            a.set_ylim(lo, hi)
    save(fig, "kn_inflation_nll_vs_n.png")


def cross():
    df = pd.read_csv(os.path.join(RES, "infl_cross.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for j, (tr, te, lab) in enumerate((("physics", "M", "train physics (161), test potentials (67)"),
                                       ("M", "physics", "train potentials (67), test physics (161)"))):
        ax = axes[j]
        ax.axhline(0, color="0.5", lw=1)
        for rep, ls in (("base", "--"), ("root", "-")):
            ns, _, gain = curves(df[(df.train == tr) & (df.test == te) & (df.rep == rep)])
            for m in ("kn_tuned", "kn_mkn"):
                v = gain[m]
                ax.errorbar(ns, v[:, 0], v[:, 1], color=C[m], ls=ls, marker="o", ms=4, lw=2,
                            capsize=2, label=f"{LAB[m]}, " + ("root context" if rep == "root"
                                                              else "original representation"))
        ax.text(0.02, 0.97, lab, transform=ax.transAxes, va="top")
        ax.set_xlabel("n (context length)")
        ax.set_xticks(ns)
    axes[0].set_ylabel("NLL(Katz norm.) - NLL(KN) [nats]")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    save(fig, "kn_inflation_cross.png")


if __name__ == "__main__":
    within()
    cross()
