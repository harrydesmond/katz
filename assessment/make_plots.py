"""Diagnostic plots for the Kneser-Ney vs Katz assessment (PNG in Plots/)."""

import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PLOTS = os.path.join(os.path.dirname(HERE), "Plots")
OLD = os.path.join(PLOTS, "Old")
os.makedirs(OLD, exist_ok=True)

C = {"katz": "#2a78d6", "kn_mkn": "#eb6834", "kn_tuned": "#1baf7a", "kn_cv": "#1baf7a",
     "kn_d": "#6250d6"}
LAB = {"katz": "Katz", "kn_mkn": "KN (Chen-Goodman D)", "kn_tuned": "KN (CV-tuned D)",
       "kn_cv": "KN (CV-tuned D)", "kn_d": "KN (fixed D)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def save(fig, name):
    path = os.path.join(PLOTS, name)
    if os.path.isfile(path):
        shutil.copy(path, os.path.join(OLD, name))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def nll_vs_n():
    df = pd.read_csv(os.path.join(RES, "cv_grouped.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for j, rep in enumerate(("base", "root")):
        g = df[df.rep == rep]
        ns = sorted(g.n.unique())
        mean, gain = {m: [] for m in C if m in g.model.unique()}, {}
        for n in ns:
            piv = g[g.n == n].pivot(index="eq", columns="model", values="total")
            fin = np.isfinite(piv).all(axis=1)
            for m in mean:
                v = -piv.loc[fin, m]
                mean[m].append((v.mean(), v.std(ddof=1) / np.sqrt(len(v))))
                if m != "katz":
                    d = piv.loc[fin, m] - piv.loc[fin, "katz"]
                    gain.setdefault(m, []).append((d.mean(), d.std(ddof=1)
                                                   / np.sqrt(len(d))))
        ax = axes[0, j]
        for m, v in mean.items():
            v = np.array(v)
            ax.errorbar(ns, v[:, 0], v[:, 1], color=C[m], marker="o", ms=5, lw=2,
                        capsize=2, label=LAB[m])
        ax.set_ylabel("held-out NLL per equation [nats]")
        ax.text(0.02, 0.97, "root context" if rep == "root" else "original representation",
                transform=ax.transAxes, va="top")
        ax = axes[1, j]
        ax.axhline(0, color="0.5", lw=1)
        for m, v in gain.items():
            v = np.array(v)
            ax.errorbar(ns, v[:, 0], v[:, 1], color=C[m], marker="o", ms=5, lw=2,
                        capsize=2, label=LAB[m])
        ax.set_ylabel("NLL(Katz) - NLL(KN) [nats]")
        ax.set_xlabel("n (context length)")
        ax.set_xticks(ns)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    ymin = min(a.get_ylim()[0] for a in axes[0])
    ymax = max(a.get_ylim()[1] for a in axes[0])
    for a in axes[0]:
        a.set_ylim(ymin, ymax)
    save(fig, "kn_nll_vs_n.png")


def synthetic():
    df = pd.read_csv(os.path.join(RES, "synthetic.csv"))
    truths = ["corpus", "dirichlet_b30", "dirichlet_b10", "dirichlet_b3"]
    tl = {"corpus": "corpus-fitted truth", "dirichlet_b30": "random, weak context",
          "dirichlet_b10": "random, medium context", "dirichlet_b3": "random, strong context"}
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey="row")
    for i, n in enumerate((2, 3)):
        for j, t in enumerate(truths):
            ax = axes[i, j]
            ax.axhline(0, color="0.5", lw=1)
            g = df[(df.truth == t) & (df.n == n)]
            for m in ("kn_mkn", "kn_cv", "kn_d"):
                h = g[g.model == m].groupby("M").excess_vs_katz
                mu, se = h.mean(), h.std(ddof=1) / np.sqrt(h.count())
                ax.plot(mu.index, mu, color=C[m], marker="o", ms=4, lw=2, label=LAB[m])
                ax.fill_between(mu.index, mu - se, mu + se, color=C[m], alpha=0.2, lw=0)
            ax.set_xscale("log")
            ax.axvline(161, color="0.7", lw=1, ls=":")
            if i == 0:
                ax.text(0.03, 0.97, tl[t], transform=ax.transAxes, va="top")
            if j == 0:
                ax.set_ylabel(f"n={n}: KL(KN) - KL(Katz) [nats/tree]")
            if i == 1:
                ax.set_xlabel("training trees M")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    save(fig, "kn_synthetic.png")


def landscape():
    df = pd.read_csv(os.path.join(RES, "discount_landscape.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    cols = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}
    for j, side in enumerate(("left", "right")):
        ax = axes[j]
        for n in (2, 3, 4):
            g = df[(df.n == n) & (df.rep == "root")]
            s = g[g.kind == "single"]
            ax.plot(s.x, s[side], color=cols[n], lw=2, label=f"n={n}, single D")
            mk = g[g.kind == "mkn"][side].iloc[0]
            ax.axhline(mk, color=cols[n], lw=1, ls="--")
        ax.set_xlabel("discount D")
        ax.set_ylabel(f"LOO NLL per equation, {side} model [nats]")
    axes[0].legend(title="dashed: Chen-Goodman", fontsize=8, title_fontsize=8)
    save(fig, "kn_discount_landscape.png")


def discrimination():
    df = pd.read_csv(os.path.join(RES, "discrimination.csv"))
    dc = pd.read_csv(os.path.join(RES, "discrimination_const.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, kind in zip(axes[:2], ("swap1", "relabel")):
        g = df[(df.kind == kind) & (df.rep == "root")]
        ns = sorted(g.n.unique())
        for k, m in enumerate(("katz", "kn_mkn", "kn_d")):
            h = g[g.model == m].groupby("n").auc
            ax.errorbar(np.arange(len(ns)) + (k - 1) * 0.15, h.mean(),
                        h.std(ddof=1) / np.sqrt(h.count()), fmt="o", color=C[m],
                        ms=6, capsize=2, label=LAB[m])
        ax.set_xticks(range(len(ns)), [f"n={n}" for n in ns])
        ax.set_ylabel("AUC (real vs corrupted)")
        ax.text(0.03, 0.03, {"swap1": "one operator swapped",
                             "relabel": "all operators relabelled"}[kind],
                transform=ax.transAxes)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    ax = axes[2]
    g = dc[(dc.kind == "constswap") & (dc.rep == "root") & (dc.n == 3)]
    for k, m in enumerate(("katz", "kn_d")):
        h = g[g.model == m].groupby("const").auc
        ax.errorbar(np.arange(2) + (k - 0.5) * 0.15, h.mean(),
                    h.std(ddof=1) / np.sqrt(h.count()), fmt="o", color=C[m], ms=6,
                    capsize=2, label=LAB[m])
    ax.axhline(0.5, color="0.5", lw=1)
    ax.set_xticks(range(2), ["constants as x", "constants as a"])
    ax.set_ylabel("AUC (real vs constant-variable swap)")
    ax.text(0.03, 0.97, f"n=3, root context\n{g['eq'].nunique()} eqs with a\n"
            "constant-variable sibling pair", transform=ax.transAxes, va="top")
    save(fig, "kn_discrimination.png")


def per_equation():
    df = pd.read_csv(os.path.join(RES, "cv_grouped.csv"))
    g = df[(df.n == 3) & (df.rep == "root")]
    piv = -g.pivot(index="eq", columns="model", values="total")
    fin = np.isfinite(piv).all(axis=1)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(piv.loc[fin, "katz"], piv.loc[fin, "kn_tuned"], s=14, color=C["kn_tuned"],
               edgecolor="white", lw=0.5)
    lim = [0, np.nanmax(piv[fin].values) * 1.05]
    ax.plot(lim, lim, color="0.5", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Katz held-out NLL [nats]")
    ax.set_ylabel("KN (CV-tuned) held-out NLL [nats]")
    ax.text(0.03, 0.97, f"n=3, root context\n{(~fin).sum()} eqs infinite under Katz",
            transform=ax.transAxes, va="top")
    save(fig, "kn_per_equation.png")


def benchmark():
    df = pd.read_csv(os.path.join(RES, "benchmark_results.csv"))
    benches = ["nguyen_8", "korns_1", "korns_4", "korns_6", "korns_7"]
    btitle = {"nguyen_8": "Nguyen-8: sqrt(x)", "korns_1": "Korns-1: a+bx",
              "korns_4": "Korns-4: a+b sin x", "korns_6": "Korns-6: a+b sqrt(x)",
              "korns_7": "Korns-7: a(1-exp(-bx))"}
    pri = {"mdl": ("#7d7c78", "MDL (k log n)"), "katz2": ("#2a78d6", "Katz n=2 (paper)"),
           "kn2": ("#eb6834", "KN n=2"), "katz3r": ("#1baf7a", "Katz n=3 root"),
           "kn3r": ("#6250d6", "KN n=3 root"), "kn3rc": ("#e34948", "KN n=3 root + consts")}
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharex=True)
    for j, b in enumerate(benches):
        for k, (p, (col, lab)) in enumerate(pri.items()):
            g = df[(df.bench == b) & (df.prior == p)]
            if len(g) == 0:
                continue
            med = g.groupby("N").pref.median()
            axes[0, j].plot(med.index, med, color=col, marker="o", ms=4, lw=2, label=lab)
            top = g.assign(t=g["rank"] == 1).groupby("N").t.mean()
            axes[1, j].plot(top.index * (1 + 0.04 * (k - 2.5)), top, color=col, marker="o",
                            ms=4, lw=1.5, label=lab)
        axes[0, j].axhline(0, color="0.5", lw=1)
        axes[0, j].set_xscale("log")
        axes[0, j].text(0.03, 0.97, btitle[b], transform=axes[0, j].transAxes, va="top")
        axes[1, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_xlabel("number of data points N")
        for a in axes[:, j]:
            a.set_xscale("log")
            a.set_xticks([10, 100, 1000, 10000])
            a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0, 0].set_ylabel("DL(best other) - DL(truth) [nats]\n(median over seeds)")
    axes[1, 0].set_ylabel("fraction of seeds with truth ranked first")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.03))
    save(fig, "kn_esr_benchmark.png")


if __name__ == "__main__":
    nll_vs_n()
    synthetic()
    landscape()
    discrimination()
    per_equation()
    if os.path.isfile(os.path.join(RES, "benchmark_results.csv")):
        benchmark()
