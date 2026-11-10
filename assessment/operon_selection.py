"""Does the function prior help in Operon's final selection? Compares, for the same runs (arms
A-C of operon_bloat.py, final populations saved), the model with the minimum description length
under the KN prior (`solved`) with the one under ESR's default k log n function codelength
(`mdl_solved`), and checks that the rerun reproduces the selections of the first campaign.

Sensitivity check (`*_zero` columns): ESR's rule for unresolved parameters. A parameter of a
selected model with |theta_i| < Delta_i = sqrt(12 / F_ii) (F the Fisher information of the data)
is set to zero, the other parameters are refitted, and the reduced model is tested with the same
structural truth test; e.g. sqrt(a0 + a1*x) with a0 ~ 0 becomes sqrt(a1*x).

    python3 operon_selection.py      # reads results/operon_pop.csv and results/operon_bloat.csv;
                                     # writes results/operon_selection.csv
"""

import ast
import os
import warnings

import numpy as np
import pandas as pd
import scipy.optimize
import sympy
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
KEY = ["arm", "bench", "N", "dseed", "gseed"]
A = [sympy.Symbol(f"a{i}", real=True) for i in range(40)]


def mcnemar(a, b):
    """Exact McNemar test on paired booleans: (a only, b only, p)"""
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0)


def set_unresolved_to_zero(model, theta, x, y, sig):
    """ESR's rule: parameters with |theta_i| < sqrt(12/F_ii) are set to zero and the others
    refitted by least squares. Returns (reduced model string, its parameters, names set to zero),
    or None if every parameter is resolved."""
    import operon_bloat as ob

    e = sympy.sympify(model, locals={"x0": ob.X, **{f"a{i}": A[i] for i in range(40)}})
    th = np.asarray(theta, float)
    ps = A[: len(th)]
    small = []
    for i, p in enumerate(ps):
        d = sympy.lambdify([ob.X] + ps, sympy.diff(e, p), modules="numpy")
        with np.errstate(all="ignore"):
            g = np.broadcast_to(np.asarray(d(x, *th), float), x.shape)
        F = np.sum(g**2) / sig**2
        if np.isfinite(F) and F > 0 and abs(th[i]) < np.sqrt(12 / F):
            small.append(i)
    if not small:
        return None
    e = e.subs({ps[i]: 0 for i in small})
    keep = [i for i in range(len(th)) if i not in small and ps[i] in e.free_symbols]
    e = e.xreplace({ps[i]: A[j] for j, i in enumerate(keep)})
    th = th[keep]
    if len(th):
        f = sympy.lambdify([ob.X] + A[: len(th)], e, modules="numpy")

        def resid(p):
            with np.errstate(all="ignore"):
                v = np.broadcast_to(np.asarray(f(x, *p), float), x.shape)
            return (y - np.where(np.isfinite(v), v, 1e10)) / sig

        th = scipy.optimize.least_squares(resid, th).x
    return str(e), th.tolist(), [f"a{i}" for i in small]


def zero_setting_verdicts(df):
    """Truth verdicts of both selections after setting unresolved parameters to zero"""
    import operon_bloat as ob
    from esr_benchmark import mock_data

    warnings.filterwarnings("ignore")
    out = {"solved_zero": [], "mdl_solved_zero": [], "reduced": [], "mdl_reduced": []}
    for _, r in df.iterrows():
        x, y, sig = mock_data(r.bench, int(r.N), int(r.dseed))
        for pre, col in (("sel_", ""), ("mdl_sel_", "mdl_")):
            solved, red = bool(r[col + "solved"]), ""
            if not solved:
                s = set_unresolved_to_zero(r[pre + "model"], ast.literal_eval(r[pre + "theta"]),
                                           x, y, sig)
                if s is not None:
                    red = f"{','.join(s[2])} = 0: {s[0]}"
                    solved = bool(ob.is_truth_structural(s[0], s[1], r.bench))
            out[col + "solved_zero"].append(solved)
            out[col + "reduced"].append(red)
    return df.assign(**out)


def table(df, a, b):
    t = df.groupby("arm")[[a, b]].mean()
    for arm, g in df.groupby("arm"):
        x, y, p = mcnemar(g[a], g[b])
        t.loc[arm, "prior_only"], t.loc[arm, "kln_only"], t.loc[arm, "p"] = x, y, p
    x, y, p = mcnemar(df[a], df[b])
    t.loc["all"] = [df[a].mean(), df[b].mean(), x, y, p]
    return t.round(3)


def main():
    pd.set_option("display.width", 200)
    df = pd.read_csv(os.path.join(R, "operon_pop.csv"))
    df["solved"] = df.solved.astype(bool)
    df["mdl_solved"] = df.mdl_solved.astype(bool)
    print(f"{len(df)} runs:", df.groupby("arm").size().to_dict(),
          "| same model selected by both:", round(df.same_selection.mean(), 3))

    print("\n== truth selected (structural test): with the KN prior vs with k log n ==")
    print(table(df, "solved", "mdl_solved"))
    print("\n== by benchmark and N (prior / k log n) ==")
    g = df.groupby(["bench", "N", "arm"])[["solved", "mdl_solved"]].mean().unstack("arm")
    print(g.round(2))
    print("\n== selected model size (simplified nodes) and test NMSE, medians ==")
    print(df.groupby("arm")[["sel_n_simplified", "mdl_sel_n_simplified", "test_nmse",
                             "mdl_test_nmse"]].median().round(4))

    df = zero_setting_verdicts(df)
    print("\n== sensitivity: unresolved parameters set to zero (ESR's rule) ==")
    print(table(df, "solved_zero", "mdl_solved_zero"))
    print(df.groupby(["bench", "N"])[["solved_zero", "mdl_solved_zero"]].mean().round(2))
    rescued = df[df.solved_zero & ~df.solved]
    print("prior selections rescued:", len(rescued), "| k log n selections rescued:",
          int((df.mdl_solved_zero & ~df.mdl_solved).sum()))
    print(rescued.reduced.str.replace(r"^.*: ", "", regex=True).value_counts().to_dict())
    left = df[df.mdl_solved_zero & ~df.solved_zero]
    print("still found only with k log n:", left.groupby("bench").size().to_dict())
    cols = KEY + ["solved", "mdl_solved", "solved_zero", "mdl_solved_zero", "sel_model",
                  "mdl_sel_model", "reduced", "mdl_reduced"]
    df[cols].to_csv(os.path.join(R, "operon_selection.csv"), index=False)

    f = os.path.join(R, "operon_bloat.csv")
    if os.path.isfile(f):
        old = pd.read_csv(f)
        j = df.merge(old[KEY + ["sel_model", "solved"]], on=KEY, suffixes=("", "_old"))
        print(f"\n== reproducibility vs the first campaign: {len(j)} matched runs ==")
        print("   same selected model:", int((j.sel_model == j.sel_model_old).sum()),
              "| same truth verdict:", int((j.solved == j.solved_old.astype(bool)).sum()))


if __name__ == "__main__":
    main()
