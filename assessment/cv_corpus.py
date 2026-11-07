"""Experiment 1: held-out (leave-one-equation-out) log-likelihood of real corpus equations.

The held-out negative log-probability of an equation is the structural codelength
(nats) that the prior would assign to a new physics equation, i.e. exactly the quantity
the function prior contributes to the MDL/Bayesian model selection. Lower is better.

Outputs: assessment/results/cv_grid.csv (one row per equation x configuration)
"""

import itertools
import time

from common import (
    RESULTS,
    corpus_tuples,
    heldout,
    make_prior,
    np,
    os,
    paired_summary,
    pd,
)

REPS = {
    "base": dict(),
    "root": dict(root_context=True),
    "parent": dict(right_backoff="parent"),
    "root+parent": dict(root_context=True, right_backoff="parent"),
    "const": dict(code_constants=True),
    "const+root+parent": dict(
        code_constants=True, root_context=True, right_backoff="parent"
    ),
}
MODELS = {
    "katz": ("katz", None),
    "kn_mkn": ("kn", dict(discounts="mkn")),
    "kn_y": ("kn", dict(discounts="kn")),
}


def run_grid(corpus="physics", ns=(1, 2, 3, 4)):
    rows = []
    for n, (rep, rkw) in itertools.product(ns, REPS.items()):
        T = corpus_tuples(
            corpus,
            n,
            code_constants=rkw.get("code_constants", False),
            root_context=rkw.get("root_context", False),
        )
        folds = [np.array([i]) for i in range(len(T))]  # leave-one-out
        for mname, (sm, skw) in MODELS.items():
            prior = make_prior(n, sm, smoothing_kwargs=skw, **rkw)
            df = heldout(prior, T, folds)
            df["n"], df["rep"], df["model"], df["corpus"] = n, rep, mname, corpus
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def summarise(df):
    out = []
    for (n, rep), g in df.groupby(["n", "rep"]):
        piv = g.pivot(index="eq", columns="model", values="total")
        row = dict(n=n, rep=rep, N=len(piv))
        for m in MODELS:
            row[f"ninf_{m}"] = int((~np.isfinite(piv[m])).sum())
        fin = np.isfinite(piv).all(axis=1)
        for m in MODELS:
            row[f"NLL_{m}"] = -piv.loc[fin, m].mean()
        for m in ("kn_mkn", "kn_y"):
            s = paired_summary(-piv[m], -piv["katz"])  # >0 means KN better
            row[f"gain_{m}"] = s["mean"]
            row[f"gain_{m}_se"] = s["se"]
            row[f"fracbetter_{m}"] = s["frac_pos"]
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    t0 = time.time()
    df = run_grid("physics")
    df.to_csv(os.path.join(RESULTS, "cv_grid.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(RESULTS, "cv_grid_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summ.round(3).to_string(index=False))
    print(f"time {time.time() - t0:.1f}s")
