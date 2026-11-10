"""Experiment 1b: out-of-source generalisation and application-scale context lengths.

(a) Train on the 100 Feynman-lectures equations, test on the 61 equations from other
    sources (20 textbook 'bonus' equations + 41 Wikipedia equations). KN discounts are
    tuned by 10-fold CV on the training equations only (KatzPrior.tune_discounts).
(b) Leave-one-out on all 161 equations for n = 8, 10 (the values used in the halo-profile
    and inflation applications), Katz vs untuned KN.
"""

import sys
import time

from common import RESULTS, corpus_tuples, heldout, make_prior, np, os, pd


def out_of_source(ns=(1, 2, 3, 4, 6, 8, 10), train="feynman", test="other"):
    rows = []
    for n in ns:
        for rep, rkw in (("base", {}), ("root", {"root_context": True})):
            Ttr = corpus_tuples(train, n, root_context=rkw.get("root_context", False))
            Tte = corpus_tuples(test, n, root_context=rkw.get("root_context", False))
            for name in ("katz", "katz_norm", "kn_mkn", "kn_tuned"):
                p = (make_prior(n, "katz", smoothing_kwargs={"normalise": True}, **rkw)
                     if name == "katz_norm" else
                     make_prior(n, "katz" if name == "katz" else "kn", **rkw))
                p.eq_tuples = Ttr
                if name == "kn_tuned":
                    p.tune_discounts("single", k=10)
                df = heldout(p, None, None, train_pool=Ttr, test_tuples=Tte)
                df["n"], df["rep"], df["model"] = n, rep, name
                rows.append(df)
    return pd.concat(rows, ignore_index=True)


def large_n(ns=(8, 10)):
    rows = []
    for n in ns:
        T = corpus_tuples("physics", n, root_context=True)
        folds = [np.array([i]) for i in range(len(T))]
        for name in ("katz", "kn_mkn"):
            p = make_prior(n, "katz" if name == "katz" else "kn", root_context=True)
            df = heldout(p, T, folds)
            df["n"], df["rep"], df["model"] = n, "root", name
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def table(df):
    out = []
    for (n, rep), g in df.groupby(["n", "rep"]):
        piv = g.pivot(index="eq", columns="model", values="total")
        fin = np.isfinite(piv).all(axis=1)
        row = dict(n=n, rep=rep, N=len(piv), n_finite=int(fin.sum()))
        for m in piv.columns:
            row[f"ninf_{m}"] = int((~np.isfinite(piv[m])).sum())
            row[f"NLL_{m}"] = -piv.loc[fin, m].mean()
            if m != "katz":
                d = piv.loc[fin, m] - piv.loc[fin, "katz"]
                row[f"gain_{m}"] = d.mean()
                row[f"gain_{m}_se"] = d.std(ddof=1) / np.sqrt(fin.sum())
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "reverse":  # train on the other 61, test on Feynman
        a = out_of_source(train="other", test="feynman")
        a.to_csv(os.path.join(RESULTS, "out_of_source_reverse.csv"), index=False)
        print(table(a).round(3).to_string(index=False))
        print(f"time {time.time() - t0:.1f}s")
        sys.exit()
    a = out_of_source()
    a.to_csv(os.path.join(RESULTS, "out_of_source.csv"), index=False)
    print("Out-of-source (train Feynman 100 -> test other 61):")
    print(table(a).round(3).to_string(index=False))
    b = large_n()
    b.to_csv(os.path.join(RESULTS, "loo_large_n.csv"), index=False)
    print("LOO, root context, large n:")
    print(table(b).round(3).to_string(index=False))
    print(f"time {time.time() - t0:.1f}s")
