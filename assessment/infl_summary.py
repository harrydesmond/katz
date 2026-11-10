"""Tables for the inflation-corpus Katz vs KN tests: python infl_summary.py within|cross|discrim
Columns: held-out NLL per equation (finite under all models) and paired log-probability gains
'A-B' = mean of log p_A - log p_B (> 0: A better), with standard errors."""
import os
import sys

import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)


def table(df, keys):
    out = []
    for k, g in df.groupby(keys):
        piv = g.pivot(index="eq", columns="model", values="logp")
        fin = np.isfinite(piv).all(axis=1)
        row = dict(zip(keys, k), N=len(piv), katz_zero=int((~np.isfinite(piv["katz"])).sum()),
                   katznorm_zero=int((~np.isfinite(piv["katz_norm"])).sum()))
        for m in piv.columns:
            row[m] = -piv.loc[fin, m].mean()
        for ref in ("katz", "katz_norm"):  # gains vs the published and the normalised Katz
            for m in ("kn_fixed", "kn_mkn", "kn_tuned") + (("katz_norm",) if ref == "katz" else ()):
                d = (piv.loc[fin, m] - piv.loc[fin, ref]).to_numpy()
                tag = f"{m}-{ref}"
                row[tag] = d.mean()
                row[f"se({tag})"] = d.std(ddof=1) / np.sqrt(len(d))
                if m == "kn_tuned" and ref == "katz_norm":
                    row["frac_tuned_better_norm"] = np.mean(d > 0)
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "cross":
        df = pd.read_csv(os.path.join(R, "infl_cross.csv"))
        t = table(df, ["train", "test", "rep", "n"])
    elif what == "within":
        df = pd.read_csv(os.path.join(R, "infl_within.csv"))
        t = table(df, ["corpus", "cv", "rep", "n"])
    elif what == "discrim":
        df = pd.read_csv(os.path.join(R, "infl_discrimination.csv"))
        rows = []
        for k, g in df.groupby(["corpus", "kind", "n", "rep"]):
            piv = g.pivot(index="eq", columns="model", values="auc")
            row = dict(zip(["corpus", "kind", "n", "rep"], k))
            row.update({m: piv[m].mean() for m in piv.columns})
            for m in ("kn_fixed", "kn_mkn"):
                d = piv[m] - piv.katz_norm
                row[f"{m}-katz_norm"] = d.mean()
                row[f"se({m}-katz_norm)"] = d.std(ddof=1) / np.sqrt(len(d))
            rows.append(row)
        t = pd.DataFrame(rows)
    print(t.round(3 if what == "discrim" else 2).to_string(index=False))
