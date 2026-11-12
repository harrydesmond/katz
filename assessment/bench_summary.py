"""Summary statistics of the ESR prior benchmark (results/benchmark_results.csv, written by
esr_benchmark.py collect) and comparison with the old likelihood-grouping truth test
(results/legacy_truth.csv, optional).

    python3 bench_summary.py
"""

import os
import warnings

import pandas as pd
from scipy.stats import binomtest

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
P_ALL = ["mdl", "katz2", "katz2n", "katz2rn", "kn2", "katz3", "kn3", "katz3r", "katz3rn", "kn3r",
         "kn3rc"]
# paired comparisons: smoothing at fixed n and context, root context at n = 3, n without root
# context, n plus root context, constant coding, and every prior against the MDL baseline
PAIRS = [("kn2", "katz2"), ("kn3", "katz3"), ("kn3r", "katz3r"),
         ("katz3r", "katz3"), ("kn3r", "kn3"), ("katz3", "katz2"), ("kn3", "kn2"),
         ("katz3r", "katz2"), ("kn3r", "kn2"), ("kn3rc", "kn3r"),
         # normalised Katz (katz*n): normalisation alone, root context and n with it, and the
         # new Katz default (root context, normalised) against the paper's settings
         ("katz2n", "katz2"), ("katz3rn", "katz3r"), ("katz2rn", "katz2n"), ("katz3rn", "katz2rn"),
         ("katz2rn", "katz2"), ("katz3rn", "katz2"), ("kn3r", "katz3rn"),
         ("katz2", "mdl"), ("kn2", "mdl"), ("katz3", "mdl"), ("kn3", "mdl"), ("katz3r", "mdl"),
         ("kn3r", "mdl"), ("kn3rc", "mdl"), ("katz2n", "mdl"), ("katz2rn", "mdl"),
         ("katz3rn", "mdl")]


def mcnemar(w, a, b):
    """Exact McNemar test on paired truth-first outcomes: (a only, b only, p)"""
    b10, b01 = int((w[a] & ~w[b]).sum()), int((~w[a] & w[b]).sum())
    p = binomtest(b10, b10 + b01).pvalue if b10 + b01 else 1.0
    return b10, b01, p


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    df = pd.read_csv(os.path.join(R, "benchmark_results.csv"))
    df["top"] = df.top.astype(float)
    df["stop"] = (df.rank_strict == 1).astype(float)
    m = df[df.bench != "korns_4_213"]
    P = [p for p in P_ALL if p in set(df.prior)]
    print("rows", len(df), "| main", len(m), "| truth not found", int(m.pref_strict.isna().sum()),
          "| top undetermined", int(m.top.isna().sum()))

    for col, lab in (("top", "functional"), ("stop", "strict (listed variants only)")):
        t = m.groupby(["bench", "prior"])[col].mean().unstack("prior")[P]
        t.loc["ALL"] = m.groupby("prior")[col].mean()[P]
        print(f"\n== fraction of mocks with the truth ranked first, {lab} ==")
        print(t.round(2))

    w = m.pivot_table(index=["bench", "N", "seed"], columns="prior", values="top",
                      aggfunc="first").astype(bool)
    wx = w[w.index.get_level_values(0) != "nguyen_8"]
    print("\n== paired comparisons (exact McNemar): a only : b only, p ==")
    for a, b in [(a, b) for a, b in PAIRS if a in P and b in P]:
        x, y, p = mcnemar(w, a, b)
        x2, y2, p2 = mcnemar(wx, a, b)
        print(f"   {a:6s} vs {b:6s}: {x:2d}:{y:2d}  p={p:.3g}   | without Nguyen-8 {x2}:{y2} p={p2:.3g}")

    print("\n== median DL(best other) - DL(truth) [nats] by N ==")
    print(m.groupby(["prior", "N"]).pref.median().unstack("N").loc[P].round(2))
    for b in sorted(m.bench.unique()):
        print(f"\n== {b}: fraction truth first by N ==")
        print(m[m.bench == b].groupby(["prior", "N"]).top.mean().unstack("N").loc[P].round(1))

    print("\n== best function is a constant or x:",
          m.groupby("prior").best_is_null.mean()[P].round(3).to_dict())
    print("== unlisted equivalents of the truth found in", int((m.n_equiv > 0).sum()), "of", len(m),
          "rows; strict and functional verdicts differ in", int((m.top != m.stop).sum()))
    eq = m[m.n_equiv > 0].groupby("bench").equiv.apply(
        lambda v: sorted(set("; ".join(v).split("; "))))
    for b, v in eq.items():
        print(f"   {b}: {len(v)} distinct, e.g. {v[:5]}")

    k = df[df.bench.isin(["korns_4", "korns_4_213"])]
    kk = k.pivot_table(index=["N", "seed", "prior"], columns="bench", values=["top", "pref"],
                       aggfunc="first").dropna()
    if len(kk):
        print("\n== Korns-4 offset -2.3 vs -2.13 (paired mocks, N = 10, 100, 1000) ==")
        print(kk["top"].groupby(level="prior").mean().loc[P].round(2))
        d = kk["pref"]["korns_4_213"] - kk["pref"]["korns_4"]
        print("   pref(-2.13) - pref(-2.3), median:",
              d.groupby(level="prior").median().loc[P].round(2).to_dict())
        print("   same truth-first verdict in", int((kk["top"]["korns_4"] == kk["top"]["korns_4_213"]).sum()),
              "of", len(kk), "pairs")

    f = os.path.join(R, "legacy_truth.csv")
    if os.path.isfile(f):
        j = pd.read_csv(f).merge(m, on=["bench", "N", "seed", "prior"])
        j["old_top"] = (j.old_rank == 1).astype(float)
        diff = j[j.old_top != j.top]
        print(f"\n== old likelihood grouping vs new truth test: {len(j)} rows, verdicts differ in "
              f"{len(diff)} ==")
        print(j.groupby("prior")[["old_top", "top"]].mean().loc[[p for p in P if p in set(j.prior)]]
              .round(3))
        if len(diff):
            print(diff[["bench", "N", "seed", "prior", "old_pref", "pref", "best_other", "old_best"]]
                  .to_string(index=False))


if __name__ == "__main__":
    main()
