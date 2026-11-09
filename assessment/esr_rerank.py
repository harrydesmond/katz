"""Experiment 5: how much does replacing Katz by Kneser-Ney change real ESR rankings?

Uses existing ESR outputs (read-only): every fitted function's -logL and parameter
codelength from combine_DL_comp*.dat. The function-structure term is replaced by
-log P(f) from priors trained on the full Physics corpus (constant terms sum log|c| are
common to all priors compared here and are omitted, so only differences in DL between
priors are meaningful, not absolute values).

Datasets:
    eos   : EoS_DE  output_de_core_maths_desy5_bao_cmb_v8, complexity 1-8 (raw, before that
            project's physical filters; used only as a test bed for ranking stability)
    panth : ESR example output_panth_dimful (Pantheon+), complexity 5

Locations: environment variables EOS_FIT_DIR (the EoS_DE output directory, not public) and
ESR_DIR (an ESR clone, default ./ESR in the repository root).
"""

import itertools
import time

from scipy.stats import spearmanr

from common import BASIS, RESULTS, ROOT, load_corpus, make_prior, np, os, pd
from katz.prior import KatzPrior

ESR_DIR = os.environ.get("ESR_DIR", os.path.join(ROOT, "ESR"))
SOURCES = {
    "eos": (os.environ.get("EOS_FIT_DIR", ""), range(1, 9)),
    "panth": (os.path.join(ESR_DIR, "esr", "fitting", "output", "output_panth_dimful"), [5]),
}
CONFIGS = {  # name: (n, smoothing, root_context, tune)
    "katz_n2": (2, "katz", False, False),
    "katz_n3_root": (3, "katz", True, False),
    "kn_n3_root": (3, "kn", True, False),
    "kn_n3_root_tuned": (3, "kn", True, True),
    "katz_n10_root": (10, "katz", True, False),
    "kn_n10_root": (10, "kn", True, False),
}


def load(source):
    d, comps = SOURCES[source]
    rows = []
    for c in comps:
        dl = np.atleast_2d(np.loadtxt(f"{d}/combine_DL_comp{c}.dat"))
        with open(f"{d}/combine_DL_fcn_comp{c}.dat") as f:
            fcn = [x.strip() for x in f.readlines()]
        for row, s in zip(dl, fcn):
            if s == "None" or not np.isfinite(row[0]):
                continue
            rows.append(dict(comp=c, fcn=s, DL_orig=row[0], negloglike=row[-3],
                             codelen_params=row[-2], codelen_fcn_orig=row[-1]))
    df = pd.DataFrame(rows).drop_duplicates("fcn").reset_index(drop=True)
    return df


def priors(fcns):
    out = {}
    eqs = load_corpus("physics")
    for (n, rc), names in itertools.groupby(
            sorted(CONFIGS, key=lambda k: CONFIGS[k][:3:2]), key=lambda k: CONFIGS[k][:3:2]):
        names = list(names)
        base = KatzPrior(n, BASIS, equations=eqs, root_context=rc)
        tuples = []
        for s in fcns:
            try:
                tuples.append(base.coder.process_all_equations(n + 1, [s], base.maxvar))
            except Exception:  # noqa: BLE001
                tuples.append(None)
        for name in names:
            _, sm, _, tune = CONFIGS[name]
            p = make_prior(n, sm, equations=eqs, root_context=rc)
            if tune:
                p.tune_discounts("single", k=10)
            out[name] = np.array([p.logprior_tuples(t) if t is not None else np.nan
                                  for t in tuples])
            print(f"   prior {name} done", flush=True)
    return pd.DataFrame(out)


def top_k(dl, fcn, k):
    """Index labels of the k best functions (lowest finite DL; ties broken by the function
    string, so the result is deterministic) and the number of functions outside the
    selection that tie with the k-th"""
    fin = np.isfinite(dl.to_numpy(dtype=float))
    d, f, idx = dl.to_numpy(dtype=float)[fin], fcn.to_numpy()[fin].astype(str), dl.index[fin]
    order = np.lexsort((f, d))
    sel = order[:k]
    ties = int(np.sum(d[order[k:]] == d[sel[-1]])) if len(sel) and len(order) > k else 0
    return set(idx[sel]), ties


def compare(df, a, b, top=(10, 50)):
    """Ranking agreement between DL variants a and b. Primary metrics use each prior's
    full ranking (functions with zero prior probability under a prior, DL = inf, are not
    admissible under it); 'shared_' metrics use only the functions finite under both.
    top{k}_overlap = |top-k(a) & top-k(b)| / min(k, n_a, n_b), where n is the number of
    admissible functions (NaN if none); top{k}_ties counts functions tied with the k-th
    function but left out, under a and b combined."""
    res = {}
    A_dl, B_dl = df[f"DL_{a}"], df[f"DL_{b}"]
    na, nb = int(np.isfinite(A_dl).sum()), int(np.isfinite(B_dl).sum())
    for t in top:
        A, ta = top_k(A_dl, df.fcn, t)
        B, tb = top_k(B_dl, df.fcn, t)
        m = min(t, na, nb)
        res[f"top{t}_overlap"] = len(A & B) / m if m else np.nan
        res[f"top{t}_ties"] = ta + tb
        res[f"top{t}_new_in_b"] = int(np.sum(np.isposinf(A_dl[list(B)])))  # zero prior under a
    wa, _ = top_k(A_dl, df.fcn, 1)
    wb, _ = top_k(B_dl, df.fcn, 1)
    res["winner_a"] = df.fcn[next(iter(wa))] if wa else None
    res["winner_b"] = df.fcn[next(iter(wb))] if wb else None
    res["same_winner"] = bool(wa and wa == wb)
    res["frac_zero_prior_a"] = float(np.mean(np.isposinf(A_dl)))  # NaN = unparseable
    res["frac_zero_prior_b"] = float(np.mean(np.isposinf(B_dl)))
    fin = np.isfinite(A_dl) & np.isfinite(B_dl)
    d = df[fin]
    res["shared_spearman"] = (spearmanr(d[f"DL_{a}"], d[f"DL_{b}"]).correlation
                              if len(d) > 1 else np.nan)
    for t in top:
        A, _ = top_k(d[f"DL_{a}"], d.fcn, t)
        B, _ = top_k(d[f"DL_{b}"], d.fcn, t)
        m = min(t, len(d))
        res[f"shared_top{t}_overlap"] = len(A & B) / m if m else np.nan
    sa, _ = top_k(d[f"DL_{a}"], d.fcn, 1)
    sb, _ = top_k(d[f"DL_{b}"], d.fcn, 1)
    res["shared_same_winner"] = bool(sa and sa == sb)
    return res


if __name__ == "__main__":
    t0 = time.time()
    summary = []
    for source in SOURCES:
        df = load(source)
        print(f"{source}: {len(df)} unique fitted functions", flush=True)
        lp = priors(df.fcn.tolist())
        df = pd.concat([df, lp], axis=1)
        for name in CONFIGS:  # zero prior probability (log-prior -inf) gives DL = inf
            df[f"DL_{name}"] = df.negloglike + df.codelen_params - df[name]
        df.to_csv(os.path.join(RESULTS, f"rerank_{source}.csv"), index=False)
        fin = np.isfinite(df[list(CONFIGS)]).all(axis=1)
        print(f"   prior finite for all configs: {fin.sum()}/{len(df)}")
        for a, b in [("katz_n2", "kn_n3_root"), ("katz_n3_root", "kn_n3_root"),
                     ("kn_n3_root", "kn_n3_root_tuned"), ("katz_n10_root", "kn_n10_root"),
                     ("katz_n2", "katz_n3_root")]:
            r = compare(df, a, b)
            dlp = (df[a.replace("DL_", "")] - df[b.replace("DL_", "")])[fin]
            r.update(dict(source=source, a=a, b=b, dlogprior_sd=dlp.std(),
                          spearman_logprior=spearmanr(df.loc[fin, a], df.loc[fin, b]
                                                      ).correlation))
            summary.append(r)
        for name in ["DL_orig"] + [f"DL_{c}" for c in CONFIGS]:
            top = df.nsmallest(5, name)
            print(f"   top-5 by {name}: " + "; ".join(
                f"{f} ({v - top[name].iloc[0]:.2f})" for f, v in zip(top.fcn, top[name])))
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(RESULTS, "rerank_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print(s.round(3).to_string(index=False))
    print(f"time {time.time() - t0:.1f}s")
