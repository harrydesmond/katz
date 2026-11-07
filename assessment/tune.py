"""Experiment 2: calibrating the Kneser-Ney discounts.

(a) Landscape: cross-validated held-out NLL of the left and right models as a function of
    (i) a single discount D used at every order and count, and (ii) a global scale s
    multiplying the Chen-Goodman (MKN) estimates.
(b) Nested cross-validation: discounts tuned on inner folds (per-order single discount,
    scale of MKN, or count-dependent discounts D1, D2, D3+ shared by all orders), evaluated
    on outer folds, compared with Katz and untuned KN on the same outer folds.

The left and right models are independent, so their discounts are tuned separately.
"""

import sys
import time

import scipy.optimize

from common import RESULTS, corpus_tuples, kfold, make_prior, np, os, pd


class KNFolds:
    """Kneser-Ney left/right models fitted on the training part of each fold, with the
    held-out prediction events of each fold, so that discounts can be changed cheaply"""

    def __init__(self, prior, eq_tuples, folds, train_idx=None):
        self.models = []
        self.events = []
        idx = np.arange(len(eq_tuples)) if train_idx is None else np.asarray(train_idx)
        for f in folds:
            test = set(idx[f].tolist())
            prior.fit([eq_tuples[i] for i in idx if i not in test])
            left, right = [], []
            for i in idx[f]:
                ll, rr = prior.split_events(eq_tuples[i])
                left += [(tt[-1], tt[:-1]) for tt in ll]
                right += [(tt[-1], tt[-(prior.n + 1) : -1]) for tt in rr]
            self.models.append((prior.backoff_left, prior.backoff_right))
            self.events.append((left, right))

    def nll(self, side, discounts="mkn", scale=1.0):
        """Total held-out negative log-probability of one side (0=left, 1=right)"""
        tot = 0.0
        for models, events in zip(self.models, self.events):
            m = models[side]
            m.set_discounts(discounts, scale)
            for w, h in events[side]:
                tot -= np.log(m.prob(w, h))
        return tot


def _to_discounts(theta):
    return 1.0 / (1.0 + np.exp(-np.asarray(theta)))  # in (0, 1)


_COUNT_MAX = np.array([1.0, 2.0, 3.0])  # D_r <= r


def _count_discounts(theta, order):
    """(D1, D2, D3+) from unconstrained parameters, the same at every order. Two parameters
    mean D1 and D2+ (= D3+)."""
    th = np.asarray(theta, dtype=float)
    if th.size == 2:
        th = np.r_[th, th[1]]
        d = _COUNT_MAX[[0, 1, 1]] / (1.0 + np.exp(-th))
    else:
        d = _COUNT_MAX / (1.0 + np.exp(-th))
    return np.tile(d, (order, 1))


def _logit(p):
    return np.log(p / (1.0 - p))


def tune(kf, side, param, order):
    """Minimise the held-out NLL over the discount parameters; returns kwargs for KneserNey"""
    if param == "scale":
        res = scipy.optimize.minimize_scalar(
            lambda s: kf.nll(side, "mkn", s), bounds=(0.05, 3.0), method="bounded",
            options=dict(xatol=1e-3),
        )
        return dict(discounts="mkn", discount_scale=float(res.x))
    if param == "single":
        res = scipy.optimize.minimize_scalar(
            lambda d: kf.nll(side, float(d)), bounds=(1e-3, 1.0), method="bounded",
            options=dict(xatol=1e-3),
        )
        return dict(discounts=float(res.x))
    if param == "perorder":
        x0 = np.zeros(order)  # D = 0.5 everywhere
        res = scipy.optimize.minimize(
            lambda th: kf.nll(side, _to_discounts(th)), x0, method="Nelder-Mead",
            options=dict(xatol=1e-2, fatol=1e-3, maxiter=400 * order),
        )
        return dict(discounts=_to_discounts(res.x).tolist())
    if param in ("count2", "count3"):
        k = int(param[-1])
        x0 = _logit(0.8 / _COUNT_MAX[[0, 1, 1][:k] if k == 2 else [0, 1, 2]])  # D = 0.8
        res = scipy.optimize.minimize(
            lambda th: kf.nll(side, _count_discounts(th, order)), x0, method="Nelder-Mead",
            options=dict(xatol=1e-2, fatol=1e-3, maxiter=600),
        )
        return dict(discounts=_count_discounts(res.x, order).tolist())
    raise ValueError(param)


def landscape(n, rep_kw, corpus="physics"):
    T = corpus_tuples(
        corpus, n, rep_kw.get("code_constants", False), rep_kw.get("root_context", False)
    )
    prior = make_prior(n, "kn", **rep_kw)
    kf = KNFolds(prior, T, [np.array([i]) for i in range(len(T))])
    rows = []
    for D in np.linspace(0.05, 1.0, 20):
        rows.append(dict(kind="single", x=D, left=kf.nll(0, D), right=kf.nll(1, D)))
    for s in np.linspace(0.1, 2.0, 20):
        rows.append(
            dict(kind="scale", x=s, left=kf.nll(0, "mkn", s), right=kf.nll(1, "mkn", s))
        )
    rows.append(dict(kind="mkn", x=1.0, left=kf.nll(0, "mkn"), right=kf.nll(1, "mkn")))
    rows.append(dict(kind="kn", x=1.0, left=kf.nll(0, "kn"), right=kf.nll(1, "kn")))
    out = pd.DataFrame(rows)
    for c in ("left", "right"):
        out[c] /= len(T)  # per equation
    return out


def nested(n, rep_kw, params=("scale", "single", "perorder"), corpus="physics",
           k_outer=10, k_inner=10, seed=0):
    """Outer k-fold; for every outer fold tune on inner k-fold of the training equations"""
    T = corpus_tuples(
        corpus, n, rep_kw.get("code_constants", False), rep_kw.get("root_context", False)
    )
    N = len(T)
    outer = kfold(N, k_outer, seed)
    prior_kn = make_prior(n, "kn", **rep_kw)
    prior_katz = make_prior(n, "katz", **rep_kw)
    rows = []
    chosen = []
    for fi, f in enumerate(outer):
        train = np.array([i for i in range(N) if i not in set(f.tolist())])
        inner = kfold(len(train), k_inner, seed + 100 + fi)
        kf = KNFolds(prior_kn, T, inner, train_idx=train)
        settings = {"kn_mkn": ({"discounts": "mkn"}, {"discounts": "mkn"})}
        for p in params:
            settings[f"kn_{p}"] = tuple(tune(kf, side, p, n + 1) for side in (0, 1))
            chosen.append(dict(fold=fi, param=p, left=settings[f"kn_{p}"][0],
                               right=settings[f"kn_{p}"][1]))
        prior_katz.fit([T[i] for i in train])
        for i in f:
            pl, pr = prior_katz.event_probs(T[i])
            with np.errstate(divide="ignore"):
                rows.append(dict(eq=int(i), model="katz", left=-np.log(pl).sum(),
                                 right=-np.log(pr).sum()))
        prior_kn.fit([T[i] for i in train])
        for name, (kl, kr) in settings.items():
            prior_kn.backoff_left.set_discounts(**kl)
            prior_kn.backoff_right.set_discounts(**kr)
            for i in f:
                pl, pr = prior_kn.event_probs(T[i])
                rows.append(dict(eq=int(i), model=name, left=-np.log(pl).sum(),
                                 right=-np.log(pr).sum()))
    df = pd.DataFrame(rows)
    df["total"] = df.left + df.right
    return df, chosen


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "landscape"
    reps = {"base": {}, "root": {"root_context": True}}
    t0 = time.time()
    if which == "landscape":
        out = []
        for n in (2, 3, 4):
            for rname, rkw in reps.items():
                df = landscape(n, rkw)
                df["n"], df["rep"] = n, rname
                out.append(df)
        out = pd.concat(out, ignore_index=True)
        out.to_csv(os.path.join(RESULTS, "discount_landscape.csv"), index=False)
        for (n, r), g in out.groupby(["n", "rep"]):
            best = {c: g[g.kind == "single"].loc[g[g.kind == "single"][c].idxmin()]
                    for c in ("left", "right")}
            mk = g[g.kind == "mkn"].iloc[0]
            print(f"n={n} {r:5s} | single-D optimum: left D={best['left'].x:.2f} "
                  f"({best['left'].left:.3f}) right D={best['right'].x:.2f} "
                  f"({best['right'].right:.3f}) | MKN: left {mk.left:.3f} right {mk.right:.3f}")
    else:
        n = int(sys.argv[2])
        rname = sys.argv[3]
        # optional: comma-separated tuning modes and an output-file suffix
        params = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else (
            "scale", "single", "perorder")
        suffix = f"_{sys.argv[5]}" if len(sys.argv) > 5 else ""
        df, chosen = nested(n, reps[rname], params=params)
        df.to_csv(os.path.join(RESULTS, f"nested_n{n}_{rname}{suffix}.csv"), index=False)
        pd.DataFrame(chosen).to_csv(
            os.path.join(RESULTS, f"nested_n{n}_{rname}{suffix}_chosen.csv"), index=False)
        piv = df.pivot(index="eq", columns="model", values="total")
        fin = np.isfinite(piv).all(axis=1)
        print(f"n={n} {rname}: {(~fin).sum()} eqs infinite under some model; "
              f"mean NLL over {fin.sum()}:")
        for m in piv.columns:
            d = piv.loc[fin, "katz"] - piv.loc[fin, m]
            print(f"   {m:12s} NLL {piv.loc[fin, m].mean():.3f}   gain vs katz "
                  f"{d.mean():+.3f} +- {d.std(ddof=1) / np.sqrt(fin.sum()):.3f}"
                  f"  (better on {np.mean(d > 0):.2f})")
    print(f"time {time.time() - t0:.1f}s")
