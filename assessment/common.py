"""Shared utilities for assessing Kneser-Ney vs Katz function priors.

All evaluations are done at the level of the n-tuples produced by SymbolCoder, so each
equation is parsed once (sympy parsing dominates the cost) and models are refitted on
subsets of equations with KatzPrior.fit.
"""

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from katz.prior import KatzPrior  # noqa: E402

warnings.filterwarnings("ignore")

BASIS = [
    ["a", "x"],
    ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
    ["+", "-", "*", "/", "pow"],
]
CACHE = os.path.join(ROOT, "assessment", "cache")
RESULTS = os.path.join(ROOT, "assessment", "results")
PLOTS = os.path.join(ROOT, "Plots")
for d in (CACHE, RESULTS, PLOTS):
    os.makedirs(d, exist_ok=True)


def load_corpus(name="physics"):
    """
    Equation strings of a corpus.

    'physics' : all 161 equations (100 Feynman + 20 bonus + 41 Wikipedia)
    'feynman' : the 100 Feynman-lectures equations
    'other'   : the 61 non-Feynman equations (bonus + Wikipedia)
    """
    phys = pd.read_csv(os.path.join(ROOT, "data", "NewPhysics.csv"))
    feyn = set(pd.read_csv(os.path.join(ROOT, "data", "NewFeynman.csv"))["Filename"])
    is_feyn = phys["Filename"].isin(feyn).to_numpy()
    eqs = phys["New Formula"].to_numpy()
    if name == "physics":
        return list(eqs)
    if name == "feynman":
        return list(eqs[is_feyn])
    if name == "other":
        return list(eqs[~is_feyn])
    raise ValueError(name)


def make_prior(n, smoothing="katz", equations=("sin(x0) + x1",), **kw):
    """KatzPrior on a (by default trivial) corpus; refit later with .fit(tuples).
    In these scripts 'kn' without smoothing_kwargs means the Chen-Goodman (mkn) discounts."""
    if smoothing == "kn" and kw.get("smoothing_kwargs") is None:
        kw["smoothing_kwargs"] = {"discounts": "mkn"}
    return KatzPrior(n, BASIS, equations=list(equations), smoothing=smoothing, **kw)


def corpus_tuples(name, n, code_constants=False, root_context=False):
    """Per-equation n-tuples of a corpus (cached on disk)"""
    fname = os.path.join(
        CACHE, f"tuples_{name}_n{n}_c{int(code_constants)}_r{int(root_context)}.pkl"
    )
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
    eqs = load_corpus(name)
    kp = KatzPrior(
        n,
        BASIS,
        equations=eqs,
        code_constants=code_constants,
        root_context=root_context,
    )
    out = kp.eq_tuples
    with open(fname, "wb") as f:
        pickle.dump(out, f)
    return out


def score_tuples(prior, t):
    """
    Log-probabilities of one equation, split into root, left (non-root first
    children) and right (second children) contributions, plus event count
    """
    pl, pr = prior.event_probs(t)
    with np.errstate(divide="ignore"):
        ll, lr = np.log(pl), np.log(pr)
    return dict(
        root=ll[0], left=ll[1:].sum(), right=lr.sum(), total=ll.sum() + lr.sum(),
        nev=len(pl) + len(pr),
    )


def kfold(N, k, seed):
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(N), k)


def heldout(prior, eq_tuples, folds, train_pool=None, test_tuples=None):
    """
    Held-out log-probabilities of every equation.

    If test_tuples is None: cross-validation over eq_tuples with the given folds.
    Otherwise: fit once on train_pool and score test_tuples.

    Returns a DataFrame with one row per held-out equation.
    """
    rows = []
    if test_tuples is not None:
        prior.fit(train_pool)
        for i, t in enumerate(test_tuples):
            rows.append(dict(eq=i, **score_tuples(prior, t)))
        return pd.DataFrame(rows)
    N = len(eq_tuples)
    for f in folds:
        fs = set(f.tolist())
        prior.fit([eq_tuples[i] for i in range(N) if i not in fs])
        for i in f:
            rows.append(dict(eq=int(i), **score_tuples(prior, eq_tuples[i])))
    return pd.DataFrame(rows).sort_values("eq").reset_index(drop=True)


def paired_summary(a, b):
    """Mean and standard error of the paired difference b - a (finite pairs only)"""
    d = np.asarray(b) - np.asarray(a)
    m = np.isfinite(d)
    d = d[m]
    return dict(
        mean=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)), frac_pos=np.mean(d > 0),
        n=len(d),
    )
