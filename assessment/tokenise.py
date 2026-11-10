"""Experiment: adaptive tokenisation of expression trees.

The package codes every equation as a tree over a fixed, hand-chosen alphabet (the operator
basis plus the leaves x, y). Here the tokens are instead learned from the training corpus:

  bin2  tree byte-pair encoding (BPE) on the coded binary trees: repeatedly merge the most
        frequent (parent token, child slot, child token) pair into a new token whose arity
        (number of open slots) is at most 2, so the result is still a unary/binary tree;
  binN  the same with no arity limit (e.g. *(_,*(_,_)) becomes one ternary token);
  flat  chains of + and * are first flattened into n-ary nodes whose token carries the
        number of arguments ('*3'), removing the bracketing imposed by sympy's binarisation;
        BPE (no arity limit) is then applied on top.

M = 0 of bin2/binN is the package representation, and M = 0 of flat is pure flattening. A
merged tree expands back to the (flattened) coded tree uniquely, so -log P(tree) is a valid
codelength for every tokenisation. The deterministic tokenisation wastes the probability of
non-canonical token trees, so the code is valid but not optimal.

The prediction events generalise SymbolCoder: the root and every first child are predicted by
the left model from the n nearest ancestors; every later child by the right model from the
n - 1 nearest ancestors and its previous sibling. The Kneser-Ney discounts (one per side, for
each n and M) and M itself are chosen by inner 5-fold CV on the training equations of each
outer leave-one-family-out fold, with the merges re-learned on every inner training split,
exactly as 'kn_tuned' in cv_grouped.py (which M = 0 reproduces).

Corpora: 'physics' (161 equations, families as in cv_grouped.py) and the inflationary
potentials 'A', 'B' and 'M' of infl_common.py (each in its own basis, families as in
infl_cv.py, whose root-context leave-one-family-out 'kn_tuned' M = 0 reproduces).

Usage (from assessment/; CORPORA is a comma-separated list, default physics):
    python3 tokenise.py verify [CORPORA]    # checks against the package (laptop, ~1 min)
    python3 tokenise.py screen [CORPORA]    # fixed D = 0.8, 10 grouped folds, n = 3 (laptop)
    python3 tokenise.py run [CORPORA]       # nested CV -> results/tokenise[_X].csv (cluster)
    python3 tokenise.py summary [CORPORA]   # tables + Plots/tok_gain_vs_merges[_X].png
"""

import collections
import hashlib
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import sympy

from common import BASIS, CACHE, PLOTS, RESULTS, corpus_tuples, kfold, load_corpus
from cv_grouped import families as physics_families

from katz.kneser_ney import KneserNey  # noqa: E402  (common puts the repo root on the path)
from katz.prior import KatzPrior  # noqa: E402


def arity0(basis):
    """Arity of the base tokens of a basis"""
    return {"x": 0, "y": 0, **{o: 1 for o in basis[1]}, **{o: 2 for o in basis[2]}}


CORPUS = "physics"  # corpus being analysed; set before the worker processes are forked


def corpus_basis(name=None):
    name = CORPUS if name is None else name
    if name == "physics":
        return BASIS
    import infl_common

    return infl_common.BASES[name]


def corpus_families(name=None):
    name = CORPUS if name is None else name
    if name == "physics":
        return physics_families()
    import infl_common

    return infl_common.families(name)


def suffix(name=None):
    name = CORPUS if name is None else name
    return "" if name == "physics" else f"_{name}"


FLAT_OPS = ("*", "+")
KMAX = 10  # largest n-ary arity in the vocabulary of the flattened representation
DGRID = np.linspace(0.05, 1.0, 20)  # as cv_grouped.py kn_tuned
MGRID = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 100]
NS = (2, 3, 4)
# name: (flattening, arity limit of merged tokens, merge counts M to evaluate)
#   flattening False: binary trees; 'k': n-ary '*k'/'+k' tokens carrying the arity;
#   'end': n-ary '*'/'+' whose children are ended by an END word of the right model
VARIANTS = {
    "bin2": (False, 2, MGRID),
    "binN": (False, None, MGRID),
    "flat": ("k", None, MGRID),
    "flatE": ("end", None, [0]),
}
INNER_K = 5
ROOT_CONTEXT = True
MIN_COUNT = 2  # a pair seen once is never merged


# ---------------------------------------------------------------------------- coded trees


def corpus_labels():
    """Prefix label lists of the Physics corpus from SymbolCoder.equation2labels (cached)"""
    import katz.process_equations as pe
    import katz.tree_parser as tp

    eqs = load_corpus("physics")
    src = b""
    for mod in (pe, tp):
        with open(mod.__file__, "rb") as f:
            src += f.read()
    key = hashlib.sha256(
        "\n".join(eqs).encode() + repr(BASIS).encode() + src + sympy.__version__.encode()
    ).hexdigest()[:12]
    fname = os.path.join(CACHE, f"toklabels_physics_{key}.pkl")
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
    kp = KatzPrior(2, BASIS, equations=eqs)
    nsym = max(
        [int(kp.maxvar)]
        + [int(k) + 1 for eq in eqs for k in re.findall(r"(?<![A-Za-z_0-9])[xa](\d+)", eq)]
    )
    x = sympy.symbols([f"x{i}" for i in range(nsym)], real=True)
    a = sympy.symbols([f"a{i}" for i in range(nsym)], real=True)
    locs = {**{f"x{i}": x[i] for i in range(nsym)}, **{f"a{i}": a[i] for i in range(nsym)}}
    labels = [kp.coder.equation2labels(eq, locs) for eq in eqs]
    tmp = f"{fname}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(labels, f)
    os.replace(tmp, fname)
    return labels


def coded_tree(labels, basis=BASIS):
    """Coded binary tree (nested tuples of token names) exactly as SymbolCoder codes it:
    every leaf is 'x' (constants are coded as variables), and the second of two leaf
    siblings with different labels is 'y'"""
    ar = arity0(basis)
    pos = [0]

    def build():
        lab = labels[pos[0]]
        pos[0] += 1
        return [lab] + [build() for _ in range(ar.get(lab, 0))]

    def code(node):
        lab, kids = node[0], node[1:]
        tok = lab if lab in ar else "x"
        if (len(kids) == 2 and kids[0][0] not in ar and kids[1][0] not in ar
                and kids[0][0] != kids[1][0]):
            return (tok, code(kids[0]), ("y",))
        return (tok,) + tuple(code(k) for k in kids)

    tree = build()
    assert pos[0] == len(labels)
    return code(tree)


def corpus_trees(name="physics"):
    if name == "physics":
        labs = corpus_labels()
    else:
        import infl_common

        labs = infl_common.labels(name)[0]
    return [coded_tree(lab, corpus_basis(name)) for lab in labs]


def walk(t):
    yield t
    for k in t[1:]:
        yield from walk(k)


def is_flat(tok, op):
    return re.fullmatch(re.escape(op) + r"\d+", tok) is not None


def flatten(t, tag=True):
    """Merge chains of the same associative operator (+ or *) into one n-ary node, whose
    token is '*k' (k = number of arguments) if tag, else '*'. The x/y coding of the leaves
    is kept from the binary tree."""
    tok, kids = t[0], [flatten(k, tag) for k in t[1:]]
    if tok in FLAT_OPS:
        out = []
        for k in kids:
            if (is_flat(k[0], tok) if tag else k[0] == tok):
                out += list(k[1:])
            else:
                out.append(k)
        return (f"{tok}{len(out)}" if tag else tok,) + tuple(out)
    return (tok,) + tuple(kids)


# ---------------------------------------------------------------------------- tokeniser


def apply_merge(t, mg):
    """Apply one merge (parent, slot, child, new) top-down over a tree"""
    p, s, c, new = mg
    tok, kids = t[0], list(t[1:])
    if tok == p and len(kids) > s and kids[s][0] == c:
        kids = kids[:s] + list(kids[s][1:]) + kids[s + 1:]
        tok = new
    return (tok,) + tuple(apply_merge(k, mg) for k in kids)


def _parse_pattern(s, i=0):
    """Parse a pattern string such as 'pow(*2(x,_),y)' into a nested tuple with '_' holes"""
    j = i
    while j < len(s) and s[j] not in "(),":
        j += 1
    name = s[i:j]
    if j < len(s) and s[j] == "(":
        kids, j = [], j + 1
        while True:
            k, j = _parse_pattern(s, j)
            kids.append(k)
            if s[j] == ",":
                j += 1
            else:  # ')'
                return (name,) + tuple(kids), j + 1
    return (name,), j


class Tokeniser:
    def __init__(self, flat, amax, basis=BASIS):
        self.flat, self.amax, self.basis = flat, amax, basis
        self.end = flat == "end"
        self.merges = []
        self.arity = arity0(basis)
        if flat == "k":
            for op in FLAT_OPS:
                for k in range(2, KMAX + 1):
                    self.arity[f"{op}{k}"] = k
        self.pattern = {
            t: t if a == 0 else t + "(" + ",".join(["_"] * a) + ")"
            for t, a in self.arity.items()
        }

    def base(self, t):
        return flatten(t, tag=self.flat == "k") if self.flat else t

    def learn(self, trees, M):
        """Greedy BPE: up to M merges of the most frequent pair (count >= MIN_COUNT)"""
        if self.end and M > 0:
            raise ValueError("merges are not implemented for variable-arity '*'/'+'")
        trees = [self.base(t) for t in trees]
        for _ in range(M):
            C = collections.Counter()
            for t in trees:
                for node in walk(t):
                    for s, ch in enumerate(node[1:]):
                        C[(node[0], s, ch[0])] += 1
            cand = [
                (cnt, k) for k, cnt in C.items()
                if cnt >= MIN_COUNT and (
                    self.amax is None or self.arity[k[0]] - 1 + self.arity[k[2]] <= self.amax
                )
            ]
            if len(cand) == 0:
                break
            cnt, (p, s, c) = min(cand, key=lambda z: (-z[0], repr(z[1])))
            holes = [m.start() for m in re.finditer("_", self.pattern[p])]
            pat = self.pattern[p]
            new = pat[: holes[s]] + self.pattern[c] + pat[holes[s] + 1:]
            self.arity[new] = self.arity[p] - 1 + self.arity[c]
            self.pattern[new] = new
            self.merges.append((p, s, c, new))
            trees = [apply_merge(t, self.merges[-1]) for t in trees]
        return self

    def encode(self, t, M):
        t = self.base(t)
        for mg in self.merges[:M]:
            t = apply_merge(t, mg)
        return t

    def expand(self, t):
        """Inverse of encode: token tree -> (flattened) coded tree"""
        kids = [self.expand(k) for k in t[1:]]
        pat = _parse_pattern(self.pattern[t[0]])[0]
        it = iter(kids)

        def fill(p):
            if p == ("_",):
                return next(it)
            return (p[0],) + tuple(fill(q) for q in p[1:])

        out = fill(pat)
        assert next(it, None) is None
        return out

    def vocab(self, M):
        """Words the left and right models may predict"""
        ops = [o for o in self.basis[1] + self.basis[2]
               if not (self.flat == "k" and o in FLAT_OPS)]
        if self.flat == "k":
            ops += [f"{op}{k}" for op in FLAT_OPS for k in range(2, KMAX + 1)]
        VL = ["x"] + ops + [mg[3] for mg in self.merges[:M]]
        if self.flat or M > 0:  # 'y' can then be a first child (e.g. of pow(x,_))
            VL.append("y")
        VR = list(dict.fromkeys(VL + ["y"] + (["END"] if self.end else [])))
        return VL, VR


# ---------------------------------------------------------------------------- model


def events(t, n, end=False, rc=ROOT_CONTEXT):
    """Prediction events of a tree of any arity: left events (context, word) and right
    events (context, word, excl). With end=True the children of the n-ary '*'/'+' are
    followed by the word END, and excl marks right events at which END is impossible (any
    child of a fixed-arity node, the second child of an n-ary node): their probability is
    renormalised over the words other than END."""
    L, R = [], []
    pad = ("None",) * n if rc else ()
    L.append((pad, t[0]))

    def rec(node, lin):
        lin = lin + (node[0],)
        kids = node[1:]
        if len(kids) == 0:
            return
        full = pad + lin
        L.append((full[-n:], kids[0][0]))
        anc = full[-(n - 1):] if n > 1 else ()
        nary = end and node[0] in FLAT_OPS
        for j in range(1, len(kids)):
            R.append((anc + (kids[j - 1][0],), kids[j][0], end and (not nary or j == 1)))
        if nary:
            R.append((anc + (kids[-1][0],), "END", False))
        for k in kids:
            rec(k, lin)

    rec(t, ())
    return L, R


def fit(trees, n, VL, VR, DL=0.8, DR=0.8, end=False):
    L, R = [], []
    for t in trees:
        ll, rr = events(t, n, end)
        L += ll
        R += rr
    mL = KneserNey([c + (w,) for c, w in L], order=n + 1, vocab=VL, discounts=DL)
    mR = KneserNey([c + (w,) for c, w, _ in R], order=n + 1, vocab=VR, discounts=DR)
    return mL, mR


def logp_right(m, c, w, excl):
    p = m.prob(w, c)
    if excl:
        p = p / (1.0 - m.prob("END", c))
    return np.log(p)


def score(models, t, n, end=False):
    ll, rr = events(t, n, end)
    lp = sum(np.log(models[0].prob(w, c)) for c, w in ll)
    return lp + sum(logp_right(models[1], c, w, x) for c, w, x in rr)


def event_counts(trees, n, end=False):
    ev = (collections.Counter(), collections.Counter())
    for t in trees:
        for side, e in enumerate(events(t, n, end)):
            ev[side].update(e)
    return ev


def side_nll(m, side, counts):
    if side == 0:
        return -sum(c * np.log(m.prob(w, h)) for (h, w), c in counts.items())
    return -sum(c * logp_right(m, h, w, x) for (h, w, x), c in counts.items())


# ---------------------------------------------------------------------------- experiments


TREES = {}


def _trees():
    if CORPUS not in TREES:
        TREES[CORPUS] = corpus_trees(CORPUS)
    return TREES[CORPUS]


def run_fold(args):
    """Nested CV for one outer fold: rows for every held-out equation, variant, n and M"""
    gi, test_idx = args
    trees = _trees()
    N = len(trees)
    test = set(int(i) for i in test_idx)
    train = [i for i in range(N) if i not in test]
    rows = []
    basis = corpus_basis()
    for vname, (flat, amax, Ms) in VARIANTS.items():
        nll = np.zeros((len(NS), len(Ms), 2, len(DGRID)))
        for f in kfold(len(train), INNER_K, gi):  # same inner folds as tune_discounts
            fs = set(f.tolist())
            itrain = [trees[train[j]] for j in range(len(train)) if j not in fs]
            itest = [trees[train[j]] for j in f]
            tok = Tokeniser(flat, amax, basis).learn(itrain, max(Ms))
            for mi, M in enumerate(Ms):
                tr = [tok.encode(t, M) for t in itrain]
                te = [tok.encode(t, M) for t in itest]
                VL, VR = tok.vocab(M)
                for ni, n in enumerate(NS):
                    models = fit(tr, n, VL, VR, end=tok.end)
                    ev = event_counts(te, n, tok.end)
                    for di, D in enumerate(DGRID):
                        for side in (0, 1):
                            models[side].set_discounts(float(D))
                            nll[ni, mi, side, di] += side_nll(models[side], side, ev[side])
        tok = Tokeniser(flat, amax, basis).learn([trees[i] for i in train], max(Ms))
        nmerge = len(tok.merges)
        for ni, n in enumerate(NS):
            dbest = nll[ni].argmin(axis=-1)  # (M, side)
            inner = nll[ni].min(axis=-1).sum(axis=-1)  # (M,)
            msel = int(np.argmin(inner))
            for mi, M in enumerate(Ms):
                DL, DR = float(DGRID[dbest[mi, 0]]), float(DGRID[dbest[mi, 1]])
                VL, VR = tok.vocab(M)
                models = fit([tok.encode(trees[i], M) for i in train], n, VL, VR, DL, DR,
                             end=tok.end)
                for i in sorted(test):
                    t = tok.encode(trees[i], M)
                    rows.append(dict(
                        eq=i, family=gi, variant=vname, n=n, M=M,
                        M_eff=min(M, nmerge), selected=(mi == msel),
                        nll=-score(models, t, n, tok.end), nodes=sum(1 for _ in walk(t)),
                        DL=DL, DR=DR, inner_nll=inner[mi],
                    ))
    return rows


def verify():
    """(1) Events reproduce SymbolCoder/KatzPrior on all corpus equations; (2) encode is
    invertible; (3) the models are normalised over the vocabulary; (4) M = 0 reproduces the
    package's held-out log-probability."""
    trees = _trees()
    N = len(trees)
    basis = corpus_basis()
    print(f"corpus {CORPUS}: {N} equations, {len(corpus_families())} families")

    def ref_tuples(n):
        if CORPUS == "physics":
            return corpus_tuples("physics", n, root_context=ROOT_CONTEXT)
        import infl_common

        return infl_common.tuples(CORPUS, n, root_context=ROOT_CONTEXT)

    def ref_prior(n):
        eq = ["sin(x0) + x1"] if "sin" in basis[1] else ["x0 + x1"]
        return KatzPrior(n, basis, equations=eq, root_context=ROOT_CONTEXT)

    for n in (1, 2, 3, 4):
        T = ref_tuples(n)
        kp = ref_prior(n)
        inv = {v: k for k, v in kp.coder.code.items()}
        bad = 0
        for t, tt in zip(trees, T):
            left, right = kp.split_events(tt)
            ref = (
                [(tuple(inv[c] for c in u[:-1]), inv[u[-1]]) for u in left],
                [(tuple(inv[c] for c in u[-(n + 1):-1]), inv[u[-1]]) for u in right],
            )
            ll, rr = events(t, n)
            bad += (ll, [(c, w) for c, w, _ in rr]) != ref
        print(f"events n={n}: mismatches {bad}/{N}")
    for vname, (flat, amax, Ms) in VARIANTS.items():
        M = max(Ms)
        tok = Tokeniser(flat, amax, basis).learn(trees, M)
        enc = [tok.encode(t, M) for t in trees]
        if tok.end:
            ok = all(len(nd) > 2 for t in enc for nd in walk(t) if nd[0] in FLAT_OPS)
            ok &= all(nd[0] not in FLAT_OPS for t in enc for nd in walk(t)
                      for k in nd[1:] if k[0] == nd[0])
            print(f"{vname}: n-ary nodes have >= 2 args and no same-op child: {ok}")
        else:
            bad = sum(tok.expand(e) != tok.base(t) for e, t in zip(enc, trees))
            okar = all(len(nd) - 1 == tok.arity[nd[0]] for t in enc for nd in walk(t))
            ar = max(tok.arity[nd[0]] for t in enc for nd in walk(t))
            print(f"{vname}: {len(tok.merges)} merges, expand(encode) mismatches {bad}/{N}, "
                  f"arity consistent {okar}, max arity {ar}")
            print("  first merges:", [mg[3] for mg in tok.merges[:12]])
        VL, VR = tok.vocab(M)
        mL, mR = fit(enc, 3, VL, VR, end=tok.end)
        dev = 0.0
        for m, side in ((mL, 0), (mR, 1)):
            ctxs = {e[0] for t in enc[:40] for e in events(t, 3, tok.end)[side]}
            ctxs |= {("None", "None", "sin"), ("zzz",)}
            for c in ctxs:
                dev = max(dev, abs(sum(m.prob(w, c) for w in m.vocab) - 1))
        if tok.end:  # renormalised right distribution at an excl context
            c = next(e[0] for t in enc for e in events(t, 3, True)[1] if e[2])
            s = sum(np.exp(logp_right(mR, c, w, True)) for w in mR.vocab if w != "END")
            dev = max(dev, abs(s - 1))
        print(f"  max |sum_w P(w|ctx) - 1| = {dev:.2e}")
    # Held-out comparison against KatzPrior for 5 families, n = 3, D = 0.8
    T = ref_tuples(3)
    tok = Tokeniser(False, 2, basis)
    VL, VR = tok.vocab(0)
    diff = 0.0
    for f in corpus_families()[:5]:
        fs = set(f.tolist())
        kp = ref_prior(3)
        kp.fit([T[i] for i in range(N) if i not in fs])
        models = fit([trees[i] for i in range(N) if i not in fs], 3, VL, VR)
        for i in f:
            diff = max(diff, abs(kp.logprior_tuples(T[i]) - score(models, trees[i], 3)))
    print(f"held-out logP vs KatzPrior (M=0, n=3): max |diff| = {diff:.2e}")


def screen():
    """Quick look: fixed D = 0.8, 10 folds of whole families, n = 3"""
    trees = _trees()
    N = len(trees)
    fam = corpus_families()
    rng = np.random.default_rng(0)
    order = rng.permutation(len(fam))
    folds = [np.concatenate([fam[j] for j in order[k::10]]) for k in range(10)]
    res = collections.defaultdict(lambda: np.zeros(N))
    for f in folds:
        fs = set(f.tolist())
        train = [trees[i] for i in range(N) if i not in fs]
        for vname, (flat, amax, Ms) in VARIANTS.items():
            tok = Tokeniser(flat, amax, corpus_basis()).learn(train, max(Ms))
            for M in Ms:
                VL, VR = tok.vocab(M)
                models = fit([tok.encode(t, M) for t in train], 3, VL, VR, end=tok.end)
                for i in f:
                    res[(vname, M)][i] = -score(models, tok.encode(trees[i], M), 3, tok.end)
    base = res[("bin2", 0)]
    print(f"{CORPUS}, n=3, D=0.8, 10 grouped folds: base NLL {base.mean():.3f}")
    for vname, (_, _, Ms) in VARIANTS.items():
        line = []
        for M in Ms:
            d = base - res[(vname, M)]
            line.append(f"{M}:{d.mean():+.2f}±{d.std(ddof=1) / np.sqrt(N):.2f}")
        print(f"{vname:5s} gain vs base:", "  ".join(line))


def summary():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.join(RESULTS, f"tokenise{suffix()}.csv"))
    base = df[(df.variant == "bin2") & (df.M == 0)].set_index(["n", "eq"]).nll
    ref_n = 3  # common reference: the package at n = 3 (kn3rc of the ESR benchmark)

    def gains(g, n):
        """Paired gains vs the package at the same n and at ref_n (positive = better)"""
        row = {}
        for key, nb in (("", n), ("_ref3", ref_n)):
            d = base.loc[nb].reindex(g["eq"]).to_numpy() - g.nll.to_numpy()
            row[f"gain{key}"] = d.mean()
            row[f"se{key}"] = d.std(ddof=1) / np.sqrt(len(d))
            row[f"frac_better{key}"] = np.mean(d > 0)
        return row

    out = []
    for (v, n, M), g in df.groupby(["variant", "n", "M"]):
        out.append(dict(variant=v, n=n, M=M, NLL=g.nll.mean(), **gains(g, n),
                        nodes=g.nodes.mean()))
    for (v, n), g in df[df.selected].groupby(["variant", "n"]):
        Ms = g.groupby("family").M.first()
        out.append(dict(variant=v, n=n, M="sel", NLL=g.nll.mean(), **gains(g, n),
                        nodes=g.nodes.mean(), M_sel_median=Ms.median(),
                        M_sel_range=f"{Ms.min()}-{Ms.max()}"))
    s = pd.DataFrame(out)
    s.to_csv(os.path.join(RESULTS, f"tokenise{suffix()}_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(s.round(3).to_string(index=False))

    colors = {"bin2": "#1f77b4", "binN": "#ff7f0e", "flat": "#2ca02c", "flatE": "#d62728"}
    labels = {"bin2": "tree-BPE, arity ≤ 2", "binN": "tree-BPE, any arity",
              "flat": "flattened (*k tokens) + BPE", "flatE": "flattened (END word)"}
    fig, axes = plt.subplots(1, len(NS), figsize=(4 * len(NS), 3.6), sharey=True)
    for ax, n in zip(axes, NS):
        for k, v in enumerate(VARIANTS):
            g = s[(s.variant == v) & (s.n == n) & (s.M != "sel")]
            M = g.M.astype(int).to_numpy()
            x = np.where(M == 0, 0.5, M) * (1 + 0.04 * (k - 1.5))  # small offsets
            ax.errorbar(x, g.gain_ref3, yerr=g.se_ref3, color=colors[v], marker="o", ms=3,
                        capsize=2, lw=1.2, label=labels[v])
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("number of merges M (M = 0 plotted at 0.5)")
        ax.text(0.04, 0.05, f"n = {n}", transform=ax.transAxes)
    axes[0].set_ylabel(f"held-out gain vs package at n = {ref_n} (nats/eq)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center",
               ncol=len(VARIANTS))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    stem = f"tok_gain_vs_merges{suffix()}"
    fname = os.path.join(PLOTS, f"{stem}.png")
    if os.path.isfile(fname):
        os.makedirs(os.path.join(PLOTS, "Old"), exist_ok=True)
        os.replace(fname, os.path.join(PLOTS, "Old", f"{stem}_{int(time.time())}.png"))
    fig.savefig(fname, dpi=150)
    print("wrote", fname)


if __name__ == "__main__":
    t0 = time.time()
    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"
    corpora = sys.argv[2].split(",") if len(sys.argv) > 2 else ["physics"]
    for CORPUS in corpora:  # module global, read by the functions above and by the workers
        if mode == "verify":
            verify()
        elif mode == "screen":
            screen()
        elif mode == "time":
            rows = run_fold((0, corpus_families()[0]))
            print(pd.DataFrame(rows).groupby(["variant", "n", "M"]).nll.mean()
                  .unstack("M").round(2))
        elif mode == "run":
            from multiprocessing import Pool

            _trees()  # parse once (cached) before forking
            nproc = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_ON_NODE", "2")))
            jobs = list(enumerate(corpus_families()))
            with Pool(nproc) as pool:
                res = pool.map(run_fold, jobs, chunksize=1)
            df = pd.DataFrame([r for rows in res for r in rows])
            df.to_csv(os.path.join(RESULTS, f"tokenise{suffix()}.csv"), index=False)
            summary()
        elif mode == "summary":
            summary()
    print(f"time {time.time() - t0:.1f}s")
