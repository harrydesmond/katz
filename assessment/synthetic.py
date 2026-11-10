"""Experiment 3: synthetic corpora drawn from known generative models of expression trees.

A "truth" is a probabilistic tree grammar over the coded tokens (x, y, unary and binary
operators) in which
    root           ~ P_root(w)
    first child    ~ P_L(w | grandparent, parent)
    second child   ~ P_R(w | parent, first child)
with arity fixed by the token (leaves 0, unary 1, binary 2) and 'y' (a leaf different from
the first-child leaf) allowed only after a leaf first child. Truths are built either from
the real Physics corpus (hierarchical-Dirichlet smoothing of its counts: neutral between
Katz and Kneser-Ney) or drawn at random from a hierarchical Dirichlet process with a given
concentration (small = strong, sparse context dependence).

Training corpora of M trees are sampled, Katz and Kneser-Ney priors fitted, and their
excess codelength relative to the truth, the mean of log p_true(T) - log q(T) (nats per tree)
over the trees of a large independent test sample that every model scores as finite, is measured.
It estimates the KL divergence only when no test tree is excluded.
"""

import itertools
import json
import sys
import time

import numpy as np

from common import RESULTS, corpus_tuples, kfold, make_prior, os, pd

MAX_NODES = 80


class Truth:
    def __init__(self, coder, kind="corpus", alpha=1.0, beta=(5.0, 5.0), seed=0,
                 target_arity=None, max_depth=8):
        c = coder.code
        self.none = c["None"]
        self.x, self.y = c["x"], c["y"]
        self.unary = [c[o] for o in coder.basis_functions[1]]
        self.binary = [c[o] for o in coder.basis_functions[2]]
        self.VL = [self.x] + self.unary + self.binary
        self.VR = self.VL + [self.y]
        self.arity = {w: 0 for w in (self.x, self.y)}
        self.arity.update({w: 1 for w in self.unary})
        self.arity.update({w: 2 for w in self.binary})
        self.kind = kind
        self.max_depth = max_depth  # nodes at this depth are forced to be leaves
        rng = np.random.default_rng(seed)
        # Corpus statistics (n=2 tuples, no root context)
        T = corpus_tuples("physics", 2)
        croot, cl1, cl2, cl3, cr1, cr2, cr3 = ({} for _ in range(7))

        def inc(d, k):
            d[k] = d.get(k, 0) + 1

        for t in T:
            inc(croot, t[0][0])
            for tt in t[1:]:
                anc, lft, r = tt[:-2], tt[-2], tt[-1]
                inc(cl1, lft)
                inc(cl2, (anc[-1], lft))
                if len(anc) >= 2:
                    inc(cl3, (anc[-2], anc[-1], lft))
                if r != self.none:
                    inc(cr1, r)
                    inc(cr2, (anc[-1], r))
                    inc(cr3, (anc[-1], lft, r))
        self.uniL = self._norm({w: cl1.get(w, 0) + 0.5 for w in self.VL})
        self.uniR = self._norm({w: cr1.get(w, 0) + 0.5 for w in self.VR})
        ops = self.unary + self.binary
        ctx_l = list(itertools.product([None] + ops, ops))  # (grandparent, parent)
        ctx_r = list(itertools.product(ops, self.VL))  # (parent, left sibling)

        self.PL, self.PR = {}, {}
        if kind == "corpus":
            a = alpha
            self.Proot = self._norm(
                {w: croot.get(w, 0) + a * self.uniL[w] for w in self.VL})
            PL2 = {v: self._norm({w: cl2.get((v, w), 0) + a * self.uniL[w]
                                  for w in self.VL}) for v in ops}
            for g, v in ctx_l:
                if g is None:
                    self.PL[(g, v)] = PL2[v]
                else:
                    self.PL[(g, v)] = self._norm(
                        {w: cl3.get((g, v, w), 0) + a * PL2[v][w] for w in self.VL})
            PR2 = {v: self._norm({w: cr2.get((v, w), 0) + a * self.uniR[w]
                                  for w in self.VR}) for v in self.binary}
            for v, lft in ctx_r:
                if v in self.binary:
                    self.PR[(v, lft)] = self._mask_y(lft, {
                        w: cr3.get((v, lft, w), 0) + a * PR2[v][w] for w in self.VR})
        else:
            b1, b2 = beta
            if target_arity is not None:
                # tilt the base distributions towards leaves to keep trees finite
                self.uniL = self._tilt(self.uniL, target_arity)
                self.uniR = self._tilt(self.uniR, target_arity)
            self.Proot = self._dirichlet(rng, {w: croot.get(w, 0) + 0.5 for w in self.VL},
                                         20.0)
            PL2 = {v: self._dirichlet(rng, self.uniL, b1) for v in ops}
            for g, v in ctx_l:
                self.PL[(g, v)] = (PL2[v] if g is None
                                   else self._dirichlet(rng, PL2[v], b2))
            PR2 = {v: self._dirichlet(rng, self.uniR, b1) for v in self.binary}
            for v, lft in ctx_r:
                if v in self.binary:
                    self.PR[(v, lft)] = self._mask_y(lft, self._dirichlet(rng, PR2[v], b2))

    def _tilt(self, d, target):
        """Rescale the leaf probabilities so that the mean arity equals target"""
        leaf = sum(v for w, v in d.items() if self.arity[w] == 0)
        ar = sum(v * self.arity[w] for w, v in d.items())
        # after scaling leaves by f: ar / (f * leaf + 1 - leaf) = target
        f = (ar / target - (1 - leaf)) / leaf
        return self._norm({w: v * (f if self.arity[w] == 0 else 1.0) for w, v in d.items()})

    @staticmethod
    def _norm(d):
        s = sum(d.values())
        return {k: v / s for k, v in d.items()}

    @staticmethod
    def _dirichlet(rng, base, conc):
        keys = list(base)
        p = rng.dirichlet(conc * np.array([base[k] for k in keys]) + 1e-3)
        return dict(zip(keys, p))

    def _mask_y(self, left, d):
        if left != self.x:
            d = dict(d)
            d[self.y] = 0.0
        return self._norm(d)

    def _leafonly(self, d):
        return self._norm({w: (v if self.arity[w] == 0 else 0.0) for w, v in d.items()})

    def _dist(self, side, key, depth):
        """Conditional distribution of a child at the given depth"""
        d = (self.PL if side == "L" else self.PR)[key]
        if self.max_depth is not None and depth >= self.max_depth:
            d = self._leafonly(d)
        return d

    def _draw(self, rng, d):
        keys = list(d)
        cum = np.cumsum([d[w] for w in keys])
        return keys[min(np.searchsorted(cum, rng.random() * cum[-1]), len(keys) - 1)]

    def sample_tree(self, rng):
        """Tree as nested lists [token, child, child]; None if it exceeds MAX_NODES"""
        count = [0]

        def grow(tok, parent, depth):
            count[0] += 1
            if count[0] > MAX_NODES:
                raise RecursionError
            node = [tok]
            ar = self.arity[tok]
            if ar >= 1:
                left = self._draw(rng, self._dist("L", (parent, tok), depth + 1))
                node.append(grow(left, tok, depth + 1))
                if ar == 2:
                    right = self._draw(rng, self._dist("R", (tok, left), depth + 1))
                    node.append(grow(right, tok, depth + 1))
            return node

        try:
            return grow(self._draw(rng, self.Proot), None, 0)
        except RecursionError:
            return None

    def logp(self, tree):
        lp = np.log(self.Proot[tree[0]])

        def rec(node, parent, depth):
            nonlocal lp
            if len(node) >= 2:
                lp += np.log(self._dist("L", (parent, node[0]), depth + 1)[node[1][0]])
                rec(node[1], node[0], depth + 1)
            if len(node) == 3:
                lp += np.log(self._dist("R", (node[0], node[1][0]), depth + 1)[node[2][0]])
                rec(node[2], node[0], depth + 1)

        rec(tree, None, 0)
        return lp

    def sample(self, rng, M):
        out, rejected = [], 0
        while len(out) < M:
            t = self.sample_tree(rng)
            if t is None:
                rejected += 1
            else:
                out.append(t)
        return out, rejected


def tree_to_tuples(tree, n, none_code, root_context=False):
    """Same n-tuples as SymbolCoder.process_all_equations(n + 1, ...)"""
    out = []

    def rec(node, lineage):
        lin = lineage + [node[0]]
        if len(node) > 1:
            right = node[2][0] if len(node) == 3 else none_code
            tup = tuple(lin[-n:]) + (node[1][0], right)
            if root_context:
                tup = (none_code,) * (n + 2 - len(tup)) + tup
            out.append(tup)
            for ch in node[1:]:
                rec(ch, lin)

    rec(tree, [])
    if len(out) == 0:  # single leaf
        return [((none_code,) * n if root_context else ()) + (tree[0],)]
    root = ((none_code,) * n if root_context else ()) + (tree[0],)
    return [root] + out


class CachedPrior:
    """Memoise event probabilities of a fitted prior (pure speed-up)"""

    def __init__(self, prior):
        self.p = prior
        self.cl, self.cr = {}, {}

    def logq(self, t):
        left, right = self.p.split_events(t)
        lp = 0.0
        for tt in left:
            k = tt
            if k not in self.cl:
                self.cl[k] = self.p.backoff_left.get_pbo(tt[-1], tt[:-1])
            lp += np.log(self.cl[k]) if self.cl[k] > 0 else -np.inf
        for tt in right:
            k = tt[-(self.p.n + 1):]
            if k not in self.cr:
                self.cr[k] = self.p.backoff_right.get_pbo(tt[-1], k[:-1])
            lp += np.log(self.cr[k]) if self.cr[k] > 0 else -np.inf
        return lp


def tune_single_D(prior, tuples, k=5, seed=0, grid=np.linspace(0.1, 1.0, 19)):
    """Single discount per side chosen by k-fold CV on the training trees"""
    folds = kfold(len(tuples), k, seed)
    nll = np.zeros((2, len(grid)))
    for f in folds:
        fs = set(f.tolist())
        prior.fit([tuples[i] for i in range(len(tuples)) if i not in fs])
        ev = [prior.split_events(tuples[i]) for i in f]
        for j, D in enumerate(grid):
            prior.backoff_left.set_discounts(float(D))
            prior.backoff_right.set_discounts(float(D))
            for left, right in ev:
                nll[0, j] -= sum(np.log(prior.backoff_left.prob(tt[-1], tt[:-1]))
                                 for tt in left)
                nll[1, j] -= sum(np.log(prior.backoff_right.prob(
                    tt[-1], tt[-(prior.n + 1):-1])) for tt in right)
    return float(grid[np.argmin(nll[0])]), float(grid[np.argmin(nll[1])])


def run_one(truth_spec, M, rep, ntest=3000, ns=(1, 2, 3), root_context=True):
    """One training corpus of M trees from a truth; returns a list of result rows"""
    coder = make_prior(2).coder
    truth = Truth(coder, **truth_spec)
    rng_test = np.random.default_rng(12345)
    test, rej = truth.sample(rng_test, ntest)
    logp_true = np.array([truth.logp(t) for t in test])
    rng = np.random.default_rng(1000 * rep + M)
    train, _ = truth.sample(rng, M)
    rows = []
    for n in ns:
        tr = [tree_to_tuples(t, n, truth.none, root_context) for t in train]
        te = [tree_to_tuples(t, n, truth.none, root_context) for t in test]
        models = {}
        pk = make_prior(n, "katz", root_context=root_context)
        pk.fit(tr)
        models["katz"] = pk
        pkn = make_prior(n, "katz", root_context=root_context,
                         smoothing_kwargs={"normalise": True})
        pkn.fit(tr)
        models["katz_norm"] = pkn  # Katz with proper conditional distributions
        pm = make_prior(n, "kn", root_context=root_context)
        pm.fit(tr)
        models["kn_mkn"] = pm
        if n in (2, 3):
            Dl, Dr = tune_single_D(make_prior(n, "kn", root_context=root_context), tr)
            pt = make_prior(n, "kn", root_context=root_context)
            pt.fit(tr)
            pt.backoff_left.set_discounts(Dl)
            pt.backoff_right.set_discounts(Dr)
            models["kn_cv"] = pt
            pf = make_prior(n, "kn", root_context=root_context,
                            smoothing_kwargs=dict(discounts=0.75 if n == 2 else 0.85))
            pf.fit(tr)
            models["kn_d"] = pf
        ex = {}
        for name, p in models.items():
            cp = CachedPrior(p)
            ex[name] = logp_true - np.array([cp.logq(t) for t in te])
        common = np.all([np.isfinite(v) for v in ex.values()], axis=0)
        for name, e in ex.items():
            fin = np.isfinite(e)
            rows.append(dict(
                truth=truth_spec["kind"] + ("" if truth_spec["kind"] == "corpus"
                                            else f"_b{truth_spec['beta'][0]:g}"),
                M=M, rep=rep, n=n, model=name,
                excess=e[common].mean(),  # on trees finite under every model of this n
                excess_se=e[common].std(ddof=1) / np.sqrt(common.sum()),
                excess_own=e[fin].mean(), frac_inf=1 - fin.mean(),
                frac_common=common.mean(), H_true=-logp_true.mean(),
                nodes=np.mean([len(t) for t in te]), rejected_test=rej,
                excess_vs_katz=(e[common] - ex["katz"][common]).mean(),
                excess_vs_katz_se=(e[common] - ex["katz"][common]).std(ddof=1)
                / np.sqrt(common.sum()),
                excess_vs_katz_norm=(e[common] - ex["katz_norm"][common]).mean(),
                excess_vs_katz_norm_se=(e[common] - ex["katz_norm"][common]).std(ddof=1)
                / np.sqrt(common.sum()),
            ))
    return rows


TRUTHS = {
    "corpus": dict(kind="corpus", alpha=1.0),
    "dir_strong": dict(kind="dirichlet", beta=(3.0, 3.0), seed=1),
    "dir_mid": dict(kind="dirichlet", beta=(10.0, 10.0), seed=3),
    "dir_weak": dict(kind="dirichlet", beta=(30.0, 30.0), seed=2),
}
SIZES = (25, 50, 100, 160, 300, 1000, 3000)


def _job(args):
    tname, M, rep = args
    return run_one(TRUTHS[tname], M, rep)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    t0 = time.time()
    if mode == "validate":
        # check tree_to_tuples against SymbolCoder on real equations
        from common import BASIS, load_corpus
        from katz.prior import KatzPrior

        for n, rc in itertools.product((1, 2, 3), (False, True)):
            eqs = load_corpus("physics")
            kp = KatzPrior(n, BASIS, equations=eqs, root_context=rc)
            bad = 0
            import sympy

            xs = sympy.symbols([f"x{i}" for i in range(30)], real=True)
            as_ = sympy.symbols([f"a{i}" for i in range(30)], real=True)
            locs = {**{f"x{i}": xs[i] for i in range(30)},
                    **{f"a{i}": as_[i] for i in range(30)}}
            for eq in eqs:
                labels = kp.coder.equation2labels(eq, locs)
                ar = {o: 1 for o in BASIS[1]}
                ar.update({o: 2 for o in BASIS[2]})
                pos = [0]

                def build():
                    lab = labels[pos[0]]
                    pos[0] += 1
                    return [lab] + [build() for _ in range(ar.get(lab, 0))]

                tree = build()

                def code(node, sib=None):
                    lab = node[0]
                    if lab in ar:
                        tok = kp.coder.code[lab]
                    else:
                        tok = kp.coder.code["x"]
                    out = [tok] + [None] * (len(node) - 1)
                    kids = node[1:]
                    if len(kids) == 2 and kids[0][0] not in ar and kids[1][0] not in ar \
                            and kids[0][0] != kids[1][0]:
                        out[1] = code(kids[0])
                        out[2] = [kp.coder.code["y"]]
                    else:
                        for i, kd in enumerate(kids):
                            out[i + 1] = code(kd)
                    return out

                mine = tree_to_tuples(code(tree), n, kp.coder.code["None"], rc)
                ref = kp.coder.process_all_equations(n + 1, [eq], kp.maxvar)
                bad += mine != ref
            print(f"n={n} root_context={rc}: mismatches {bad}/{len(eqs)}")
    elif mode == "test":
        rows = run_one(TRUTHS["corpus"], 100, 0, ntest=500)
        print(pd.DataFrame(rows)[["truth", "M", "n", "model", "excess", "excess_se",
                                  "frac_inf", "H_true", "nodes"]].round(3))
    elif mode == "run":
        from multiprocessing import Pool

        nproc = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_ON_NODE", "2")))
        reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        jobs = [(t, M, r) for t in TRUTHS for M in SIZES for r in range(reps)]
        with Pool(nproc) as pool:
            res = pool.map(_job, jobs, chunksize=1)
        df = pd.DataFrame([r for rows in res for r in rows])
        df.to_csv(os.path.join(RESULTS, "synthetic.csv"), index=False)
        print(df.groupby(["truth", "M", "n", "model"]).excess.mean().unstack().round(3))
        with open(os.path.join(RESULTS, "synthetic_meta.json"), "w") as f:
            json.dump(dict(truths=TRUTHS, sizes=SIZES, reps=reps, max_nodes=MAX_NODES),
                      f, default=str)
    print(f"time {time.time() - t0:.1f}s")
