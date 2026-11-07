"""Experiment 4: discriminating real physics equations from corrupted copies.

For each held-out family of equations (leave-one-family-out, as in cv_grouped.py), the
prior is trained on the remaining equations and must assign a higher probability to each
real equation than to K corrupted copies with the same tree shape and node count:
    swap1   : one operator replaced by a different operator of the same arity
    relabel : every operator replaced by a random operator of the same arity
A prior that only counts nodes (or is uniform per node) scores AUC = 0.5.
Metrics: AUC (fraction of real-vs-corrupt pairs ranked correctly, ties = 1/2) and the
mean log-probability margin log p(real) - log p(corrupt).
"""

import itertools
import time

import sympy

from common import BASIS, RESULTS, load_corpus, make_prior, np, os, pd
from cv_grouped import families
from katz.prior import KatzPrior

K = 20
ARITY = {**{o: 1 for o in BASIS[1]}, **{o: 2 for o in BASIS[2]}}


def const_swap_sites(labels):
    """Binary nodes whose two children are a numerical constant and a variable"""
    sites, pos = [], [0]

    def walk():
        i = pos[0]
        pos[0] += 1
        kids = [walk() for _ in range(ARITY.get(labels[i], 0))]
        if len(kids) == 2:
            a, b = labels[kids[0]], labels[kids[1]]
            leaf = [x not in ARITY for x in (a, b)]
            isconst = [x == "a" for x in (a, b)]
            if all(leaf) and isconst[0] != isconst[1]:
                sites.append((kids[0], kids[1]))
        return i

    walk()
    return sites


def corrupt(labels, kind, rng):
    lab = list(labels)
    if kind == "constswap":
        sites = const_swap_sites(lab)
        i, j = sites[rng.integers(len(sites))]
        lab[i], lab[j] = lab[j], lab[i]
        return lab
    ops = [i for i, x in enumerate(lab) if x in ARITY]
    if kind == "swap1":
        i = rng.choice(ops)
        same = [o for o in BASIS[ARITY[lab[i]]] if o != lab[i]]
        lab[i] = rng.choice(same)
    else:
        for i in ops:
            lab[i] = rng.choice(BASIS[ARITY[lab[i]]])
    return lab


def build_sets(n, root_context, seed=0, code_constants=False,
               kinds=("swap1", "relabel")):
    """Tuples of every corpus equation and of its corruptions"""
    eqs = load_corpus("physics")
    kp = KatzPrior(n, BASIS, equations=eqs, root_context=root_context,
                   code_constants=code_constants)
    xs = sympy.symbols([f"x{i}" for i in range(kp.maxvar)], real=True)
    as_ = sympy.symbols([f"a{i}" for i in range(kp.maxvar)], real=True)
    locs = {**{f"x{i}": xs[i] for i in range(kp.maxvar)},
            **{f"a{i}": as_[i] for i in range(kp.maxvar)}}
    rng = np.random.default_rng(seed)
    real, fake = [], []
    for eq in eqs:
        labels = kp.coder.equation2labels(eq, locs)
        real.append(kp.coder.labels2ntuples(n + 1, labels))
        f = {}
        for kind in kinds:
            if kind == "constswap":
                sites = const_swap_sites(labels)
                if len(sites) == 0:
                    f[kind] = []
                    continue
                # every distinct single swap once
                f[kind] = []
                for a, b in sites:
                    lab = list(labels)
                    lab[a], lab[b] = lab[b], lab[a]
                    f[kind].append(kp.coder.labels2ntuples(n + 1, lab))
                continue
            f[kind] = [kp.coder.labels2ntuples(n + 1, corrupt(labels, kind, rng))
                       for _ in range(K)]
        fake.append(f)
    return real, fake


def run(n, rep, code_constants=False, kinds=("swap1", "relabel")):
    rkw = {"root_context": True} if rep == "root" else {}
    if code_constants:
        rkw["code_constants"] = True
    real, fake = build_sets(n, rkw.get("root_context", False),
                            code_constants=code_constants, kinds=kinds)
    N = len(real)
    rows = []
    for f in families():
        fs = set(f.tolist())
        train = [real[i] for i in range(N) if i not in fs]
        for name in ("katz", "kn_mkn", "kn_d"):
            sm = "katz" if name == "katz" else "kn"
            skw = dict(discounts=0.8) if name == "kn_d" else None
            p = make_prior(n, sm, smoothing_kwargs=skw, **rkw)
            p.fit(train)
            for i in f:
                lr = p.logprior_tuples(real[i])
                for kind in kinds:
                    if len(fake[i][kind]) == 0:
                        continue
                    lf = np.array([p.logprior_tuples(t) for t in fake[i][kind]])
                    with np.errstate(invalid="ignore"):
                        auc = np.mean((lr > lf) + 0.5 * (lr == lf))
                        margin = np.mean(lr - lf[np.isfinite(lf)]) if np.isfinite(lr) \
                            else np.nan
                    rows.append(dict(eq=int(i), n=n, rep=rep, model=name, kind=kind,
                                     auc=auc, margin=margin,
                                     const=int(code_constants)))
    return rows


if __name__ == "__main__":
    import sys

    t0 = time.time()
    rows = []
    if len(sys.argv) > 1 and sys.argv[1] == "const":
        for n, rep, cc in itertools.product((2, 3), ("base", "root"), (False, True)):
            rows += run(n, rep, code_constants=cc,
                        kinds=("constswap", "swap1", "relabel"))
            print(f"done n={n} {rep} const={cc} {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(RESULTS, "discrimination_const.csv"), index=False)
        s = df.groupby(["kind", "n", "rep", "const", "model"]).agg(
            auc=("auc", "mean"),
            auc_se=("auc", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            neq=("auc", "size")).round(3)
        pd.set_option("display.width", 200)
        print(s.to_string())
        print(f"time {time.time() - t0:.1f}s")
        sys.exit()
    for n, rep in itertools.product((2, 3, 6), ("base", "root")):
        rows += run(n, rep)
        print(f"done n={n} {rep} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "discrimination.csv"), index=False)
    s = df.groupby(["kind", "n", "rep", "model"]).agg(
        auc=("auc", "mean"), auc_se=("auc", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        margin=("margin", "mean")).round(3)
    pd.set_option("display.width", 200)
    print(s.to_string())
    print(f"time {time.time() - t0:.1f}s")
