"""Held-out codelengths of inflationary potentials: Katz vs Kneser-Ney.

within : leave-one-out (loo) and leave-one-family-out (fam) on corpora A, B, M; kn_tuned has its
         discount chosen by inner 5-fold CV on the training equations of each fold (nested).
cross  : train on the 161 physics equations, test on the 67 potentials (M), and the reverse;
         kn_tuned tuned by 10-fold CV on the training corpus.
Usage (from assessment/): python infl_cv.py within|cross   (workers: $NPROC, default 2)
Results: results/infl_within.csv, results/infl_cross.csv (log-probability per held-out equation)
"""
import itertools
import sys
import time

from infl_common import (RESULTS, families, make_prior, np, os, pd, score, tuples)

MODELS = ("katz", "katz_norm", "kn_fixed", "kn_mkn", "kn_tuned")
NS = (1, 2, 3, 4, 6, 9)
REPS = {"base": {}, "root": {"root_context": True},
        "root+const": {"root_context": True, "code_constants": True}}
GRID = np.linspace(0.05, 1.0, 20)


def fitted(name, n, model, rkw, train, seed):
    p = make_prior(name, n, model, **rkw)
    p.eq_tuples = train
    if model == "kn_tuned":
        p.tune_discounts("single", k=5 if len(train) < 100 else 10, seed=seed, grid=GRID)
    else:
        p.fit(train)
    return p


def run_within(args):
    name, n, rep = args
    rkw = REPS[rep]
    T = tuples(name, n, **rkw)
    N = len(T)
    rows = []
    for cv in ("loo", "fam"):
        folds = [np.array([i]) for i in range(N)] if cv == "loo" else families(name)
        for fi, f in enumerate(folds):
            fs = set(f.tolist())
            train = [T[i] for i in range(N) if i not in fs]
            for model in MODELS:
                p = fitted(name, n, model, rkw, train, seed=fi)
                for i in f:
                    rows.append(dict(corpus=name, cv=cv, n=n, rep=rep, model=model, eq=int(i),
                                     logp=score(p, T[i])))
    return rows


def run_cross(args):
    (train_name, test_name), n, rep = args
    rkw = REPS[rep]
    Ttr = tuples(train_name, n, **rkw)
    Tte = tuples(test_name, n, **rkw)
    rows = []
    for model in MODELS:
        p = fitted(train_name, n, model, rkw, Ttr, seed=0)
        for i, t in enumerate(Tte):
            rows.append(dict(train=train_name, test=test_name, n=n, rep=rep, model=model,
                             eq=i, logp=score(p, t)))
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    mode = sys.argv[1]
    nproc = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    if mode == "within":
        configs = list(itertools.product(("A", "B", "M"), NS, REPS))
        fn, out = run_within, "infl_within.csv"
    elif mode == "cross":
        configs = list(itertools.product((("physics", "M"), ("M", "physics")), NS + (10,), REPS))
        fn, out = run_cross, "infl_cross.csv"
    with Pool(min(nproc, len(configs))) as pool:
        res = pool.map(fn, sorted(configs, key=lambda c: -c[1]), chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(RESULTS, out), index=False)
    print(f"{out}: {len(df)} rows, {time.time() - t0:.0f}s")
