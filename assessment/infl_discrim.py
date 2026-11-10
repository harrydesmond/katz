"""Discriminating real inflationary potentials from corrupted copies (as discrimination.py for the
physics corpus): leave-one-family-out; each real tree is compared with K = 20 corruptions of the
same shape (swap1: one operator replaced by another of the same arity in the corpus's basis;
relabel: every operator replaced). AUC = fraction of real-vs-corrupt pairs ranked correctly."""
import itertools
import time

from infl_common import (BASES, RESULTS, _coder_prior, families, labels, make_prior, np, os,
                         pd)

K = 20


def corrupt(lab, kind, rng, basis):
    arity = {**{o: 1 for o in basis[1]}, **{o: 2 for o in basis[2]}}
    lab = list(lab)
    ops = [i for i, x in enumerate(lab) if x in arity]
    if kind == "swap1":
        i = rng.choice(ops)
        lab[i] = rng.choice([o for o in basis[arity[lab[i]]] if o != lab[i]])
    else:
        for i in ops:
            lab[i] = rng.choice(basis[arity[lab[i]]])
    return lab


def run(args):
    name, n, rep = args
    rkw = {"root_context": True} if rep == "root" else {}
    labs, _ = labels(name)
    basis = BASES[name]
    kp = _coder_prior(basis, root_context=rkw.get("root_context", False))
    rng = np.random.default_rng(0)
    real = [kp.coder.labels2ntuples(n + 1, lab) for lab in labs]
    fake = [{k: [kp.coder.labels2ntuples(n + 1, corrupt(lab, k, rng, basis)) for _ in range(K)]
             for k in ("swap1", "relabel")} for lab in labs]
    N = len(real)
    rows = []
    for f in families(name):
        fs = set(f.tolist())
        train = [real[i] for i in range(N) if i not in fs]
        for model in ("katz", "katz_norm", "kn_fixed", "kn_mkn"):
            p = make_prior(name, n, model, **rkw)
            p.fit(train)
            for i in f:
                lr = p.logprior_tuples(real[i])
                for kind in ("swap1", "relabel"):
                    lf = np.array([p.logprior_tuples(t) for t in fake[i][kind]])
                    auc = np.mean((lr > lf) + 0.5 * (lr == lf))
                    rows.append(dict(corpus=name, n=n, rep=rep, model=model, kind=kind,
                                     eq=int(i), auc=auc))
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    nproc = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    configs = list(itertools.product(("A", "B", "M"), (2, 3, 6), ("base", "root")))
    with Pool(min(nproc, len(configs))) as pool:
        res = pool.map(run, sorted(configs, key=lambda c: -c[1]), chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(RESULTS, "infl_discrimination.csv"), index=False)
    print(f"{len(df)} rows {time.time() - t0:.0f}s")
