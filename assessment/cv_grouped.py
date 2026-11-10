"""Experiment 1c: leave-one-family-out cross-validation.

The Feynman corpus contains families of closely related equations from the same lecture
section (e.g. I.6.2a, I.6.2, I.6.2b), which leave-one-out lets a model memorise. Here a
whole family (lecture chapter.section; each non-Feynman equation is its own family) is
held out at a time. KN discounts are tuned on the training equations of each fold by inner
5-fold CV (nested), so 'kn_tuned' is an honest estimate.
"""

import re
import sys
import time

from common import (
    ROOT,
    RESULTS,
    corpus_tuples,
    make_prior,
    np,
    os,
    pd,
    score_tuples,
)


def families():
    p = pd.read_csv(os.path.join(ROOT, "data", "NewPhysics.csv"))

    def grp(f):
        m = re.match(r"^(I{1,3})\.(\d+)", str(f))
        return f"{m.group(1)}.{m.group(2)}" if m else str(f)

    g = p.Filename.map(grp).to_numpy()
    return [np.where(g == u)[0] for u in pd.unique(g)]


def run_config(args):
    n, rep = args
    rkw = {"root_context": True} if rep == "root" else {}
    T = corpus_tuples("physics", n, root_context=rkw.get("root_context", False))
    N = len(T)
    rows = []
    for gi, f in enumerate(families()):
        fs = set(f.tolist())
        train = [T[i] for i in range(N) if i not in fs]
        for name in ("katz", "katz_norm", "kn_mkn", "kn_tuned"):
            # katz_norm: the Katz model with every conditional distribution made proper
            p = (make_prior(n, "katz", smoothing_kwargs={"normalise": True}, **rkw)
                 if name == "katz_norm" else
                 make_prior(n, "katz" if name == "katz" else "kn", **rkw))
            if name == "kn_tuned":
                p.eq_tuples = train
                p.tune_discounts("single", k=5, seed=gi,
                                 grid=np.linspace(0.05, 1.0, 20))
            else:
                p.fit(train)
            for i in f:
                rows.append(dict(eq=int(i), family=gi, n=n, rep=rep, model=name,
                                 **score_tuples(p, T[i])))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    ns = (1, 2, 3, 4, 6, 8, 10)
    configs = [(n, rep) for n in ns for rep in ("base", "root")]
    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        for n, rep in configs:
            corpus_tuples("physics", n, root_context=(rep == "root"))
        sys.exit()
    from multiprocessing import Pool

    nproc = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    with Pool(min(nproc, len(configs))) as pool:
        res = pool.map(run_config, configs, chunksize=1)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(RESULTS, "cv_grouped.csv"), index=False)
    for (n, rep), g in df.groupby(["n", "rep"]):
        piv = g.pivot(index="eq", columns="model", values="total")
        fin = np.isfinite(piv).all(axis=1)
        msg = f"n={n:2d} {rep:4s} N={fin.sum()} "
        for m in piv.columns:
            msg += f"| {m} {-piv.loc[fin, m].mean():.3f} "
            if m != "katz":
                d = piv.loc[fin, m] - piv.loc[fin, "katz"]
                msg += f"(gain {d.mean():+.3f}+-{d.std(ddof=1) / np.sqrt(fin.sum()):.3f}) "
        print(msg)
    print(f"time {time.time() - t0:.1f}s")
