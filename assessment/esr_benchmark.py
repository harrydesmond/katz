"""Experiment 7: does the function prior help ESR recover known generating functions?

Reproduces the benchmark design of Bartlett, Desmond & Ferreira (2023, Fig. 2; see
scripts/fit_benchmark.py): Nguyen-8 and Korns-1/4/6/7 mocks with Gaussian noise of half the
standard deviation of the function over its domain, N = 10..10^4 points, 5 random seeds,
operator basis {x, a, sqrt, square, sin, cos, +, *, -, /, pow}, complexity 1..8.

The ESR fits (likelihood and parameter codelength) do not depend on the function prior, so
each mock is fitted once with the default MDL prior; every alternative prior is then applied
by re-running combine_DL with a different function-prior file (one value per tree of the
function library, as in scripts/fit_benchmark.apply_language_prior).

Usage (MPI ranks via Slurm, see run_bench_*.sh):
    python3 esr_benchmark.py gen C                 # generate the function library, complexity C
    python3 esr_benchmark.py mocks                 # write all mock data files
    python3 esr_benchmark.py fit I [CMIN CMAX]     # fit mock number I (0..174)
    python3 esr_benchmark.py priors                # function-prior files for every prior
    python3 esr_benchmark.py combine I             # apply every prior to mock I
    python3 esr_benchmark.py collect               # rank the truth under every method
"""

import itertools
import os
import sys
import time
import warnings

import numpy as np
import sympy
from mpi4py import MPI

warnings.filterwarnings("ignore")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
WORK = os.environ.get("BENCH_DIR", os.path.join(HERE, "bench_work"))
LIB = os.path.join(WORK, "lib")
DATA = os.path.join(WORK, "data")
OUT = os.path.join(WORK, "output")
FN_SET = "new_osc_maths"
CMAX = int(os.environ.get("BENCH_CMAX", "8"))

BENCHMARKS = {  # name: (function, x range, variants of the truth)  [scripts/fit_benchmark.py]
    "nguyen_8": ("sqrt(x)", [0, 4], ["sqrt(x)"]),
    "korns_1": ("1.57 + 2.43 * x", [-50, 50], [
        "a0 + a1*x", "a0 + x/a1", "a0*(-a1 + x)", "a0*(a1 - x)", "a0*(a1 + x)",
        "(-a0 + x)/pow(Abs(a1),(1/4))"]),
    "korns_4": ("-2.3 + 0.13 * sin(x)", [-50, 50], [
        "a0 + a1*sin(x)", "a0 - sin(x)*cos(a1)", "a0 + sin(x)*cos(a1)"]),
    "korns_6": ("1.3 + 0.13 * sqrt(x)", [0, 50], [
        "a0 + a1*sqrt(x)", "a0 + sqrt(x)*sqrt(Abs(a1))", "a0*(a1 - sqrt(x))"]),
    "korns_7": ("213.80940889 * (1 - exp(-0.54723748542 * x))", [0, 50], [
        "a0 - pow(Abs(a1),x)", "a0 + pow(Abs(a1),x)/a2", "a0 + a1/pow(Abs(a2),x)",
        "a0*(a1 - pow(Abs(a2),x))", "a0 + a1*pow(Abs(a2),x)",
        "a0 + a1*pow(Abs(a2),(2*x))", "a0 - pow(Abs(a1),x)/sqrt(Abs(a2))",
        "(a0 - pow(Abs(a1),x))/sqrt(Abs(a2))",
        "a0 - pow(Abs(a1),(x/2))*sqrt(Abs(1/a2))", "a0*(cos(a1) - pow(Abs(a2),x))",
        "a0 - pow(Abs(a1),x)/cos(a2)", "(a0 - pow(Abs(a1),x))/cos(a2)",
        "a0*(-sin(a2) + pow(Abs(a1),x))"]),
}
ALL_N = [10, 30, 100, 300, 1000, 3000, 10000]
NSAMP = 5
FRAC_SIG = 0.5
MOCKS = [(name, nx, s) for name in BENCHMARKS for nx in ALL_N for s in range(NSAMP)]

# Priors: label -> KatzPrior settings (None = ESR default k log n prior, 'aifeyn')
PRIOR_BASIS = [
    ["a", "x"],
    ["sqrt", "square", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
    ["+", "-", "*", "/", "pow"],
]
PRIORS = {
    "katz2": dict(n=2, smoothing="katz"),  # the paper's language-model prior
    "kn2": dict(n=2, smoothing="kn", tune=True),  # smoothing changed only
    "katz3r": dict(n=3, smoothing="katz", root_context=True),  # representation only
    "kn3r": dict(n=3, smoothing="kn", root_context=True, tune=True),  # recommended
    "kn3rc": dict(n=3, smoothing="kn", root_context=True, code_constants=True,
                  tune=True),
}


def mock_name(i):
    name, nx, s = MOCKS[i]
    return f"{name}_{nx}_{FRAC_SIG}_{s}"


def likelihood(i):
    from esr.fitting.likelihood import GaussLikelihood

    return GaussLikelihood(mock_name(i) + ".txt", mock_name(i), data_dir=DATA,
                           fn_set=FN_SET, fn_dir=LIB, base_out_dir=OUT)


def say(*a):
    if rank == 0:
        print(*a, flush=True)


def make_mocks():
    """Identical to scripts/fit_benchmark.get_sig and make_data"""
    if rank != 0:
        return
    os.makedirs(DATA, exist_ok=True)
    x = sympy.Symbol("x")
    for name, (f, xr, _) in BENCHMARKS.items():
        fun = sympy.lambdify(x, sympy.sympify(f), modules="numpy")
        np.random.seed(0)
        sig = FRAC_SIG * np.std(fun(np.sort(np.random.uniform(*xr, 10000))))
        for nx, s in itertools.product(ALL_N, range(NSAMP)):
            np.random.seed(s)
            xd = np.sort(np.random.uniform(*xr, nx))
            yd = fun(xd) + np.random.normal(size=nx) * sig
            np.savetxt(os.path.join(DATA, f"{name}_{nx}_{FRAC_SIG}_{s}.txt"),
                       np.vstack([xd, yd, sig * np.ones(nx)]).T)
    print(f"wrote {len(MOCKS)} mocks to {DATA}")


def fit(i, cmin=1, cmax=CMAX):
    from esr.fitting import combine_DL, match, test_all, test_all_Fisher

    like = likelihood(i)
    for c in range(cmin, cmax + 1):
        if os.path.isfile(os.path.join(like.out_dir, f"final_{c}.dat")):
            say(f"FIT {mock_name(i)} complexity {c} already done")
            continue
        t0 = time.time()
        test_all.main(c, like, tmax=5)
        comm.Barrier()
        test_all_Fisher.main(c, like, tmax=5)
        comm.Barrier()
        match.main(c, like, tmax=5)
        comm.Barrier()
        combine_DL.main(c, like)
        comm.Barrier()
        say(f"FIT {mock_name(i)} complexity {c} done in {time.time() - t0:.1f}s")


def make_prior(label):
    from katz.prior import KatzPrior
    import pandas as pd

    kw = dict(PRIORS[label])
    n = kw.pop("n")
    tune = kw.pop("tune", False)
    eqs = list(pd.read_csv(os.path.join(REPO, "data", "NewPhysics.csv"))["New Formula"])
    kp = KatzPrior(n, PRIOR_BASIS, equations=eqs, **kw)
    if tune:
        res = kp.tune_discounts("single", k=10)
        say(f"   {label}: tuned discounts left {res['left']} right {res['right']}")
    return kp


def write_priors():
    """{label}_{c}.txt in the library: -log prior + sum log|c_i| for every tree"""
    from katz.esr_prior import get_logconst

    for c in range(1, CMAX + 1):
        get_logconst(c, LIB + "/", overwrite=False)  # parallel over ranks
    comm.Barrier()
    for label in PRIORS:
        kp = make_prior(label)
        for c in range(1, CMAX + 1):
            fname = os.path.join(LIB, f"compl_{c}", f"trees_{c}.txt")
            with open(fname) as f:
                lines = f.read().splitlines()
            lo, hi = np.array_split(np.arange(len(lines)), size)[rank][[0, -1]] \
                if len(lines) >= size else (0, len(lines) - 1)
            mine = np.full(len(lines), np.nan)
            import ast
            import re

            for j in range(len(lines)):
                if j < lo or j > hi:
                    continue
                tree = re.sub(r"np\.str_\((['\"])(.*?)\1\)", r"\1\2\1", lines[j])
                tree = re.sub(r"(['\"])\s+(['\"])", r"\1, \2", tree)
                try:
                    t = kp.coder.process_all_equations(kp.n + 1, [ast.literal_eval(tree)],
                                                       kp.maxvar)
                    mine[j] = -kp.logprior_tuples(t)
                except Exception:  # noqa: BLE001
                    mine[j] = np.inf
            allv = comm.gather(mine, root=0)
            if rank == 0:
                vals = np.nanmin(np.vstack(allv), axis=0)  # each line filled by one rank
                logconst = np.atleast_1d(np.loadtxt(
                    os.path.join(LIB, f"compl_{c}", f"logconst_{c}.txt")))
                out = vals + logconst
                np.savetxt(os.path.join(LIB, f"compl_{c}", f"{label}_{c}.txt"), out)
                print(f"   prior {label} complexity {c}: {len(out)} trees, "
                      f"{np.sum(~np.isfinite(out))} infinite", flush=True)
            comm.Barrier()


def combine(i, cmin=1, cmax=CMAX):
    from esr.fitting import combine_DL

    like = likelihood(i)
    for label in PRIORS:
        like.fnprior_prefix = f"{label}_"
        like.combineDL_prefix = f"combine_DL_{label}_"
        like.final_prefix = f"final_{label}_"
        for c in range(cmin, cmax + 1):
            combine_DL.main(c, like)
            comm.Barrier()
    say(f"COMBINE {mock_name(i)} done")


def read_final(out_dir, prefix):
    """All functions of complexity 1..CMAX in the final files: (fcn, DL, negloglike)"""
    import csv

    fun, dl, nll = [], [], []
    for c in range(1, CMAX + 1):
        fname = os.path.join(out_dir, f"{prefix}{c}.dat")
        if not os.path.isfile(fname):
            return None
        with open(fname) as f:
            for row in csv.reader(f, delimiter=";"):
                fun.append(row[1])
                dl.append(float(row[2]))
                nll.append(float(row[4]))
    return fun, np.array(dl), np.array(nll)


def truth_preference(fun, dl, nll, truths):
    """As scripts/fit_benchmark.process_fit (MDL methods): sort by DL, drop duplicates
    (same likelihood, then same string), then compare the best truth variant with the
    best other function. Returns (DL(best other) - DL(truth), rank of the truth)."""
    m = np.argsort(dl, kind="stable")
    fun = [fun[j] for j in m]
    dl, nll = dl[m], nll[m]
    _, u = np.unique(nll, return_index=True)
    u = np.sort(u)
    fun = [fun[j] for j in u]
    dl = dl[u]
    is_true = np.array([f in truths for f in fun])
    if not is_true.any():
        return np.nan, np.nan
    t = dl[is_true].min()
    other = dl[~is_true]
    return other.min() - t, 1 + int(np.sum(other < t))


def collect():
    import pandas as pd

    rows = []
    for i, (name, nx, s) in enumerate(MOCKS):
        out_dir = os.path.join(OUT, "output_" + mock_name(i))
        for label in ["mdl"] + list(PRIORS):
            prefix = "final_" if label == "mdl" else f"final_{label}_"
            res = read_final(out_dir, prefix)
            if res is None:
                continue
            pref, rk = truth_preference(*res, BENCHMARKS[name][2])
            rows.append(dict(bench=name, N=nx, seed=s, prior=label, pref=pref, rank=rk))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(WORK, "benchmark_results.csv"), index=False)
    if len(df):
        pd.set_option("display.width", 200)
        print(df.groupby(["bench", "prior", "N"]).pref.median().unstack("N").round(2))
        print(df.assign(top=df["rank"] == 1).groupby(["bench", "prior"]).top.mean()
              .unstack("prior").round(2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    t0 = time.time()
    if cmd == "gen":
        import esr.generation.duplicate_checker as dc

        c = int(sys.argv[2])
        done = all(os.path.isfile(os.path.join(LIB, f"compl_{c}", f"{f}_{c}.txt"))
                   for f in ("all_equations", "trees", "aifeyn", "unique_equations"))
        if done:
            say(f"library complexity {c} already generated")
        else:
            dc.main(FN_SET, c, fn_dir=LIB)
    elif cmd == "mocks":
        make_mocks()
    elif cmd == "fit":
        args = [int(a) for a in sys.argv[2:]]
        fit(*args)
    elif cmd == "priors":
        write_priors()
    elif cmd == "combine":
        combine(*[int(a) for a in sys.argv[2:]])
    elif cmd == "collect":
        if rank == 0:
            collect()
    say(f"{cmd} finished in {time.time() - t0:.1f}s")
