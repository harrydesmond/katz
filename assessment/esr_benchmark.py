"""Experiment 7: does the function prior help ESR recover known generating functions?

Reproduces the benchmark design of Bartlett, Desmond & Ferreira (2023, Fig. 2; see
scripts/fit_benchmark.py): Nguyen-8 and Korns-1/4/6/7 mocks with Gaussian noise of half the
standard deviation of the function over its domain, N = 10..10^4 points, 5 random seeds,
operator basis {x, a, sqrt, square, sin, cos, +, *, -, /, pow}, complexity 1..8.

The ESR fits (likelihood and parameter codelength) do not depend on the function prior, so
each mock is fitted once with the default MDL prior; every alternative prior is then applied
by re-running combine_DL with a different function-prior file (one value per tree of the
function library, as in scripts/fit_benchmark.apply_language_prior).

Usage (MPI ranks via Slurm, see jobs/run_bench_*.sh):
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
    # Korns-4 is -2.3 + 0.13 sin(x) (Korns 2011; McDermott et al. 2012; scripts/fit_benchmark.py);
    # Table 1 of Bartlett et al. (2023) prints -2.13. The offset does not affect the recovery
    # problem, since the noise is set by the standard deviation of the function.
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

            failed = []
            for j in range(len(lines)):
                if j < lo or j > hi:
                    continue
                tree = re.sub(r"np\.str_\((['\"])(.*?)\1\)", r"\1\2\1", lines[j])
                tree = re.sub(r"(['\"])\s+(['\"])", r"\1, \2", tree)
                try:
                    t = kp.coder.process_all_equations(kp.n + 1, [ast.literal_eval(tree)],
                                                       kp.maxvar)
                    mine[j] = -kp.logprior_tuples(t)  # inf for zero prior probability
                except Exception as e:  # noqa: BLE001
                    failed.append((j, lines[j], repr(e)))  # stays NaN
            allf = comm.gather(failed, root=0)
            allv = comm.gather(mine, root=0)
            if rank == 0:
                allf = [x for f in allf for x in f]
                if allf:
                    print(f"   prior {label} complexity {c}: {len(allf)} trees could not be "
                          f"scored, e.g. {allf[:3]}", flush=True)
                    if len(allf) > 1e-3 * len(lines):
                        comm.Abort(1)  # too many failures for the benchmark to be valid
                # each line was filled by one rank; all-NaN only for failed trees
                vals = np.where(np.isnan(np.vstack(allv)), np.inf, np.vstack(allv)).min(0)
                vals[[j for j, _, _ in allf]] = np.nan
                logconst = np.atleast_1d(np.loadtxt(
                    os.path.join(LIB, f"compl_{c}", f"logconst_{c}.txt")))
                out = vals + logconst
                np.savetxt(os.path.join(LIB, f"compl_{c}", f"{label}_{c}.txt"), out)
                print(f"   prior {label} complexity {c}: {len(out)} trees, "
                      f"{np.sum(np.isinf(out))} with zero prior, {np.sum(np.isnan(out))} "
                      "failed", flush=True)
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
    """All functions of complexity 1..CMAX in the final files: (fcn, DL, negloglike, params)"""
    import csv

    fun, dl, nll, par = [], [], [], []
    for c in range(1, CMAX + 1):
        fname = os.path.join(out_dir, f"{prefix}{c}.dat")
        if not os.path.isfile(fname):
            return None
        with open(fname) as f:
            for row in csv.reader(f, delimiter=";"):
                fun.append(row[1])
                dl.append(float(row[2]))
                nll.append(float(row[4]))
                par.append([float(v) for v in row[7:]])
    npar = max((len(p) for p in par), default=0)
    par = np.array([p + [0.0] * (npar - len(p)) for p in par])
    return fun, np.array(dl), np.array(nll), par


def fitted_curve(f, params, x):
    """Values of the fitted function f (an ESR string in x, a0, a1, ...) at x"""
    xs = sympy.Symbol("x", real=True)
    a = sympy.symbols([f"a{i}" for i in range(len(params))], real=True)
    loc = {"x": xs, **{f"a{i}": a[i] for i in range(len(params))}}
    fun = sympy.lambdify([xs] + list(a), sympy.sympify(f, locals=loc), modules="numpy")
    with np.errstate(all="ignore"):
        return np.broadcast_to(np.asarray(fun(x, *params), dtype=float), np.shape(x))


NLL_RTOL = 1e-6  # likelihoods equal to this precision are treated as the same function


def truth_preference(fun, dl, nll, truths, params=None, x=None):
    """Preference for the truth in an MDL ranking, following
    scripts/fit_benchmark.process_fit: functions with the same likelihood are treated as
    one function, represented by its member of lowest description length (DL). A group
    counts as the truth if any member is one of the listed variants of the truth, so an
    equally good unlisted representation does not hide it. Non-finite rows are dropped.

    Returns (DL(best other) - DL(truth), rank of the truth, best other function,
    whether the best other function's fitted curve matches the truth's to 1e-4 of the
    data range, which flags an unlisted equivalent of the truth; None if not checked)."""
    ok = np.isfinite(dl) & np.isfinite(nll)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return np.nan, np.nan, None, None
    idx = idx[np.argsort(nll[idx], kind="stable")]
    # groups of (relatively) equal likelihood
    gid = np.zeros(len(idx), dtype=int)
    for j in range(1, len(idx)):
        a, b = nll[idx[j - 1]], nll[idx[j]]
        gid[j] = gid[j - 1] + (abs(b - a) > NLL_RTOL * max(1.0, abs(a)))
    ng = gid[-1] + 1
    gdl = np.full(ng, np.inf)
    grep = np.zeros(ng, dtype=int)
    gtrue = np.zeros(ng, dtype=bool)
    tdl, trep = np.full(ng, np.inf), np.zeros(ng, dtype=int)
    for g, i in zip(gid, idx):
        if dl[i] < gdl[g]:
            gdl[g], grep[g] = dl[i], i
        if fun[i] in truths:
            gtrue[g] = True
            if dl[i] < tdl[g]:
                tdl[g], trep[g] = dl[i], i
    if not gtrue.any():
        return np.nan, np.nan, None, None
    t = gdl[gtrue].min()
    other = gdl[~gtrue]
    if len(other) == 0:
        return np.inf, 1, None, None
    best = grep[~gtrue][np.argmin(other)]
    equiv = None
    if params is not None and x is not None:
        try:
            it = trep[gtrue][np.argmin(gdl[gtrue])]
            ft = fitted_curve(fun[it], params[it], x)
            fo = fitted_curve(fun[best], params[best], x)
            scale = max(np.ptp(ft), np.abs(ft).max(), 1e-300)
            equiv = bool(np.all(np.abs(fo - ft) <= 1e-4 * scale))
        except Exception:  # noqa: BLE001
            equiv = None
    return other.min() - t, 1 + int(np.sum(other < t)), fun[best], equiv


def collect():
    import pandas as pd

    rows = []
    for i, (name, nx, s) in enumerate(MOCKS):
        out_dir = os.path.join(OUT, "output_" + mock_name(i))
        x = None
        dfile = os.path.join(DATA, mock_name(i) + ".txt")
        if os.path.isfile(dfile):
            x = np.loadtxt(dfile)[:, 0]
        for label in ["mdl"] + list(PRIORS):
            prefix = "final_" if label == "mdl" else f"final_{label}_"
            res = read_final(out_dir, prefix)
            if res is None:
                continue
            fun, dl, nll, par = res
            pref, rk, best, equiv = truth_preference(fun, dl, nll, BENCHMARKS[name][2],
                                                     params=par, x=x)
            rows.append(dict(bench=name, N=nx, seed=s, prior=label, pref=pref, rank=rk,
                             best_other=best, best_other_equiv=equiv,
                             n_nonfinite=int(np.sum(~(np.isfinite(dl) & np.isfinite(nll))))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(WORK, "benchmark_results.csv"), index=False)
    if len(df):
        pd.set_option("display.width", 200)
        print(df.groupby(["bench", "prior", "N"]).pref.median().unstack("N").round(2))
        print(df.assign(top=df["rank"] == 1).groupby(["bench", "prior"]).top.mean()
              .unstack("prior").round(2))
        print("truth not found:", int(df.pref.isna().sum()), "of", len(df),
              "| best other numerically equivalent to the truth:",
              int((df.best_other_equiv == True).sum()))  # noqa: E712


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
