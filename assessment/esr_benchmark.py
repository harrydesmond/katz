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
    python3 esr_benchmark.py collect [I [OUT]]     # rank the truth under every method
    python3 esr_benchmark.py manifest              # record settings, checksums and versions
"""

import os
import sys
import time
import warnings

import numpy as np
import sympy

warnings.filterwarnings("ignore")
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:  # the mock definitions can be used without MPI (e.g. operon_bloat.py)
    comm, rank, size = None, 0, 1

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
    # Korns-4 is -2.3 + 0.13 sin(x) as in Korns (2011), McDermott et al. (2012) and
    # scripts/fit_benchmark.py, which produced the results of Bartlett et al. (2023); their
    # Table 1 prints -2.13. The noise level (half the standard deviation of the function) does
    # not depend on the offset, but competitors that are not translation-invariant, such as
    # a0/(a1 + sin(x)), fit differently, so results for -2.13 could differ slightly.
    "korns_4": ("-2.3 + 0.13 * sin(x)", [-50, 50], [
        "a0 + a1*sin(x)", "a0 - sin(x)*cos(a1)", "a0 + sin(x)*cos(a1)"]),
    "korns_6": ("1.3 + 0.13 * sqrt(x)", [0, 50], [
        "a0 + a1*sqrt(x)", "a0 + sqrt(x)*sqrt(Abs(a1))", "a0*(a1 - sqrt(x))"]),
    # scripts/fit_benchmark.py also lists a0 - pow(Abs(a1),x), which cannot represent the
    # truth (its amplitude is fixed at 1); every variant below reproduces the generating
    # function exactly on [0, 50] (tests/test_katz.py::TestBenchmarkTruth)
    "korns_7": ("213.80940889 * (1 - exp(-0.54723748542 * x))", [0, 50], [
        "a0 + pow(Abs(a1),x)/a2", "a0 + a1/pow(Abs(a2),x)",
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
# Sensitivity check: Korns-4 with the offset printed in Table 1 of Bartlett et al. (2023).
# Appended after the main grid so that the indices of the main mocks (0..174) are unchanged.
BENCHMARKS["korns_4_213"] = ("-2.13 + 0.13 * sin(x)", [-50, 50], BENCHMARKS["korns_4"][2])
MOCKS += [("korns_4_213", nx, s) for nx in (10, 100, 1000) for s in range(NSAMP)]

# Priors: label -> KatzPrior settings (None = ESR default k log n prior, 'aifeyn')
PRIOR_BASIS = [
    ["a", "x"],
    ["sqrt", "square", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
    ["+", "-", "*", "/", "pow"],
]
PRIORS = {
    "katz2": dict(n=2, smoothing="katz", root_context=False),  # the paper's prior
    "kn2": dict(n=2, smoothing="kn", root_context=False, tune=True),  # smoothing changed only
    "katz3r": dict(n=3, smoothing="katz", root_context=True),  # n = 3 and root context
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


def mock_data(name, nx, s):
    """Mock data set (x, y, sigma) for one benchmark, sample size and seed: identical to
    scripts/fit_benchmark.get_sig and make_data"""
    f, xr, _ = BENCHMARKS[name]
    x = sympy.Symbol("x")
    fun = sympy.lambdify(x, sympy.sympify(f), modules="numpy")
    np.random.seed(0)
    sig = FRAC_SIG * np.std(fun(np.sort(np.random.uniform(*xr, 10000))))
    np.random.seed(s)
    xd = np.sort(np.random.uniform(*xr, nx))
    yd = fun(xd) + np.random.normal(size=nx) * sig
    return xd, yd, sig


def make_mocks():
    if rank != 0:
        return
    os.makedirs(DATA, exist_ok=True)
    n = 0
    for name, nx, s in MOCKS:
        fname = os.path.join(DATA, f"{name}_{nx}_{FRAC_SIG}_{s}.txt")
        if os.path.isfile(fname):  # never rewrite data that fits may be reading
            continue
        xd, yd, sig = mock_data(name, nx, s)
        np.savetxt(fname, np.vstack([xd, yd, sig * np.ones(nx)]).T)
        n += 1
    print(f"wrote {n} new mocks to {DATA} ({len(MOCKS)} in total)")


NP_ERR = np.geterr()


def guard_numpy_errstate():
    """ESR stops slow fits with a SIGALRM timer (simplifier.time_limit). If the timer fires
    while numpy's error state is being switched, e.g. as scipy's line search enters or leaves
    np.errstate(over='raise', divide='raise', invalid='raise'), the exception escapes before
    the old state is restored. The process then raises FloatingPointError on every later
    overflow: fits fail silently, and ESR's Hessian checks in match.main crash one MPI rank
    while the others wait forever. Restore numpy's error state on leaving every timed block."""
    import contextlib

    import esr.fitting.likelihood as esr_likelihood
    import esr.generation.simplifier as simplifier

    if getattr(simplifier.time_limit, "katz_guarded", False):
        return
    unguarded = simplifier.time_limit

    @contextlib.contextmanager
    def time_limit(seconds):
        state = np.geterr()
        try:
            with unguarded(seconds):
                yield
        finally:
            np.seterr(**state)

    time_limit.katz_guarded = True
    simplifier.time_limit = time_limit
    esr_likelihood.time_limit = time_limit


def check_errstate(where):
    """Report (and reset) a numpy error state left changed by ESR"""
    if np.geterr() != NP_ERR:
        print(f"rank {rank}: numpy error state {np.geterr()} after {where}; reset", flush=True)
        np.seterr(**NP_ERR)


def esr_modules():
    """ESR fitting modules (renamed to lower case in ESR after v0.2.1)"""
    from esr.fitting import match, test_all

    try:
        from esr.fitting import combine_dl, test_all_fisher
    except ImportError:
        from esr.fitting import combine_DL as combine_dl
        from esr.fitting import test_all_Fisher as test_all_fisher
    return test_all, test_all_fisher, match, combine_dl


def fit(i, cmin=1, cmax=CMAX):
    test_all, test_all_Fisher, match, combine_DL = esr_modules()

    guard_numpy_errstate()
    like = likelihood(i)
    for c in range(cmin, cmax + 1):
        if os.path.isfile(os.path.join(like.out_dir, f"final_{c}.dat")):
            say(f"FIT {mock_name(i)} complexity {c} already done")
            continue
        t0 = time.time()
        test_all.main(c, like, tmax=5)
        check_errstate(f"test_all {c}")
        comm.Barrier()
        test_all_Fisher.main(c, like, tmax=5)
        check_errstate(f"test_all_Fisher {c}")
        comm.Barrier()
        match.main(c, like, tmax=5)
        check_errstate(f"match {c}")
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
    """{label}_{c}.txt in the library: -log prior + sum log|c_i| for every tree. Each rank
    scores a contiguous chunk of trees; trees that cannot be scored are NaN (a zero prior
    probability is +inf) and are listed with the exception in {label}_{c}_failures.txt.
    ESR's combine_DL skips functions with a non-finite description length, so both drop out
    of the ranking."""
    import ast
    import re

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
            chunk = np.array_split(np.arange(len(lines)), size)[rank]
            mine = np.full(len(chunk), np.nan)
            failed = []
            for k, j in enumerate(chunk):
                tree = re.sub(r"np\.str_\((['\"])(.*?)\1\)", r"\1\2\1", lines[j])
                tree = re.sub(r"(['\"])\s+(['\"])", r"\1, \2", tree)
                try:
                    t = kp.coder.process_all_equations(kp.n + 1, [ast.literal_eval(tree)],
                                                       kp.maxvar)
                    mine[k] = -kp.logprior_tuples(t)  # inf for zero prior probability
                except Exception as e:  # noqa: BLE001
                    failed.append((int(j), lines[j], repr(e)))  # stays NaN
            allf = comm.gather(failed, root=0)
            allv = comm.gather(mine, root=0)
            if rank == 0:
                allf = [x for f in allf for x in f]
                vals = np.concatenate(allv)  # chunks are in rank order
                assert len(vals) == len(lines)
                ffile = os.path.join(LIB, f"compl_{c}", f"{label}_{c}_failures.txt")
                with open(ffile, "w") as f:
                    f.writelines(f"{j}\t{tree}\t{err}\n" for j, tree, err in allf)
                if allf:
                    print(f"   prior {label} complexity {c}: {len(allf)} trees could not be "
                          f"scored (listed in {ffile}), e.g. {allf[:3]}", flush=True)
                    if len(allf) > 1e-3 * len(lines):
                        comm.Abort(1)  # too many failures for the benchmark to be valid
                logconst = np.atleast_1d(np.loadtxt(
                    os.path.join(LIB, f"compl_{c}", f"logconst_{c}.txt")))
                out = vals + logconst
                np.savetxt(os.path.join(LIB, f"compl_{c}", f"{label}_{c}.txt"), out)
                print(f"   prior {label} complexity {c}: {len(out)} trees, "
                      f"{np.sum(np.isinf(out))} with zero prior, {np.sum(np.isnan(out))} "
                      "failed", flush=True)
            comm.Barrier()


def combine(i, cmin=1, cmax=CMAX):
    combine_DL = esr_modules()[3]

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


_LAMBDA = {}


def _lambda(f):
    """Numpy function of an ESR string f in x, a0, a1, ...: (callable, number of parameters)"""
    if f not in _LAMBDA:
        import re

        k = max([int(i) + 1 for i in re.findall(r"(?<![A-Za-z_0-9])a(\d+)", f)] + [0])
        xs = sympy.Symbol("x", real=True)
        a = sympy.symbols([f"a{i}" for i in range(k)], real=True)
        loc = {"x": xs, **{f"a{i}": a[i] for i in range(k)}}
        fun = sympy.lambdify([xs] + list(a), sympy.sympify(f, locals=loc), modules="numpy")
        _LAMBDA[f] = (fun, k)
    return _LAMBDA[f]


def fitted_curve(f, params, x):
    """Values of the fitted function f (an ESR string in x, a0, a1, ...) at x"""
    fun, k = _lambda(f)
    with np.errstate(all="ignore"):
        return np.broadcast_to(np.asarray(fun(x, *np.asarray(params)[:k]), dtype=float),
                               np.shape(x)).astype(float)


FAMILY_TOL = 1e-6  # exact reproduction: max deviation / range of the target curve
_FAMILY = {}
_REF = {}


def _starts(f, k, rng, around=None, n=20):
    """Starting points for fitting f: near `around` if given, then random ones on several
    scales, with the bases of exponentials pow(Abs(a_i), x) drawn in (0.3, 1.8)"""
    import re

    bases = [int(j) for j in re.findall(r"pow\(Abs\(a(\d+)\)", f) if int(j) < k]
    out = []
    if around is not None:
        around = np.asarray(around, dtype=float)[:k]
        if len(around) == k and np.all(np.isfinite(around)):
            out += [around, around * (1 + 0.05 * rng.standard_normal(k))]
    for _ in range(n):
        p = rng.standard_normal(k) * rng.choice([0.01, 1.0, 10.0, 100.0, 300.0], k)
        p[bases] = rng.uniform(0.3, 1.8, len(bases))
        out.append(p)
    return out


def _fit_curve(f, T, xs, rng, around=None, tol=FAMILY_TOL):
    """Parameters with which f reproduces the curve T on xs to `tol` of its range (an
    offset-invariant scale), or None if no start reaches that"""
    import scipy.optimize

    k = _lambda(f)[1]
    ok = np.isfinite(T)
    if ok.mean() < 0.9:
        return None
    ptp = np.ptp(T[ok])
    scale = ptp if ptp > 0 else max(np.abs(T[ok]).max(), 1.0)

    def resid(p):
        r = fitted_curve(f, p, xs[ok]) - T[ok]
        return np.where(np.isfinite(r), r / scale, 1e6)

    if k == 0:
        return np.zeros(0) if np.max(np.abs(resid(np.zeros(0)))) <= tol else None
    for p0 in _starts(f, k, rng, around):
        try:
            r = scipy.optimize.least_squares(resid, p0, xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                             max_nfev=400 * (k + 1))
        except (ValueError, FloatingPointError):
            continue
        if np.max(np.abs(r.fun)) <= tol:
            return r.x
    return None


def _draws(p, rng, n):
    """n parameter vectors near p: 10% relative scatter plus a small absolute one, so that
    parameters at zero are varied too"""
    p = np.asarray(p, dtype=float)
    return [p * (1 + 0.1 * rng.standard_normal(len(p)))
            + 0.002 * np.maximum(np.abs(p), 1.0) * rng.standard_normal(len(p))
            for _ in range(n)]


def _equivalent_at(f, p_f, g, p_g, xs, rng, ndraw=3):
    """Given f(x; p_f) = g(x; p_g) on xs, whether each function also reproduces the other
    for `ndraw` parameter values near these, i.e. neither is a special case of the other"""
    for a, pa, b, pb in ((f, p_f, g, p_g), (g, p_g, f, p_f)):
        good = 0
        for theta in _draws(pa, rng, 10 * ndraw):
            T = fitted_curve(a, theta, xs)
            if np.isfinite(T).mean() < 0.9:
                continue  # e.g. sqrt of a parameter pushed negative
            if _fit_curve(b, T, xs, rng, around=pb) is None:
                return False
            good += 1
            if good == ndraw:
                break
        if good < ndraw:
            return False
    return True


def same_family(f, g, p_f, p_g, xr, npts=400, seed=0):
    """Whether g(x; b) and f(x; a) describe the same functions on the domain xr near
    f(x; p_f): g reproduces f(x; p_f) exactly (to FAMILY_TOL of the curve's range, which is
    offset-invariant), and each reproduces the other for parameters near that point. This
    identifies representations such as x**0.5 for sqrt(x) and rejects models that only fit
    equally well (e.g. pow(x, a0) with a0 ~ 0.5). p_g is only a starting point."""
    key = (f, g, tuple(xr), tuple(np.round(np.asarray(p_f, dtype=float), 4)))
    if key not in _FAMILY:
        xs = np.linspace(xr[0], xr[1], npts)
        _FAMILY[key] = False
        # a local optimiser can miss an exact solution, but cannot produce one by chance:
        # retry with other random starts and accept the first success
        for s in range(seed, seed + 3):
            rng = np.random.default_rng(s)
            try:
                q = _fit_curve(g, fitted_curve(f, p_f, xs), xs, rng, around=p_g)
                if q is not None and _equivalent_at(f, p_f, g, q, xs, rng):
                    _FAMILY[key] = True
                    break
            except Exception:  # noqa: BLE001  (unparseable string)
                break
    return _FAMILY[key]


def truth_reference(bench, npts=400, seed=0):
    """The first listed variant of the truth and parameters with which it reproduces the
    generating function of the benchmark exactly"""
    if bench not in _REF:
        f, xr, variants = BENCHMARKS[bench]
        xs = np.linspace(xr[0], xr[1], npts)
        T = fitted_curve(f, [], xs)
        rng = np.random.default_rng(seed)
        for v in variants:
            p = _fit_curve(v, T, xs, rng)
            if p is not None:
                _REF[bench] = (v, p)
                break
        else:
            raise ValueError(f"no listed variant reproduces the truth of {bench}")
    return _REF[bench]


def same_family_as_truth(g, bench, p_g=None):
    """Whether g is the same family as the generating function of `bench` (same_family
    with the truth at its generating parameters; independent of any data set, so cached
    by benchmark and function). p_g, e.g. the fitted parameters, is only a starting point."""
    v, p_v = truth_reference(bench)
    key = (bench, g)
    if key not in _FAMILY:
        _FAMILY[key] = same_family(v, g, p_v, p_g, BENCHMARKS[bench][1])
    return _FAMILY[key]


def is_null(f):
    """A single-leaf function: a constant or the bare variable"""
    import re

    return bool(re.fullmatch(r"\s*(-?a\d+|x|-x|-?[0-9.]+)\s*", f))


def truth_preference(fun, dl, nll, truths, params=None, xr=None, sig=None, max_checks=50,
                     bench=None):
    """Rank of the generating function in an MDL ranking.

    Each distinct function string is represented by its row of lowest description length
    (DL); non-finite rows are dropped. 'Strict' results count only the listed variants of
    the truth. The 'functional' results also count a competitor as the truth if it is the
    same family of functions on the domain xr (same_family; needs params and xr): the
    competitors are tested in order of DL until the first one that is not. With `bench`, the
    test is against the generating function of that benchmark (same_family_as_truth), which
    does not depend on the fitted parameters; otherwise against the best listed variant at
    its fitted parameters. Likelihoods are not used to decide equivalence.

    Returns a dict: pref = DL(best other) - DL(truth) and top (truth ranked first) for the
    functional comparison; pref_strict and rank_strict; best_other; n_equiv and equiv (the
    competitors found equivalent to the truth); best_other_maxdev/rmsdev, the maximum/RMS
    difference between the fitted curves of the best other function and of the truth on a
    dense grid over xr, in units of sig (noise) if given; best_fn and best_is_null for the
    overall winner. Values are NaN/None if the truth is not among the finite rows."""
    nan = dict(pref=np.nan, top=np.nan, pref_strict=np.nan, rank_strict=np.nan,
               best_other=None, n_equiv=0, equiv="", best_other_maxdev=np.nan,
               best_other_rmsdev=np.nan, best_fn=None, best_is_null=None)
    dl, nll = np.asarray(dl, dtype=float), np.asarray(nll, dtype=float)
    best_row = {}
    for i in np.where(np.isfinite(dl) & np.isfinite(nll))[0]:
        if fun[i] not in best_row or dl[i] < dl[best_row[fun[i]]]:
            best_row[fun[i]] = i
    if not best_row:
        return nan
    names = list(best_row)
    rows = np.array([best_row[f] for f in names])
    order = np.argsort(dl[rows], kind="stable")
    names, rows = [names[j] for j in order], rows[order]
    res = dict(nan, best_fn=names[0], best_is_null=is_null(names[0]))
    is_t = np.array([f in truths for f in names])
    if not is_t.any():
        return res
    it = rows[is_t][0]  # best listed variant of the truth
    t_strict = dl[it]
    comp = np.where(~is_t)[0]
    if len(comp) == 0:
        return dict(res, pref=np.inf, top=True, pref_strict=np.inf, rank_strict=1)
    res.update(pref_strict=dl[rows[comp[0]]] - t_strict,
               rank_strict=1 + int(np.sum(dl[rows[comp]] < t_strict)))
    # functional comparison: competitors in DL order until the first non-equivalent one
    t_func, equiv, best = t_strict, [], None
    for j in comp[:max_checks]:
        f, i = names[j], rows[j]
        if bench is not None:
            equivalent = same_family_as_truth(f, bench, None if params is None else params[i])
        elif params is not None and xr is not None:
            equivalent = same_family(fun[it], f, params[it], params[i], xr)
        else:
            equivalent = False
        if equivalent:
            equiv.append(f)
            t_func = min(t_func, dl[i])
            continue
        best = j
        break
    res.update(n_equiv=len(equiv), equiv="; ".join(equiv))
    if best is None:
        if len(equiv) < len(comp):  # max_checks reached
            return res
        return dict(res, pref=np.inf, top=True)
    ib = rows[best]
    res.update(pref=dl[ib] - t_func, top=bool(dl[ib] > t_func), best_other=names[best])
    if params is not None and xr is not None:
        xs = np.linspace(xr[0], xr[1], 400)
        dev = fitted_curve(names[best], params[ib], xs) - fitted_curve(fun[it], params[it], xs)
        dev = dev[np.isfinite(dev)] / (sig if sig else 1.0)
        if len(dev):
            res.update(best_other_maxdev=float(np.max(np.abs(dev))),
                       best_other_rmsdev=float(np.sqrt(np.mean(dev**2))))
    return res


def collect(indices=None, out=None):
    """Rank the truth under every method for all finished mocks (or only `indices`) and
    write the table to `out` (default WORK/benchmark_results.csv)"""
    import pandas as pd

    rows = []
    for i, (name, nx, s) in enumerate(MOCKS):
        if indices is not None and i not in indices:
            continue
        out_dir = os.path.join(OUT, "output_" + mock_name(i))
        sig = None
        dfile = os.path.join(DATA, mock_name(i) + ".txt")
        if os.path.isfile(dfile):
            sig = float(np.loadtxt(dfile)[0, 2])
        for label in ["mdl"] + list(PRIORS):
            prefix = "final_" if label == "mdl" else f"final_{label}_"
            res = read_final(out_dir, prefix)
            if res is None:
                continue
            fun, dl, nll, par = res
            r = truth_preference(fun, dl, nll, BENCHMARKS[name][2], params=par,
                                 xr=BENCHMARKS[name][1], sig=sig, bench=name)
            truth_rows = [j for j, f in enumerate(fun) if f in BENCHMARKS[name][2]]
            rows.append(dict(bench=name, N=nx, seed=s, prior=label, **r,
                             n_truth_rows=len(truth_rows),
                             n_truth_finite=int(np.sum(np.isfinite(dl[truth_rows]))),
                             n_nonfinite=int(np.sum(~(np.isfinite(dl) & np.isfinite(nll))))))
    df = pd.DataFrame(rows)
    df.to_csv(out or os.path.join(WORK, "benchmark_results.csv"), index=False)
    if len(df) and indices is None:
        pd.set_option("display.width", 200)
        print(df.groupby(["bench", "prior", "N"]).pref.median().unstack("N").round(2))
        print(df.assign(top=df.top.astype(float)).groupby(["bench", "prior"]).top.mean()
              .unstack("prior").round(2))
        strict_top = df.rank_strict == 1
        print("truth not found:", int(df.pref_strict.isna().sum()), "of", len(df),
              "| unlisted equivalents of the truth found:", int((df.n_equiv > 0).sum()),
              "| top differs strict vs functional:",
              int((strict_top != (df.top == True)).sum()),  # noqa: E712
              "| best function is a constant or x:", int((df.best_is_null == True).sum()))  # noqa: E712
        eq = df[df.n_equiv > 0]
        if len(eq):
            print(eq.groupby("bench").equiv.apply(lambda v: sorted(set("; ".join(v).split("; ")))))


def manifest():
    """benchmark_manifest.json: everything that defines the benchmark results (mock data,
    prior settings and tuned discounts, library and prior-file checksums, software
    versions) and where each listed variant of the truth occurs in the library"""
    import hashlib
    import json
    import platform
    import subprocess

    import scipy

    def sha(path):
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def git(path):
        try:
            return subprocess.run(["git", "-C", path, "describe", "--always", "--dirty"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    import esr

    esr_dir = os.path.dirname(os.path.dirname(os.path.abspath(esr.__file__)))
    h = hashlib.sha256()
    for root in ("fitting", "generation"):
        d = os.path.join(esr_dir, "esr", root)
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".py"):
                h.update(fn.encode() + open(os.path.join(d, fn), "rb").read())
    from katz.prior import __version__

    out = dict(
        versions=dict(katz=__version__, katz_git=git(REPO), esr_dir=esr_dir, esr_git=git(esr_dir),
                      esr_source_sha256=h.hexdigest(), sympy=sympy.__version__,
                      numpy=np.__version__, scipy=scipy.__version__,
                      python=platform.python_version()),
        design=dict(fn_set=FN_SET, cmax=CMAX, frac_sig=FRAC_SIG, n_samples=NSAMP, all_N=ALL_N,
                    benchmarks={k: dict(function=v[0], domain=v[1], truth_variants=v[2])
                                for k, v in BENCHMARKS.items()}),
        mocks=[dict(index=i, name=name, N=nx, seed=s, sigma=mock_data(name, nx, s)[2],
                    data_sha256=sha(os.path.join(DATA, mock_name(i) + ".txt")))
               for i, (name, nx, s) in enumerate(MOCKS)],
        priors={},
        library={},
        truth_in_library={},
    )
    for label in PRIORS:
        kp = make_prior(label)
        out["priors"][label] = dict(settings=PRIORS[label], basis=PRIOR_BASIS,
                                    smoothing_kwargs=[{k: np.asarray(v).tolist() for k, v in
                                                       kw.items()} for kw in kp.side_kwargs])
    for c in range(1, CMAX + 1):
        d = os.path.join(LIB, f"compl_{c}")
        files = ["trees", "all_equations", "unique_equations", "logconst"] + list(PRIORS)
        out["library"][c] = {f: sha(os.path.join(d, f"{f}_{c}.txt")) for f in files}
        with open(os.path.join(d, f"all_equations_{c}.txt")) as f:
            eqs = f.read().splitlines()
        vals = {lab: np.atleast_1d(np.loadtxt(os.path.join(d, f"{lab}_{c}.txt"))) for lab in PRIORS}
        for name, (_, _, variants) in BENCHMARKS.items():
            for v in variants:
                idx = [j for j, e in enumerate(eqs) if e == v]
                if idx:
                    rec = out["truth_in_library"].setdefault(name, {}).setdefault(v, {})
                    rec[c] = dict(n_trees=len(idx), finite_prior={
                        lab: int(np.sum(np.isfinite(vals[lab][idx]))) for lab in PRIORS})
    with open(os.path.join(WORK, "benchmark_manifest.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for name in BENCHMARKS:
        found = out["truth_in_library"].get(name, {})
        print(name, {v: sorted(found[v]) for v in found})


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
            if len(sys.argv) > 2:  # collect I [OUT]: one mock only
                collect([int(sys.argv[2])], sys.argv[3] if len(sys.argv) > 3 else None)
            else:
                collect()
    elif cmd == "manifest":
        if rank == 0:
            manifest()
    say(f"{cmd} finished in {time.time() - t0:.1f}s")
