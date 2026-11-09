"""Experiment 8: the function prior in genetic programming (Operon) -- is it useful as a
bloat penalty on the literal tree, or should bloat be controlled in the usual way and the
prior applied only to the simplified function?

Data: the Nguyen/Korns mocks of the ESR benchmark (esr_benchmark.mock_data).
Search: Operon (pyoperon >= 0.6) with four ways of handling bloat during the search:
    A  standard: NSGA-II on (MSE, length), max length 50
    B  prior as the only bloat control: single objective -log L - log P(literal tree),
       max length 200 (effectively unconstrained)
    C  both: NSGA-II on (-log L - log P(canonical function), length), max length 50
    D  none: single objective MSE, max length 200
-log L is the Gaussian negative log-likelihood with the known noise. The literal tree keeps
every node of the Operon tree (a weighted variable w*x counts as '*', 'a', 'x'); the
canonical function is the sympy-evaluated string, as in KatzPrior.logprior(str), which
removes trivial dead code (x*1, x + 0*y, x - x).
Final selection is identical for all arms: among the unique models of the final population
(and Pareto front), the minimum of DL = -log L + parameter codelength (ESR form, from the
Fisher information) + sum log|integer constants| - log P(sympy.simplify(model)).

Metrics per run: whether the selected model is the truth up to an affine transformation
(fitted curve equal, to 1e-4 of its range, to alpha * f_true + beta), its test NMSE against the
noiseless truth, the lengths of the literal and simplified trees (and their ratio, the
dead-code fraction) for the selected model and the final population, runtime and evaluations.

Usage (needs pyoperon; ESR not needed):
    python3 operon_bloat.py run ARM BENCH N DSEED GSEED      # one run -> results/operon/*.json
    python3 operon_bloat.py batch TASKFILE I                 # run line I of a task list
    python3 operon_bloat.py tasks pilot|full > tasks.txt
    python3 operon_bloat.py collect                          # -> results/operon_bloat.csv
"""

import json
import os
import signal
import sys
import time
import warnings

import numpy as np
import scipy.optimize
import sympy

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

from esr_benchmark import BENCHMARKS, PRIOR_BASIS, mock_data  # noqa: E402

OUTDIR = os.environ.get("OPERON_OUT", os.path.join(HERE, "results", "operon"))
ARMS = ("A", "B", "C", "D")
SYMBOLS = "add,sub,mul,div,sin,cos,sqrt,square,pow,constant,variable"
SETTINGS = dict(population_size=int(os.environ.get("OPERON_POP", 500)),
                max_evaluations=int(os.environ.get("OPERON_EVALS", 100_000)), generations=1000,
                optimizer_iterations=10, tournament_size=5, max_depth=10)
MAXLEN = {"A": 50, "B": 200, "C": 50, "D": 200}
MAXDEPTH = {"A": 10, "B": 30, "C": 10, "D": 30}
# truth up to an affine transformation: alpha * g(x; b) + beta  (b fitted, start values)
TRUTH_FAMILY = {
    "nguyen_8": ("sqrt(x)", []),
    "korns_1": ("x", []),
    "korns_4": ("sin(x)", []),
    "korns_6": ("sqrt(x)", []),
    "korns_7": ("exp(b0*x)", [-0.547]),
}
X = sympy.Symbol("x0", real=True)
OPNAME = {"+": "+", "-": "-", "*": "*", "/": "/", "sin": "sin", "cos": "cos",
          "sqrt": "sqrt", "square": "square", "pow": "pow", "exp": "exp", "log": "log"}


# ---------------------------------------------------------------- prior
_PRIOR = None


def prior():
    """KN n=3 with root context and tuned discounts (the 'kn3r' prior of esr_benchmark)"""
    global _PRIOR
    if _PRIOR is None:
        import pandas as pd

        from katz.prior import KatzPrior

        eqs = list(pd.read_csv(os.path.join(REPO, "data", "NewPhysics.csv"))["New Formula"])
        _PRIOR = KatzPrior(3, PRIOR_BASIS, equations=eqs, smoothing="kn", root_context=True)
        _PRIOR.tune_discounts("single", k=10)
    return _PRIOR


def literal_labels(tree):
    """Prefix labels of an Operon tree, node by node (no simplification). Children of the
    postfix node i are at i-1, i-1-(len(first child)+1), ...; n-ary +/* are binarised."""
    nodes = tree.Nodes

    def rec(i):
        nd = nodes[i]
        if nd.IsVariable:
            return ["x0"] if nd.Value == 1.0 else ["*", repr(float(nd.Value)), "x0"]
        if nd.IsConstant:
            return [repr(float(nd.Value))]
        kids, j = [], i - 1
        for _ in range(nd.Arity):
            kids.append(rec(j))
            j -= nodes[j].Length + 1
        name = OPNAME[nd.Name]
        if len(kids) == 1:
            return [name] + kids[0]
        out = kids[-1]
        for k in reversed(kids[:-1]):
            out = [name] + k + out
        return out

    return rec(len(nodes) - 1)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout


def simplified(expr, seconds=3):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        return sympy.simplify(expr)
    except (Timeout, MemoryError):  # the alarm cannot interrupt long C-level sympy calls
        return expr
    finally:
        signal.alarm(0)


def parametrise(expr):
    """Replace every Float by a parameter a_i; returns (expression, values)"""
    floats = sorted(expr.atoms(sympy.Float), key=lambda f: float(f))
    params = sympy.symbols([f"a{i}" for i in range(len(floats))], real=True)
    return expr.xreplace(dict(zip(floats, params))), params, np.array([float(f) for f in floats])


def n_labels(expr_str):
    """Number of nodes of the canonical tree of a function string"""
    kp = prior()
    locs = {"x0": X, **{f"a{i}": sympy.Symbol(f"a{i}", real=True) for i in range(40)}}
    return len(kp.coder.equation2labels(expr_str, locs))


# ---------------------------------------------------------------- model selection
def log_integer_constants(e):
    """sum log|c| over integer constants of the tree (ESR's get_logconst), not counting the
    exponents 2 and -1, which are the operators square and 1/ (or /)"""
    tot = 0.0
    for node in sympy.preorder_traversal(e):
        if isinstance(node, sympy.Pow) and node.exp in (2, -1):
            continue  # the exponent is visited separately below; skip it there
        for a in getattr(node, "args", ()):
            if isinstance(a, sympy.Integer) and abs(int(a)) > 1 and not (
                    isinstance(node, sympy.Pow) and a is node.exp and a in (2, -1)):
                tot += np.log(abs(int(a)))
    return tot


def description_length(expr, x, y, sig):
    """DL (nats) of a sympy model with numerical constants, ESR-style; returns dict"""
    e, params, theta = parametrise(simplified(expr))
    f = sympy.lambdify([X] + list(params), e, modules="numpy")
    with np.errstate(all="ignore"):
        pred = np.broadcast_to(np.asarray(f(x, *theta), dtype=float), x.shape)
    if not np.all(np.isfinite(pred)):
        return None
    nll = 0.5 * np.sum((y - pred) ** 2) / sig**2 + len(x) * np.log(np.sqrt(2 * np.pi) * sig)
    pcl = 0.0
    for i, p in enumerate(params):
        d = sympy.lambdify([X] + list(params), sympy.diff(e, p), modules="numpy")
        with np.errstate(all="ignore"):
            g = np.broadcast_to(np.asarray(d(x, *theta), dtype=float), x.shape)
        fisher = np.sum(g**2) / sig**2
        term = -0.5 * np.log(3) + 0.5 * np.log(fisher) + np.log(abs(theta[i]))
        pcl += max(term, 0.0) if np.isfinite(term) else 0.0
    lconst = log_integer_constants(e)
    s = str(e)
    try:
        logp = prior().logprior(s)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(logp):
        return None
    return dict(model=s, theta=theta.tolist(), nll=nll, param_codelen=pcl, logconst=lconst,
                neglogprior=-logp, DL=nll + pcl + lconst - logp, n_simplified=n_labels(s))


def is_truth(expr, bench, x):
    """Fitted model equal (to 1e-4 of its range over the data domain) to alpha*g + beta"""
    gstr, b0 = TRUTH_FAMILY[bench]
    xs = np.linspace(x.min(), x.max(), 300)
    f = sympy.lambdify(X, expr, modules="numpy")
    with np.errstate(all="ignore"):
        g_model = np.broadcast_to(np.asarray(f(xs), dtype=float), xs.shape)
    if not np.all(np.isfinite(g_model)):
        return False
    b = sympy.symbols([f"b{i}" for i in range(len(b0))])
    basis = sympy.lambdify([sympy.Symbol("x")] + list(b), sympy.sympify(gstr), modules="numpy")

    def resid(p):
        with np.errstate(all="ignore"):
            return p[0] * basis(xs, *p[2:]) + p[1] - g_model

    best = np.inf
    for start in ([1.0, 0.0] + b0, [-1.0, 0.0] + b0, [100.0, 100.0] + b0):
        try:
            r = scipy.optimize.least_squares(resid, start, xtol=1e-14, ftol=1e-14)
            best = min(best, np.max(np.abs(r.fun)))
        except Exception:  # noqa: BLE001
            pass
    return bool(best <= 1e-4 * max(np.ptp(g_model), 1e-12))


# ---------------------------------------------------------------- one Operon run
def run(arm, bench, nx, dseed, gseed):
    import pyoperon as op
    from pyoperon.sklearn import _KNOWN_SYMBOLS, SymbolicRegressor

    x, y, sig = mock_data(bench, nx, dseed)
    N = len(x)
    ds = op.Dataset(np.asfortranarray(np.column_stack([x, y])))
    xvar, target = sorted(ds.Variables, key=lambda v: v.Index)
    problem = op.Problem(ds)
    problem.TrainingRange = op.Range(0, N)
    problem.TestRange = op.Range(N - 1, N)
    problem.Target = target
    problem.InputHashes = [xvar.Hash]
    config = 0
    for s in SYMBOLS.split(","):
        config |= int(_KNOWN_SYMBOLS[s])
    problem.ConfigurePrimitiveSet(config)
    pset = problem.PrimitiveSet
    names = {xvar.Hash: "x0"}

    helper = SymbolicRegressor(max_length=MAXLEN[arm], max_depth=MAXDEPTH[arm])
    creator = helper._init_creator("btc", pset, [xvar.Hash])
    coeff_init = op.NormalCoefficientInitializer()
    coeff_init.ParameterizeDistribution(0, 1)
    dtable = op.DispatchTable()
    optimizer = op.LMOptimizer(dtable, problem, SETTINGS["optimizer_iterations"], 0)

    nll0 = N * np.log(np.sqrt(2 * np.pi) * sig)
    cache = {}
    stats = dict(prior_calls=0, cache_hits=0)

    def nll_of(tree):
        with np.errstate(all="ignore"):
            pred = np.asarray(op.Evaluate(dtable, tree, ds, problem.TrainingRange))
        if not np.all(np.isfinite(pred)):
            return None
        return 0.5 * np.sum((y - pred) ** 2) / sig**2 + nll0

    def logp_literal(tree):
        labels = literal_labels(tree)
        key = tuple("a" if ll[0].isdigit() or ll[0] == "-" and len(ll) > 1 else ll
                    for ll in labels)
        if key in cache:
            stats["cache_hits"] += 1
            return cache[key]
        stats["prior_calls"] += 1
        try:
            v = prior().logprior(labels)
        except Exception:  # noqa: BLE001
            v = -np.inf
        cache[key] = v
        return v

    def logp_canonical(tree):
        s = op.InfixFormatter.Format(tree, names, 4)
        if s in cache:
            stats["cache_hits"] += 1
            return cache[s]
        stats["prior_calls"] += 1
        try:
            v = prior().logprior(s)
        except Exception:  # noqa: BLE001
            v = -np.inf
        cache[s] = v
        return v

    def fit_B(rng, ind):
        nll = nll_of(ind.Genotype)
        lp = logp_literal(ind.Genotype) if nll is not None else -np.inf
        return [nll - lp if (nll is not None and np.isfinite(lp)) else 1e30]

    def fit_C(rng, ind):
        nll = nll_of(ind.Genotype)
        lp = logp_canonical(ind.Genotype) if nll is not None else -np.inf
        return [nll - lp if (nll is not None and np.isfinite(lp)) else 1e30]

    keep = []
    if arm == "A":
        evaluator = op.MultiEvaluator(problem)
        keep += [op.Evaluator(problem, dtable, op.MSE(), False), op.LengthEvaluator(problem)]
    elif arm == "B":
        evaluator = op.UserDefinedEvaluator(problem, fit_B)
    elif arm == "C":
        evaluator = op.MultiEvaluator(problem)
        keep += [op.UserDefinedEvaluator(problem, fit_C), op.LengthEvaluator(problem)]
    else:
        evaluator = op.Evaluator(problem, dtable, op.MSE(), False)
    for e in keep:
        e.Budget = SETTINGS["max_evaluations"]
        evaluator.Add(e)
    evaluator.Budget = SETTINGS["max_evaluations"]
    single = arm in ("B", "D")

    comp = op.SingleObjectiveComparison(0) if single else op.CrowdedComparison()
    fsel = op.TournamentSelector(comp)
    msel = op.TournamentSelector(comp)
    fsel.TournamentSize = msel.TournamentSize = SETTINGS["tournament_size"]
    reinserter = op.KeepBestReinserter(comp)
    cx = op.SubtreeCrossover(0.9, MAXDEPTH[arm], MAXLEN[arm])
    mut = op.MultiMutation()
    muts = []
    for k in ("onepoint", "discretepoint", "changevar", "changefunc", "insertsubtree",
              "replacesubtree", "removesubtree"):
        m = helper._init_mutation(k, [xvar.Hash], pset, creator, coeff_init)
        mut.Add(m, 1.0)
        muts.append(m)
    coeff_opt = op.CoefficientOptimizer(optimizer)
    generator = op.BasicOffspringGenerator(evaluator, cx, mut, fsel, msel, coeff_opt)
    min_arity, _ = pset.FunctionArityLimits()
    tree_init = op.UniformLengthTreeInitializer(creator)
    tree_init.ParameterizeDistribution(min_arity + 1, 10)
    tree_init.MinDepth = 1
    tree_init.MaxDepth = 1000
    seed = 1000 * gseed + dseed
    cfg = op.GeneticAlgorithmConfig(
        generations=SETTINGS["generations"], max_evaluations=SETTINGS["max_evaluations"],
        local_iterations=SETTINGS["optimizer_iterations"],
        population_size=SETTINGS["population_size"], pool_size=SETTINGS["population_size"],
        p_crossover=1.0, p_mutation=0.25, p_local=1.0, p_lamarck=1.0, epsilon=1e-5,
        seed=seed, max_time=3600)
    if single:
        gp = op.GeneticProgrammingAlgorithm(cfg, problem, tree_init, coeff_init, generator,
                                            reinserter)
    else:
        sorter = op.RankSorter()  # must outlive gp (held by pointer)
        gp = op.NSGA2Algorithm(cfg, problem, tree_init, coeff_init, generator, reinserter,
                               sorter)
    t0 = time.time()
    gp.Run(op.RandomGenerator(np.uint64(seed)), None, 1, False)
    t_search = time.time() - t0

    # ---- final selection, identical for all arms
    t0 = time.time()
    cands = {}
    pop = list(gp.Individuals) + ([gp.BestModel] if single else list(gp.BestFront))
    lit_len, simp_len = [], []
    for ind in pop:
        s = op.InfixFormatter.Format(ind.Genotype, names, 12)
        if s in cands:
            continue
        cands[s] = (ind.Genotype, len(literal_labels(ind.Genotype)))
    rows = []
    for s, (tree, nlit) in cands.items():
        try:
            expr = sympy.sympify(s, locals={"x0": X})
            r = description_length(expr, x, y, sig)
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            continue
        r["n_literal"] = nlit
        lit_len.append(nlit)
        simp_len.append(r["n_simplified"])
        rows.append((r, expr))
    best, best_expr = min(rows, key=lambda t: t[0]["DL"])
    t_select = time.time() - t0

    xs = np.linspace(x.min(), x.max(), 300)
    ftrue = sympy.lambdify(sympy.Symbol("x"), sympy.sympify(BENCHMARKS[bench][0]),
                           modules="numpy")(xs)
    fsel_ = sympy.lambdify(X, best_expr, modules="numpy")
    with np.errstate(all="ignore"):
        gx = np.broadcast_to(np.asarray(fsel_(xs), dtype=float), xs.shape)
    out = dict(arm=arm, bench=bench, N=nx, dseed=dseed, gseed=gseed,
               solved=is_truth(best_expr, bench, x),
               test_nmse=float(np.mean((gx - ftrue) ** 2) / np.var(ftrue)),
               t_search=t_search, t_select=t_select, generations=gp.Generation,
               evaluations=int((keep[0] if keep else evaluator).CallCount),
               n_candidates=len(rows),
               pop_literal_mean=float(np.mean(lit_len)),
               pop_simplified_mean=float(np.mean(simp_len)),
               pop_deadcode=float(np.mean(1 - np.array(simp_len) / np.array(lit_len))),
               **stats, **{f"sel_{k}": v for k, v in best.items()})
    out["sel_deadcode"] = 1 - best["n_simplified"] / best["n_literal"]
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, f"{arm}_{bench}_{nx}_{dseed}_{gseed}.json"), "w") as f:
        json.dump(out, f, default=float)
    return out


def tasks(which):
    if which == "pilot":
        grid = [(b, 100, d, 0) for b in ("nguyen_8", "korns_1") for d in range(2)]
    else:
        grid = [(b, n, d, g) for b in BENCHMARKS for n in (100, 1000) for d in range(5)
                for g in range(2)]
    return [f"{a} {b} {n} {d} {g}" for (b, n, d, g) in grid for a in ARMS]


def collect():
    import glob

    import pandas as pd

    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUTDIR, "*.json")))]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "results", "operon_bloat.csv"), index=False)
    pd.set_option("display.width", 200)
    cols = ["solved", "test_nmse", "sel_n_literal", "sel_n_simplified", "sel_deadcode",
            "pop_literal_mean", "pop_deadcode", "sel_DL", "t_search"]
    print(df.groupby("arm")[cols].mean().round(3))
    print(df.groupby(["bench", "N", "arm"]).solved.mean().unstack("arm").round(2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        a, b, n, d, g = sys.argv[2:7]
        print(json.dumps(run(a, b, int(n), int(d), int(g)), default=float))
    elif cmd == "batch":
        import resource

        # cap the address space so that a runaway sympy.simplify of a large, unconstrained
        # tree (arm D) raises MemoryError instead of exhausting the node's memory
        cap = int(os.environ.get("OPERON_MEM_MB", 4000)) * 2**20
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        line = open(sys.argv[2]).read().splitlines()[int(sys.argv[3])].split()
        fname = os.path.join(OUTDIR, "{}_{}_{}_{}_{}.json".format(*line))
        if not os.path.isfile(fname):  # checkpoint: skip finished runs
            run(line[0], line[1], int(line[2]), int(line[3]), int(line[4]))
    elif cmd == "tasks":
        print("\n".join(tasks(sys.argv[2])))
    elif cmd == "collect":
        collect()
