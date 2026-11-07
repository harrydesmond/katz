"""Experiment 6: build and evaluation time of Katz vs Kneser-Ney priors.

Evaluation is timed on pre-parsed n-tuples of ESR core_maths functions (parsing with sympy
is common to both models and reported separately), with fresh models (no warm cache).
"""

import time

from common import BASIS, RESULTS, ROOT, load_corpus, make_prior, np, os, pd
from katz.prior import KatzPrior

ESR_DIR = os.environ.get("ESR_DIR", os.path.join(ROOT, "ESR"))  # an ESR clone
LIB = os.path.join(ESR_DIR, "esr", "function_library", "core_maths",
                   "compl_{c}", "all_equations_{c}.txt")


if __name__ == "__main__":
    eqs = load_corpus("physics")
    rows = []
    for n, rc in ((2, False), (3, True), (10, True)):
        base = KatzPrior(n, BASIS, equations=eqs, root_context=rc)
        for c in (5, 6):
            with open(LIB.format(c=c)) as f:
                fcns = [x.strip() for x in f if "zoo" not in x][:2000]
            t0 = time.time()
            tuples = []
            for s in fcns:
                try:
                    tuples.append(base.coder.process_all_equations(n + 1, [s], base.maxvar))
                except Exception:  # noqa: BLE001
                    pass
            t_parse = (time.time() - t0) / len(fcns)
            for sm in ("katz", "kn"):
                t0 = time.time()
                p = make_prior(n, sm, equations=eqs, root_context=rc)
                t_build = time.time() - t0
                t0 = time.time()
                lp = [p.logprior_tuples(t) for t in tuples]
                t_eval = (time.time() - t0) / len(tuples)
                rows.append(dict(n=n, root=rc, comp=c, model=sm, nfcn=len(tuples),
                                 build_s=t_build, parse_ms=1e3 * t_parse,
                                 eval_ms=1e3 * t_eval, finite=np.isfinite(lp).mean()))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "runtime.csv"), index=False)
    print(df.round(4).to_string(index=False))
