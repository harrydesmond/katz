"""Redo the final selection of the Operon runs of operon_bloat.py with the current function
prior, without rerunning the search.

Every run's JSON file in results/operon_pop/ keeps the scored unique models of its final
population (`population`: model, theta, nll, param_codelen, logconst, neglogprior, ...). Here
the prior term of each model is recomputed with the current code (operon_bloat.prior(), the
tuned KN n = 3 prior with root context), the model with the minimum description length is
selected again, and the run's row of results/operon_pop.csv is updated: the `sel_*` columns,
`solved` (structural truth test), `solved_numeric`, `test_nmse`, `same_selection` and
`n_candidates`. The k log n selection (`mdl_*`) does not depend on the prior and is kept. The
search itself (arms with the prior as bloat penalty) used the prior of its time.

Usage (from assessment/; cluster, workers $NPROC):
    python3 operon_reselect.py        # rewrites results/operon_pop.csv (old copy *.bak)
"""

import glob
import json
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import sympy

import operon_bloat as ob

HERE = os.path.dirname(os.path.abspath(__file__))
POP_DIR = os.path.join(HERE, "results", "operon_pop")
CSV = os.path.join(HERE, "results", "operon_pop.csv")
KEY = ["arm", "bench", "N", "dseed", "gseed"]
NEGLOGP = {}  # model -> -log prior, filled before the workers are forked


def _neglogprior(model):
    try:
        logp = ob.prior().logprior(model)
    except Exception:  # noqa: BLE001  (as in operon_bloat.description_length)
        return np.inf
    return -logp if np.isfinite(logp) else np.inf


def _nmse(expr, bench, x):
    xs = np.linspace(x.min(), x.max(), 300)
    ftrue = sympy.lambdify(sympy.Symbol("x"), sympy.sympify(ob.BENCHMARKS[bench][0]),
                           modules="numpy")(xs)
    f = sympy.lambdify(ob.X, expr, modules="numpy")
    with np.errstate(all="ignore"):
        gx = np.broadcast_to(np.asarray(f(xs), dtype=float), xs.shape)
    return float(np.mean((gx - ftrue) ** 2) / np.var(ftrue))


def reselect(path):
    with open(path) as f:
        run = json.load(f)
    pop = run["population"]
    dl = np.array([p["nll"] + p["param_codelen"] + p["logconst"] + NEGLOGP[p["model"]]
                   for p in pop])
    fin = np.isfinite(dl)
    best = dict(pop[int(np.argmin(np.where(fin, dl, np.inf)))])
    best["neglogprior"] = NEGLOGP[best["model"]]
    best["DL"] = float(dl[fin].min())
    out = dict(arm=run["arm"], bench=run["bench"], N=run["N"], dseed=run["dseed"],
               gseed=run["gseed"], n_candidates=int(fin.sum()),
               changed=best["model"] != run["sel_model"],
               **{f"sel_{k}": v for k, v in best.items()})
    if out["changed"]:
        x, _, _ = ob.mock_data(run["bench"], int(run["N"]), int(run["dseed"]))
        expr = sympy.sympify(best["model"], locals={"x0": ob.X})
        out["solved"] = bool(ob.is_truth_structural(best["model"], best["theta"], run["bench"]))
        out["solved_numeric"] = bool(ob.is_truth(expr, run["bench"], x))
        out["test_nmse"] = _nmse(expr, run["bench"], x)
    out["same_selection"] = best["model"] == run["mdl_sel_model"]
    return out


def main():
    nproc = int(os.environ.get("NPROC", "2"))
    files = sorted(glob.glob(os.path.join(POP_DIR, "*.json")))
    models = set()
    for f in files:
        with open(f) as fh:
            models.update(p["model"] for p in json.load(fh)["population"])
    models = sorted(models)
    print(f"{len(files)} runs, {len(models)} unique models", flush=True)
    ob.prior()  # tune the discounts once, before forking
    with Pool(nproc) as pool:
        NEGLOGP.update(zip(models, pool.map(_neglogprior, models, chunksize=200)))
    with Pool(nproc) as pool:
        rows = pool.map(reselect, files)
    old = pd.read_csv(CSV)
    df = old.copy()
    pos = {tuple(r[k] for k in KEY): i for i, r in df[KEY].iterrows()}
    assert len(rows) == len(df) and all(tuple(r[k] for k in KEY) in pos for r in rows)
    for r in rows:
        i = pos[tuple(r[k] for k in KEY)]
        for k, v in r.items():
            if k in df.columns and k not in KEY:
                df.at[i, k] = str(v) if isinstance(v, list) else v
    shutil.copy(CSV, CSV + ".bak")
    df.to_csv(CSV, index=False)
    ch = [r for r in rows if r["changed"]]
    print(f"selection changed in {len(ch)} of {len(rows)} runs; solved before/after "
          f"{old.solved.mean():.3f}/{df.solved.astype(bool).mean():.3f}")
    for r in ch:
        print("  ", [r[k] for k in KEY], r["sel_model"], "solved", r["solved"])
    d = df.sel_neglogprior.astype(float) - old.sel_neglogprior.astype(float)
    print(f"prior term of the selected models: change {d.mean():+.4f} (max |.| "
          f"{d.abs().max():.4f})")


if __name__ == "__main__":
    sys.exit(main())
