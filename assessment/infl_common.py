"""Encyclopaedia Inflationaris corpora (the Sousa et al. 2024 language-model training sets) for
the Katz vs Kneser-Ney assessment.

Files (data/inflation/, semicolon-delimited, AI-Feynman-like columns):
  M : inflation_models_encyclopedia_M.csv, 67 potentials of the review in standard notation
  A : inflation_setA_pi.csv, 45 of them written with ESR operator set A
      {inv, exp, log, sqrt, sin, |.|, cube, square, +, *, -, /} (cos as sin(pi/2 - .))
  B : inflation_setB.csv, 58 written with set B {inv, exp, log, +, *, -, /, pow}
The parameter columns (v*_name, '# variables') are not used: SATI has an empty '# variables' field,
so katz.process_equations.standardise_file would drop it.
Standardisation: the inflaton x -> x0; every other symbol (amplitude M and the parameters) -> a0, a1, ...
(in order of appearance); |.| is dropped (the coder ignores Abs). Corrections ('fix=True', default):
  * set B writes sqrt(z) as z**(-1/2) (should be z**(1/2)); fixed.
  * RCHI has 16*sqrt(6) in the numerator instead of the denominator (constants only); fixed.
"""
import hashlib
import os
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
import sympy

HERE = os.path.dirname(os.path.abspath(__file__))  # assessment/
ROOT = os.environ.get("KATZ_ROOT", os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
from katz.prior import KatzPrior  # noqa: E402
from katz.process_equations import split_by_punctuation  # noqa: E402
import katz.process_equations as pe  # noqa: E402
import katz.tree_parser as tp  # noqa: E402
from common import BASIS as BASIS_PHYS, CACHE, RESULTS, load_corpus  # noqa: E402,F401

DATA = os.path.join(ROOT, "data", "inflation")
FILES = {"M": "inflation_models_encyclopedia_M.csv", "A": "inflation_setA_pi.csv",
         "B": "inflation_setB.csv"}
FUNCS = {"sqrt", "exp", "log", "sin", "cos", "abs", "square", "cube", "inv", "pi", "tan"}
BASIS_A = [["a", "x"], ["inv", "exp", "log", "sqrt", "sin", "cube", "square"], ["+", "*", "-", "/"]]
BASIS_B = [["a", "x"], ["inv", "exp", "log"], ["+", "*", "-", "/", "pow"]]
BASES = {"M": BASIS_PHYS, "A": BASIS_A, "B": BASIS_B, "physics": BASIS_PHYS}

# Families of closely related potentials (held out together); identical trees are merged too
FAMILIES = [
    ["SI", "RCHI", "SAAI", "SABI"], ["TMI", "SATI"], ["MSSMI", "GMSSMI", "RIPI", "GRIPI"],
    ["KMII", "KMIII"], ["CNAI", "CNBI", "CNCI", "CNDI"], ["SAII", "SAIII"],
    ["NI", "HNI", "PSNI", "TI"], ["DWI", "GDWI"], ["LFI", "MLFI", "GMLFI", "RCLFI"],
    ["RCMI", "RCQI"], ["SFI", "BI", "VHI", "DSI"], ["ESI", "PLI", "NFI"],
    ["LI", "NCKI", "NCLI"], ["MHI", "SDI"], ["SBI", "SSBI"], ["CWI", "OI"],
]


def load_table(name, fix=True):
    df = pd.read_csv(os.path.join(DATA, FILES[name]), delimiter=";")
    out = []
    for _, r in df.iterrows():
        eq = str(r["Formula"]).replace(" ", "")
        if fix:
            eq = eq.replace("**(-1/2)", "**(1/2)")
            eq = eq.replace("A/pi**2*16*sqrt(6)", "A/(pi**2*16*sqrt(6))")
            eq = eq.replace("A/square(pi)*16*sqrt(6)", "A/(square(pi)*16*sqrt(6))")
            eq = eq.replace("A/pi**2*16*(6)**(1/2)", "A/(pi**2*16*(6)**(1/2))")
        eq = eq.replace("abs(", "(")
        toks = split_by_punctuation(eq)
        syms = []
        for t in toks:
            if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", t) and t not in FUNCS and t not in syms:
                syms.append(t)
        sub, k = {"x": "x0"}, 0
        for s in syms:
            if s != "x":
                sub[s] = f"a{k}"
                k += 1
        out.append(dict(name=r["Filename"], old=str(r["Formula"]),
                        new="".join(sub.get(t, t) for t in toks), nparam=k))
    return pd.DataFrame(out)


def _coder_prior(basis, n=1, **kw):
    """KatzPrior on a trivial corpus; used for its coder and refitted with .fit(tuples)"""
    return KatzPrior(n, basis, equations=["sin(x0) + x1"] if "sin" in basis[1] else ["x0 + x1"],
                     **kw)


def labels(name, fix=True):
    """Prefix label lists (sympy-parsed once, cached) and names of a corpus"""
    if name == "physics":
        eqs, names = load_corpus("physics"), None
    else:
        tab = load_table(name, fix=fix)
        eqs, names = list(tab.new), list(tab.name)
    basis = BASES[name]
    src = b""
    for mod in (pe, tp):  # the coder and the parser define the labels
        with open(mod.__file__, "rb") as f:
            src += f.read()
    key = hashlib.sha256("\n".join(eqs).encode() + repr(basis).encode() + src
                         + sympy.__version__.encode()).hexdigest()[:12]
    fname = os.path.join(CACHE, f"labels_{name}_{key}.pkl")
    if os.path.isfile(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
    kp = _coder_prior(basis, root_context=False)
    nsym = 10
    xs = sympy.symbols([f"x{i}" for i in range(nsym)], real=True)
    as_ = sympy.symbols([f"a{i}" for i in range(nsym)], real=True)
    locs = {**{f"x{i}": xs[i] for i in range(nsym)}, **{f"a{i}": as_[i] for i in range(nsym)}}
    labs = [kp.coder.equation2labels(eq, locs) for eq in eqs]
    out = (labs, names)
    with open(fname, "wb") as f:
        pickle.dump(out, f)
    return out


def make_prior(name, n, model, root_context=False, code_constants=False):
    """model: 'katz' (published Katz), 'katz_norm' (Katz with every conditional clipped and
    renormalised, the valid comparator), 'kn_fixed' (package default D=0.8), 'kn_mkn'
    (Chen-Goodman), 'kn_tuned' (D=0.8 placeholder; call .tune_discounts after setting eq_tuples)"""
    sm = "katz" if model.startswith("katz") else "kn"
    skw = None
    if model == "katz_norm":
        skw = {"normalise": True}
    elif model == "kn_mkn":
        skw = {"discounts": "mkn"}
    elif model in ("kn_fixed", "kn_tuned"):
        skw = {"discounts": 0.8}
    basis = BASES[name]
    return KatzPrior(n, basis, equations=["sin(x0) + x1"] if "sin" in basis[1] else ["x0 + x1"],
                     smoothing=sm, smoothing_kwargs=skw, root_context=root_context,
                     code_constants=code_constants)


def tuples(name, n, root_context=False, code_constants=False, fix=True):
    labs, _ = labels(name, fix=fix)
    kp = _coder_prior(BASES[name], root_context=root_context, code_constants=code_constants)
    return [kp.coder.labels2ntuples(n + 1, lab) for lab in labs]


def families(name, fix=True):
    """Index groups for leave-one-family-out: hand families, merged with identical trees"""
    labs, names = labels(name, fix=fix)
    N = len(names)
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    idx = {nm: i for i, nm in enumerate(names)}
    for fam in FAMILIES:
        mem = [idx[m] for m in fam if m in idx]
        for m in mem[1:]:
            union(mem[0], m)
    seen = {}
    for i, lab in enumerate(labs):
        k = " ".join(lab)
        if k in seen:
            union(seen[k], i)
        seen[k] = i
    groups = {}
    for i in range(N):
        groups.setdefault(find(i), []).append(i)
    return [np.array(sorted(g)) for g in groups.values()]


def score(prior, t):
    pl, pr = prior.event_probs(t)
    with np.errstate(divide="ignore"):
        return float(np.log(pl).sum() + np.log(pr).sum())
