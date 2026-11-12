# Sousa-style processed label lists from the 2023 coder (code dir argv[1])
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from katz.process_equations import SymbolCoder
import esr.generation.generator as generator
from labels_old import labels as raw_labels, BA, BB
from cands import CANDS
OVR = {"1/(a0 + exp(x))**2": ['inv', 'square', '+', 'exp', 'x', 'a0'], "a0*x**4": ['*', 'a0', 'square', 'square', 'x']}
out = {}
for tag, b in [("A", BA), ("B", BB)]:
    co = SymbolCoder(b)
    for row in CANDS[tag]:
        s = row[0]
        if s in OVR: out[f"{tag}|{s}"] = OVR[s]; continue
        labs = raw_labels(co, s)
        for i in range(len(labs)):
            L = labs[i]
            if L == 'Add': labs[i] = '+'
            elif L == 'Sub': labs[i] = '-'
            elif L == 'Mul': labs[i] = '*'
            elif L == 'Div': labs[i] = '/'
            elif L.lower() in co.sympy_numerics or generator.is_float(L): labs[i] = 'a'
            else: labs[i] = L.lower()
        out[f"{tag}|{s}"] = labs
json.dump(out, open("labels_sousa.json", "w"), indent=0)
for k, v in out.items(): print(k, v)
