# ESR core_maths library (complexities 1-5, generating trees, as scripts/run_esr_prior.py) scored with the
# physics corpus by a given katz code version; argv: code_dir out.json [extra kwargs json]
import sys, json, re, warnings; warnings.filterwarnings("ignore")
code, out = sys.argv[1], sys.argv[2]; kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
sys.path.insert(0, code)
from katz.prior import KatzPrior
B = [["a", "x"], ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"], ["+", "-", "*", "/", "pow"]]
L = "/home/harry/ESR/esr/function_library/core_maths"
res = {}
for n in (2, 3):
    kp = KatzPrior(n, B, "/home/harry/Symbolic_regression/katz_pr1/data/PhysicsEquations.csv", f"std_phys_{abs(hash(code))}.csv",
                   input_delimiter=";", **kw)
    for c in range(1, 6):
        try: lines = open(f"{L}/compl_{c}/trees_{c}.txt").read().splitlines()
        except FileNotFoundError: continue
        v = []
        for ln in lines:
            labs = re.findall(r"'([^']*)'", ln)
            try: v.append(-float(kp.logprior(labs)))
            except Exception: v.append(float("nan"))
        res[f"{n}_{c}"] = v
json.dump(res, open(out, "w"))
