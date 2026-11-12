import sys, warnings, traceback; warnings.filterwarnings("ignore")
kcode, ecode = sys.argv[1], sys.argv[2]
sys.path.insert(0, ecode); sys.path.insert(0, kcode)
import esr; print("esr from", esr.__file__)
from katz.prior import KatzPrior
D = "/home/harry/Symbolic_regression/katz/data/inflation/"
BA = [["x"], ["inv", "exp", "log", "sqrt", "sin", "cube", "square", "abs"], ["+", "*", "-", "/"]]
BB = [["x"], ["inv", "exp", "log", "abs"], ["+", "*", "-", "/", "pow"]]
kA = KatzPrior(9, BA, D + "inflation_setA_pi.csv", "std_t2_A.csv", input_delimiter=";")
kB = KatzPrior(9, BB, D + "inflation_setB.csv", "std_t2_B.csv", input_delimiter=";")
from cands import CANDS
for tag, kp in (("A", kA), ("B", kB)):
    for r in CANDS[tag]:
        s = r[0]
        try: v = f"{-kp.logprior(s):7.2f}"
        except Exception as e: v = "  FAIL " + type(e).__name__
        print(tag, r[-1], v, f"pub {r[3]:.2f}", s)
