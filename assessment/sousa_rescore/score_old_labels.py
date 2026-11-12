import sys, json, warnings; warnings.filterwarnings("ignore")
code, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, code)
from katz.prior import KatzPrior
D = "/home/harry/Symbolic_regression/katz/data/inflation/"
BA = [["x"], ["inv", "exp", "log", "sqrt", "sin", "cube", "square", "abs"], ["+", "*", "-", "/"]]
BB = [["x"], ["inv", "exp", "log", "abs"], ["+", "*", "-", "/", "pow"]]
L = json.load(open("labels_sousa.json")); res = {}
for tag, f, b in [("A", "inflation_setA_pi.csv", BA), ("B", "inflation_setB.csv", BB)]:
    kp = KatzPrior(n, b, D + f, f"std_old_{tag}.csv", input_delimiter=";")
    for k, labs in L.items():
        if k.startswith(tag + "|"):
            res[k] = -float(kp.logprior(list(labs)))
json.dump(res, open(out, "w"), indent=0)
