import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/harry/Symbolic_regression/katz_pr1")
from katz.prior import KatzPrior
import katz; print(katz.__file__)
D = "/home/harry/Symbolic_regression/katz/data/inflation/"
BA = [["x"], ["inv", "exp", "log", "sqrt", "sin", "cube", "square", "abs"], ["+", "*", "-", "/"]]
BB = [["x"], ["inv", "exp", "log", "abs"], ["+", "*", "-", "/", "pow"]]
L = json.load(open("labels_sousa.json")); res = {}
for tag, f, b in [("A", "inflation_setA_pi.csv", BA), ("B", "inflation_setB.csv", BB)]:
    kp = KatzPrior(9, b, D + f, f"std_pr1_{tag}.csv", input_delimiter=";", root_context=False, normalise=False)
    print(tag, "corpus", len(kp.all_eq))
    for k, labs in L.items():
        if k.startswith(tag + "|"):
            res[k] = -float(kp.logprior(list(labs)))
json.dump(res, open("P_pr1.json", "w"), indent=0)
