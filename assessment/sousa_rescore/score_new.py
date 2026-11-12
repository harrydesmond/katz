# Score Sousa-style candidate trees with the current (kneser-ney branch) priors
import sys, json, warnings, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/harry/Symbolic_regression/katz/assessment")
import infl_common as ic
from katz.prior import KatzPrior
L = json.load(open("labels_sousa.json"))
GRID = np.linspace(0.05, 1.0, 20)
D = "/home/harry/Symbolic_regression/katz/data/inflation/"
FILES = {"A": "inflation_setA_pi.csv", "B": "inflation_setB.csv"}
res, info = {}, {}
def score(p, labs):
    t = p.coder.labels2ntuples(p.n + 1, list(labs))
    return -float(p.logprior_tuples(t))
for tag in ("A", "B"):
    basis = ic.BASES[tag]
    priors = {}
    # P2: branch, published flags, Sousa's corpus as read by the package
    priors["P2_branch_pub_n9_sousacorpus"] = KatzPrior(9, basis, in_eqfile=D + FILES[tag], out_eqfile=f"std_new_{tag}.csv",
                                                     input_delimiter=";", smoothing="katz", root_context=False)
    info[f"{tag}_P2_ncorpus"] = len(priors["P2_branch_pub_n9_sousacorpus"].eq_tuples)
    def fit(n, model, root):
        T = ic.tuples(tag, n, root_context=root, fix=True)
        p = ic.make_prior(tag, n, model, root_context=root)
        p.eq_tuples = T
        if model == "kn_tuned":
            p.tune_discounts("single", k=5 if len(T) < 100 else 10, seed=0, grid=GRID)
        else:
            p.fit(T)
        info[f"{tag}_{model}_n{n}_root{root}_ncorpus"] = len(T)
        return p
    priors["P3_katz_pub_n9_fixedcorpus"] = fit(9, "katz", False)
    priors["P4_katznorm_n9"] = fit(9, "katz_norm", False)
    priors["P5_katznorm_n3"] = fit(3, "katz_norm", False)
    priors["P6_katznorm_n3_root"] = fit(3, "katz_norm", True)
    priors["P7_kn_tuned_n3_root"] = fit(3, "kn_tuned", True)
    priors["P8_kn_tuned_n9_root"] = fit(9, "kn_tuned", True)
    for name, p in priors.items():
        if hasattr(p, "backoff_left") and hasattr(p.backoff_left, "discounts"):
            info[f"{tag}_{name}_D"] = str(getattr(p.backoff_left, "discounts", None))
        for k, labs in L.items():
            if k.startswith(tag + "|"):
                res.setdefault(k, {})[name] = score(p, labs)
json.dump({"res": res, "info": info}, open("P_new.json", "w"), indent=0)
print(json.dumps(info, indent=0))
