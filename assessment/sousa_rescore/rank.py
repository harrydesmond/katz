import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from cands import CANDS
P0 = json.load(open("P_v_540f87b.json")); P1 = json.load(open("P_orig.json")); new = json.load(open("P_new.json"))["res"]
STARO = "a0*(1 - exp(-sqrt(2/3)*x))**2"; SK = {"A": "A|STARO9", "B": "B|STARO10"}
rows = []
for tag in ("A", "B"):
    for s, c, res, fn, par, tot, src in CANDS[tag]:
        k = SK[tag] if s == STARO else f"{tag}|{s}"
        rows.append(dict(set=tag, f=("Starobinsky" if s == STARO else s), src=src, c=c, res=res, par=par, fn_pub=fn,
                         tot_pub=tot, P0=P0[k], P1=P1[k], **{kk.split("_")[0]: v for kk, v in new[k].items()}))
d = pd.DataFrame(rows)
d["offset"] = np.where(d.src == "klogn", 0.0, d.fn_pub - d.P0)
P = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
for p in P:
    d["DL" + p] = d.res + d.par + d[p] + d.offset
d.to_csv("sousa_rescore.csv", index=False)
# consistency: P0 DL of Katz rows equals published total
print("max |DL_P0 - published total| (Katz rows):", (d.DLP0 - d.tot_pub)[d.src != "klogn"].abs().max().round(3))
lab = {"P0": "Sousa (2023 code, Katz n=9)", "P1": "upstream main (right-backoff fix)", "P4": "+ branch fixes, data fixes, normalised (n=9)",
       "P5": "repaired Katz n=3", "P6": "repaired Katz n=3 + root ctx (PR1 config)", "P7": "tuned KN n=3 + root ctx (PR2)", "P8": "tuned KN n=9 + root ctx"}
for tag in ("A", "B"):
    x = d[d.set == tag]
    print(f"\n===== set {tag}: top 10 of the {len(x)}-function pool (DL, nats) =====")
    for p in ["P0", "P1", "P4", "P6", "P7", "P8"]:
        o = x.sort_values("DL" + p)
        print(f"-- {p} {lab[p]}")
        print("   " + " | ".join(f"{r.f} {getattr(r, 'DL'+p):.2f}" for r in o.head(10).itertuples()))
    print("-- rank / DL of Sousa's MDL function and of Starobinsky:")
    top = x.sort_values("DLP0").f.iloc[0]
    for f in [top, "Starobinsky"]:
        print("  ", f, {p: (int((x["DL"+p] < x.loc[x.f == f, "DL"+p].values[0]).sum()) + 1, round(float(x.loc[x.f == f, "DL"+p].values[0]), 2)) for p in P})
