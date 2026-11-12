from tables import *
def _dedupe(*tabs):
    seen, out = set(), []
    for src, tab in tabs:
        for r in tab:
            if r[0] not in seen:
                seen.add(r[0]); out.append((*r, src))
    return out
CANDS = {"A": _dedupe(("katz", A_KATZ), ("ref", A_KATZ_REF), ("klogn", A_KLOGN)),
         "B": _dedupe(("katz", B_KATZ), ("ref", B_KATZ_REF), ("klogn", B_KLOGN))}
# alternative ESR-style forms for strings the 2023 parser cannot code in set A
ALT = {"1/(a0 + exp(x))**2": ["inv(square(a0 + exp(x)))", "square(inv(a0 + exp(x)))", "square(1/(a0 + exp(x)))", "1/square(a0 + exp(x))"],
       "a0*x**4": ["a0*square(square(x))", "a0*square(x**2)", "a0*square(x)**2"]}
