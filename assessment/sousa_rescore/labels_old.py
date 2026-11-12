import sys, warnings, traceback, sympy
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from katz.process_equations import SymbolCoder, split_by_punctuation
import esr.generation.generator as generator
BA = [["x", "a", "x0", "x1", "x2", "x3", "x4"], ["inv", "exp", "log", "sqrt", "sin", "cube", "square", "abs"], ["+", "*", "-", "/"]]
BB = [["x", "a", "x0", "x1", "x2", "x3", "x4"], ["inv", "exp", "log", "abs"], ["+", "*", "-", "/", "pow"]]
def labels(coder, eq, maxvar=5):
    x = sympy.symbols([f'x{i}' for i in range(maxvar)], real=True)
    a = sympy.symbols([f'a{i}' for i in range(maxvar)], real=True)
    locs = {**{f'x{i}': x[i] for i in range(maxvar)}, **{f'a{i}': a[i] for i in range(maxvar)}}
    expr, nodes, c = generator.string_to_node(eq, coder.basis_functions, locs=locs, evalf=True, allow_eval=False, check_ops=True)
    if any(op in eq for op in coder.ignore_ops):
        s = ''.join(ss for ss in split_by_punctuation(str(expr)) if ss not in coder.ignore_ops)
        expr, nodes, c = generator.string_to_node(s, coder.basis_functions, locs=locs, evalf=True, allow_eval=False, check_ops=True)
    return nodes.to_list(coder.basis_functions)
if __name__ == "__main__":
    for b, eqs in [(BA, sys.argv[2:])]:
        co = SymbolCoder(b)
        for e in eqs:
            try: print(e, labels(co, e))
            except Exception: print(e, traceback.format_exc().splitlines()[-3:])
