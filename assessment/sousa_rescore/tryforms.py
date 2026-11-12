import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from katz.prior import KatzPrior
D = "/home/harry/Symbolic_regression/katz/data/inflation/"
BA = [["x"], ["inv", "exp", "log", "sqrt", "sin", "cube", "square", "abs"], ["+", "*", "-", "/"]]
kp = KatzPrior(9, BA, D + "inflation_setA_pi.csv", "std_try_A.csv", input_delimiter=";")
forms = [['inv','square','+','a0','exp','x'], ['inv','square','+','exp','x','a0'],
         ['/','a','square','+','a0','exp','x'], ['/','a','square','+','exp','x','a0'],
         ['square','inv','+','exp','x','a0'], ['square','/','a','+','a0','exp','x'],
         ['inv','*','+','a0','exp','x','+','a0','exp','x'], ['/', 'a', '*','+','a0','exp','x','+','a0','exp','x'],
         ['square','/','a','+','exp','x','a0']]
for f in forms:
    try: print(f"{-kp.logprior(f):7.2f}", f)
    except Exception as e: print("ERR", f, e)
for s in ["(a0 + exp(x))**(-2)", "exp(-2*log(a0 + exp(x)))", "1/(a0**2 + 2*a0*exp(x) + exp(2*x))"]:
    try: print(f"{-kp.logprior(s):7.2f}", s)
    except Exception as e: print("ERR", s, e)
