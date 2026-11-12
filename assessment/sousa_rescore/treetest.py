import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from katz.prior import KatzPrior
D = "/home/harry/Symbolic_regression/katz/data/inflation/"
BA = [["x"], ["inv", "exp", "log", "sqrt", "sin", "cube", "square", "abs"], ["+", "*", "-", "/"]]
BB = [["x"], ["inv", "exp", "log", "abs"], ["+", "*", "-", "/", "pow"]]
kA = KatzPrior(9, BA, D + "inflation_setA_pi.csv", "std_t_A.csv", input_delimiter=";")
kB = KatzPrior(9, BB, D + "inflation_setB.csv", "std_t_B.csv", input_delimiter=";")
T = [("B", "exp(a0|a1|^x) 8-node tree  [pub 8.11, res -11.66 par 3.26]", ['exp','*','a0','exp','*','x','log','a1']),
     ("B", "exp(a0|a1|^x) 6-node string", ['exp','*','a0','pow','a1','x']),
     ("A", "a0 - cube(exp(x))  [res -12.77 par .69; needs DL>1.34]", ['-','a0','cube','exp','x']),
     ("A", "a0 - square(exp(x)) [res -12.74 par .69]", ['-','a0','square','exp','x']),
     ("A", "a0/exp(exp(x))    [res -12.79 par .69]", ['/','a0','exp','exp','x']),
     ("A", "a0*exp(-exp(x)) string", ['*','a0','exp','*','a','exp','x']),
     ("A", "a0/exp(exp(exp(x))) [res -12.62 par .69]", ['/','a0','exp','exp','exp','x']),
     ("A", "inv(exp(exp(exp(exp(x)))))  [res -12.65]", ['inv','exp','exp','exp','exp','x']),
     ("A", "inv(cube(exp(exp(exp(x)))))  [res -12.38]", ['inv','cube','exp','exp','exp','x']),
     ("A", "square(+(a0,exp)) [res -12.68 par 1.06]", ['square','+','a0','exp','x']),
     ("A", "square(+(exp,a0))", ['square','+','exp','x','a0']),
     ("B", "a0/exp(exp(x)) [res -12.79 par .69; needs DL>0.57]", ['/','a0','exp','exp','x']),
     ("B", "inv(exp(exp(inv(x)))) [res -8.65 par 0]", ['inv','exp','exp','inv','x']),
     ]
for tag, name, labs in T:
    kp = kA if tag == "A" else kB
    try: print(f"{tag} {-kp.logprior(labs):7.2f}  {name}")
    except Exception as e: print(tag, "ERR", name, e)
