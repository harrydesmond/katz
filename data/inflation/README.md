# Inflationary potentials (Encyclopaedia Inflationaris)

Training corpora of the language-model prior in Sousa, Bartlett, Desmond & Ferreira (2024),
"Optimal inflationary potentials", Phys. Rev. D (arXiv:2310.16786), taken from the single-field
models of the Encyclopaedia Inflationaris (Martin, Ringeval & Vennin). Semicolon-delimited, with
AI-Feynman-like columns (`Filename` = model acronym, `Formula`, `# variables`, `v*_name`); `x` is
the inflaton and `M` the amplitude.

| file | potentials | operators |
|---|---|---|
| `inflation_models_encyclopedia_M.csv` | 67 | standard notation (master list) |
| `inflation_setA_pi.csv` | 45 | ESR set A: inv, exp, log, sqrt, sin, abs, cube, square, +, *, -, / (cos as sin(pi/2 - .)) |
| `inflation_setB.csv` | 58 | ESR set B: inv, exp, log, +, *, -, /, pow |

The files are kept as received. Known errors (handled by `assessment/infl_common.py`):

- `inflation_setB.csv` writes sqrt(z) as `z**(-1/2)` throughout; it should be `z**(1/2)`. Only the
  coded trees of CNAI, CNCI and BSUSYBI depend on the sign.
- RCHI (all files) has `A/pi**2*16*sqrt(6)`; the Encyclopaedia has A/(16 pi^2 sqrt(6)). Constants
  only.
- SAAI uses `a` in the formula but names the parameter `a0` in the `v*_name` columns.
- SATI has an empty `# variables` field, so `katz.process_equations.standardise_file` skips it.

Reading the files with the package itself (`KatzPrior(..., in_eqfile=..., input_delimiter=';')`)
works for sets A and B. On the master list the coder fails on six potentials (GMLFI, LPI, CNDI,
SAIII, NCLI, RCIPI) whose amplitude `M` is not renamed: `standardise_file` renames only the first
`max(# variables)` names. `assessment/infl_common.py` instead maps `x` to `x0` and every other
symbol to `a0`, `a1`, ..., drops `abs` and keeps all rows. Its set-A trees differ from the package
reader's for SAAI, TI, CNDI and SAIII, because sympy orders the arguments of `+` and `*` by symbol
name (`a0` vs `x3`), which changes left and right children.
