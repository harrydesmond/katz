# Katz versions and the priors of Sousa et al. (2024)

Checks behind the version note at the top of `../REPORT.md`: which package version reproduces the
published Katz priors, and how the Katz-prior rankings of Sousa et al. (2024, arXiv:2310.16786,
Tables I-IV) change with the corrected priors.

The scripts expect the old package versions as source trees beside them, made with
`git archive <sha> | tar -x -C v_<sha>` from this repository (`v_540f87b` = the October 2023
code, `v_72f404a` = tag v0.1.0, `orig` = 31cb5fa = v0.2.1, `v_436ac3e` = the Good-Turing fix
alone) and, for `test2023.py`, ESR at c2b95e3 (`git -C ~/ESR archive c2b95e3 esr`). Run them from
this directory.

- `tables.py`: the published tables (function, complexity, residuals, function and parameter
  codelengths, total DL). `cands.py`: candidate pool.
- `labels_dump.py` (with `labels_old.py`): the trees as coded by the 2023 code ->
  `labels_sousa.json`, with two hand-set trees that reproduce the published values.
- `score_old_labels.py <code dir> 9 <out>`: priors from an old version -> `P_v_540f87b.json`,
  `P_v010.json`, `P_orig.json` (v0.2.1), `P_gt.json`. `score_pr1.py`: PR #16 with
  `root_context=False, normalise=False` -> `P_pr1.json`. `score_new.py`: this branch's priors
  (Katz with and without fixes, n = 3 and 9, root context, tuned KN) -> `P_new.json`.
- `rank.py`: description lengths and rankings -> `sousa_rescore.csv`.
- `phys_esr.py <code dir> <out>`: ESR core_maths trees (complexity 1-5) scored with the physics
  corpus at n = 2 and 3 -> `phys_v010.json`, `phys_v021.json`.
- `test2023.py`, `treetest.py`, `tryforms.py`: checks of the representation used in 2023.

Results are summarised in `log.md` (2026-11-12) and the REPORT's opening paragraph.
