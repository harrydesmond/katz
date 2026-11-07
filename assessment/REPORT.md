# Kneser–Ney vs Katz back-off function priors: implementation and assessment

Date: 2026-11-07. All numbers below are produced by the scripts in `assessment/` (see
"Reproducing"). NLL = negative log-probability in nats, i.e. the structural codelength that the
prior contributes to the MDL/Bayesian model selection. Lower is better.

## Summary

1. **With tuned or fixed discounts, Kneser–Ney (KN) is a better function prior than Katz back-off
   on all within-corpus held-out and synthetic tests, by a margin that depends strongly on the
   context length n.** Across sources (trained on the Feynman equations, tested on the others) the two
   are not significantly different at n ≤ 3 (item 5), and the Chen–Goodman discount estimates make KN
   worse than Katz on small synthetic corpora (item 4). With equation families held out and discounts tuned
   by nested cross-validation, KN lowers the held-out codelength of real physics equations by
   0.34 ± 0.10 nats per equation at n = 2, 0.50 ± 0.13 at n = 3, 0.76 ± 0.18 at n = 4 and about
   1.5 nats at n = 6–10. KN is better on 69–97% of individual equations. Katz is best at n = 3 and degrades with longer contexts, whereas KN stays
   flat. Comparing each model at its best n, KN (n = 4) beats Katz (n = 3) by 0.57 ± 0.19 nats per
   equation.
2. **The largest single improvement is representational, not the smoothing.** Giving the root an
   explicit start-of-tree context (`root_context=True`) lowers the Katz codelength by
   1.97 ± 0.19 nats per equation at n = 2. In the original representation, roughly 1.3 nats of KN's
   apparent gain come from its boundary convention accidentally modelling the root better. With the
   root context in place, KN's genuine smoothing gain is the 0.3–1.5 nats quoted above.
   Root context plus tuned KN improves on the README/methods-paper default (Katz, n = 2) by
   2.5 ± 0.3 nats per equation, and on Katz at n = 10 (as used in the applications) by
   3.8 ± 0.3 nats.
3. **Katz assigns zero probability to many candidate functions; KN to none.** Katz gives −∞ to
   22.5% of the 3,680 fitted functions of a real ESR run (EoS-DE, complexity ≤ 8) and to 15% of the
   ESR core_maths complexity-6 library, e.g. every function with `inv` as a second child (`x + 1/x`),
   because that position never occurs in the corpus. Such functions are silently excluded from an
   MDL ranking.
4. **The discounts must be calibrated for corpora this small.** The textbook Chen–Goodman
   discount estimates are noisy for a 15-symbol vocabulary. On synthetic corpora of ≤ 50
   equations they make KN *worse* than Katz. A single discount per model chosen by
   cross-validation (`KatzPrior.tune_discounts()`, optimum D ≈ 0.7–0.9), or simply fixed at
   D ≈ 0.75–0.85, is better than Katz at every corpus size tested (25–3000 trees) for all four
   synthetic truths.
5. **Generalisation to equations from a different source is weaker.** Trained on the 100 Feynman
   equations and tested on the 61 others, KN and Katz tie at n = 3 (−0.33 ± 0.24 to
   +0.03 ± 0.29 nats); KN gains only at n ≥ 4, with large uncertainties. In the reverse direction
   (61 training equations) KN wins clearly (+0.68 ± 0.19 at n = 3).
6. **Constant coding is a design choice, kept as an option (`code_constants=True`).** The original
   coder maps every constant and parameter to `x`. This is defensible: in the corpus, physical
   constants (G, c, ħ) are symbols like any variable, and which symbols are regressed and which are
   fitted is a separate choice. The default therefore stays as in the original Katz prior. The cost is
   that `pow(x,a)` and `pow(a,x)` (power law vs exponential) receive identical priors, as do x² and
   x^y. With `code_constants=True` (numerals and fitted parameters coded `a`) the prior tells real
   equations from their constant–variable swaps with AUC 0.99 (0.50 by construction otherwise).
7. **In real ESR rankings the switch reorders near-ties but did not change the winner** in the two
   datasets examined (same representation, Katz → KN): EoS-DE top-10 overlap 60–80%, Pantheon+
   complexity 5 identical. Per-function log-prior differences have SD 2–3 nats, comparable to DL
   gaps between top functions.

Recommended configuration: `KatzPrior(n=3 or 4, ..., smoothing='kn', root_context=True)` followed
by `tune_discounts()`, with the default constant coding. Caveats are listed below; the most important
is that the root context makes single-leaf (constant) functions more expensive.

## 1. What was implemented

- `katz/kneser_ney.py`: `KneserNey`, interpolated Kneser–Ney with the same `get_pbo` interface as
  `BackOff`. With one discount per order it is interpolated absolute-discount KN (the package
  default, D = 0.8); with count-dependent discounts D1, D2, D3+ (the Chen–Goodman estimates
  `discounts='mkn'`, or `tune_discounts('count')`) it is modified KN.
  - Counts: each tuple is one prediction event. An event contributes its raw count at its own
    (highest available) order and continuation counts N1+(• g) at all lower orders. Tuples shorter
    than the model order (lineages that reach the root) are treated like `<s>`-initial n-grams
    (KenLM convention).
  - Closed vocabulary: the recursion ends in a uniform distribution over all basis symbols, so every
    in-basis operator has P > 0 in every context.
  - Discounts: `'mkn'` (Chen–Goodman D1, D2, D3+ per order, clipped to [10⁻³, r]), `'kn'` (one
    absolute discount per order), a scalar, or per-order arrays; `discount_scale`;
    `set_discounts()` changes them without recounting.
- `KatzPrior` new options. Apart from the smoothing, all default to the original behaviour. With
  `smoothing='katz'` the original prior is reproduced bit-for-bit on 741 evaluations except for
  single-leaf functions (see below):
  - `smoothing='kn'|'katz'` and `smoothing_kwargs`. Since version 0.2.0 the default is `'kn'` with a
    fixed discount D = 0.8 at all orders (`DEFAULT_KN_DISCOUNT`); results published before then used
    `'katz'`. `FunctionPrior` is an alias of `KatzPrior`.
  - `root_context`: keeps start-of-tree padding so the root and near-root nodes have an explicit
    context.
  - `code_constants` (default False, as in the original prior): codes numerical constants and
    parameters as `a`.
  - `right_backoff='sibling'|'parent'`: which context of a right child is dropped first.
  - `equations=[...]` (build from a list).
  - `fit()`, `split_events()`, `event_probs()`, `logprior_tuples()`.
  - `tune_discounts(method='single'|'scale', k=10)`: k-fold CV over the corpus equations, separately
    for the left and right models.
- `esr_prior.compute_logprior(..., prior_kwargs=, label=)`: output files are prefixed by `label`,
  by default the smoothing method plus the non-default representation options (`kn`, `katz`,
  `kn_root`, ...). A JSON manifest beside the output records the full prior configuration, a hash
  of the corpus, the package version and the number of functions that could not be scored;
  existing output is reused only if its manifest matches. Rank 0 alone standardises the corpus
  file and broadcasts the equations.
- Robustness (from an independent code audit): a bare variable (`x0`) no longer breaks the corpus
  reader; the number of variables of a standardised corpus is the largest index + 1 (it was the
  number of occurrences); `KneserNey` rejects zero, negative or non-finite discounts and invalid
  orders, accepts an empty corpus (uniform model, e.g. a corpus without binary operators) and has a
  bounded cache; `tune_discounts` validates its folds and grid and restores the full-corpus models
  after an error; the order of the nullary basis is deterministic. None of these changes alters the
  priors of the Physics or Feynman corpora.
- Single-leaf fix: a single-leaf tree (`x`, `a0`) used to produce its root event twice, squaring
  its probability. `SymbolCoder.nodes2ntuples` now emits only the root event. This changes the
  prior of complexity-1 functions (and of longer ESR strings that simplify to a leaf) under every
  model, e.g. Katz n = 2: log P(`a0`) = −0.99 → −0.50.
- Tests: 13 new unit tests in `tests/test_katz.py`:
  - a hand-computed trigram example and a worked bigram example (P = 0.625/0.375);
  - normalisation Σ_w P(w|h) = 1 to 10⁻¹² for all contexts and discount modes;
  - continuation-count behaviour and the Chen–Goodman estimates;
  - every new `KatzPrior` option, `tune_discounts` and the single-leaf fix;
  - a KN pass in the ESR integration test.

  The two `TestKatzPrior` value tests fail on this machine with or without these changes (sympy/ESR
  version dependence, differences of 0.05 and 0.14 nats).

## 2. Tests

| Test | What it measures | Script |
|---|---|---|
| Held-out codelength | NLL of held-out corpus equations: leave-one-out, leave-one-family-out (lecture sections such as I.6.2a/I.6.2/I.6.2b held out together) with nested discount tuning, and cross-source (Feynman ↔ other 61) | `cv_corpus.py`, `cv_grouped.py`, `cv_extra.py`, `tune.py` |
| Discount calibration | CV NLL vs discount; nested tuning of single/per-order/scaled discounts | `tune.py` |
| Synthetic | Excess codelength KL(truth ‖ prior) for corpora of 25–3000 trees drawn from known tree grammars (one fitted to the Physics corpus by hierarchical-Dirichlet smoothing, so neutral between Katz and KN; three random hierarchical-Dirichlet truths with weak/medium/strong context dependence); 10 replicates, 3000 test trees | `synthetic.py` (Glamdring) |
| Discrimination | AUC for real held-out equations vs same-shape corruptions (one operator swapped; all operators relabelled; constant–variable sibling swap) | `discrimination.py` |
| Real ESR rankings | Re-rank existing ESR fits (EoS-DE complexity 1–8; Pantheon+ complexity 5) with each prior | `esr_rerank.py` |
| Runtime | Build and evaluation time | `runtime.py` |

The synthetic tuple builder reproduces `SymbolCoder` exactly on all 161 corpus equations
(n = 1–3, both root modes).

## 3. Results

### 3.1 Held-out codelength of real equations

Leave-one-family-out, nested tuning, Physics corpus (161 equations; means over the 159 that are
finite under Katz). Gain = NLL(Katz) − NLL(KN), paired.

| n | repr. | Katz | KN (Chen–Goodman) | gain | KN (tuned) | gain |
|---|---|---|---|---|---|---|
| 2 | original | 16.28 | 14.69 | +1.60 ± 0.14 | 14.55 | +1.74 ± 0.11 |
| 3 | original | 16.27 | 15.34 | +0.93 ± 0.19 | 15.07 | +1.20 ± 0.19 |
| 10 | original | 17.54 | 14.59 | +2.95 ± 0.29 | 14.39 | +3.15 ± 0.31 |
| 2 | root ctx | 14.76 | 14.59 | +0.16 ± 0.12 | 14.42 | +0.34 ± 0.10 |
| 3 | root ctx | 14.31 | 13.92 | +0.39 ± 0.12 | 13.81 | +0.50 ± 0.13 |
| 4 | root ctx | 14.50 | 13.78 | +0.72 ± 0.16 | 13.74 | +0.76 ± 0.18 |
| 6 | root ctx | 15.26 | 14.02 | +1.24 ± 0.19 | 13.83 | +1.43 ± 0.24 |
| 10 | root ctx | 15.50 | 14.10 | +1.40 ± 0.22 | 14.01 | +1.49 ± 0.27 |

![](../Plots/kn_nll_vs_n.png)

`Plots/kn_nll_vs_n.png`: top, mean held-out NLL vs n (error bars are SEs of the mean, dominated
by equation-to-equation scatter); bottom, paired gain of KN over Katz.

Leave-one-out gives the same picture, so near-duplicate families do not drive the result. Plain
k-fold nested tuning also agrees, giving +0.32 ± 0.10, +0.48 ± 0.14, +0.77 ± 0.19 and
+1.46 ± 0.25 at n = 2, 3, 4, 6 with root context.
Tuned KN is better than Katz on 69–97% of individual equations, depending on n and representation (`Plots/kn_per_equation.png`).

Cross-source tests (root context, tuned KN):

| train → test | n = 2 | n = 3 | n = 4 | n = 10 | Katz −∞ |
|---|---|---|---|---|---|
| Feynman (100) → other (61) | +0.12 ± 0.16 | −0.33 ± 0.24 | +0.19 ± 0.39 | +0.77 ± 0.49 | 1 |
| other (61) → Feynman (100) | +0.38 ± 0.15 | +0.68 ± 0.19 | +1.05 ± 0.24 | +1.36 ± 0.26 | 7 |

With untuned Chen–Goodman discounts, Feynman → other gives +0.03 ± 0.29 at n = 3 and
+0.86 ± 0.42 at n = 4. The forward direction is the most demanding test: the 61 bonus and
Wikipedia equations are larger and stylistically different. There KN's advantage is not
significant at the best n. Tuning the discounts on the Feynman equations does not help, which
suggests mild overfitting of the discount to the training source.

### 3.2 Where the gain comes from

Leave-one-out, n = 2, mean NLL per equation split by node type:

| | root | first children | second children | total |
|---|---|---|---|---|
| Katz, original | 2.37 | 6.04 | 7.73 | 16.15 |
| KN, original | 1.08 | 5.94 | 7.58 | 14.60 |
| Katz, root ctx | 1.04 | 5.84 | 7.73 | 14.62 |
| KN, root ctx | 1.01 | 5.90 | 7.58 | 14.49 |

In the original representation the root is predicted by the unigram distribution of all first
children. That distribution is dominated by leaves, while real roots are almost always `*`, `/`, `+` or
`pow`. KN's raw counts for root events fix this by accident; an explicit root context fixes it for
both models. With the root context (leave-one-family-out, tuned KN), most of the remaining gain is in
the second-child model: 7.84 → 7.60 (n = 2), 7.46 → 7.05 (n = 3), 8.20 → 7.20 (n = 6). The
first-child gain is small at n ≤ 3 (5.80 → 5.75 at n = 3; Chen–Goodman discounts make it slightly
worse) and grows with n (6.01 → 5.62 at n = 6).

`right_backoff='parent'` changes the leave-one-out NLL by −0.25 to +0.44 nats per equation with no
consistent sign; it is not recommended.

### 3.3 Calibrating the discounts

![](../Plots/kn_discount_landscape.png)

The CV optimum of a single discount used at all orders is D ≈ 0.72 (n = 2), 0.83/0.71 (n = 3,
left/right), 0.88/0.77 (n = 4), 0.93/0.88 (n = 6). It is stable across folds (±0.01–0.06), and
moving D by ±0.15 from the optimum raises the leave-one-out NLL by at most 0.065 nats per equation. The Chen–Goodman estimates hit their clipping bounds at
some orders (e.g. D3+ = 3 at order 1). Optimal rescaling factors are 0.73–1.0, i.e. Chen–Goodman is
near-optimal at large n but too aggressive at n = 2. Per-order discounts gain a further 0.03–0.08
nats over a single tuned D (nested CV).

Count-dependent discounts D1, D2 and D3+ shared by all orders and tuned on the inner folds
(`python3 tune.py nested N REP single,perorder,count2,count3 count`) gain nothing at n = 2,
0.05 ± 0.05 nats per equation at n = 3 and 0.11 ± 0.05 at n = 4 and 6 (root context), relative to
a single tuned D. Tuning only D1 and D2+ = D3+ gives the same gain, which is comparable to that of
per-order discounts. The tuned values rise with count (D1 ≈ 0.6–0.9, D3+ ≈ 1.0–1.4 at n ≥ 3), as in
large text corpora. Modified Kneser–Ney therefore fails here because the Chen–Goodman estimates
of these discounts are noisy for so small a corpus, not because count-dependent discounting is
inappropriate.

### 3.4 Synthetic corpora

![](../Plots/kn_synthetic.png)

`Plots/kn_synthetic.png`: KL(truth‖KN) − KL(truth‖Katz) in nats per tree vs training-set size,
computed on test trees finite under every model (this excludes Katz's −∞ trees, 14% at M = 25 and
1.5% at M = 160, so it favours Katz). The dotted line marks the size of the real corpus.

- The models converge to the truth as M grows (Katz n = 2 excess 2.75 → 0.69 nats/tree from M = 25
  to 1000 for the corpus-fitted truth). Misspecified n = 1 models plateau as they should.
- Tuned or fixed-D KN beats Katz for every truth and size (56 of 56 truth × size × n cells, all
  but two at ≥ 3σ). At M = 160 the gain is 0.09–0.22 nats/tree at n = 2 (12% of the Katz excess
  for the corpus-fitted truth) and 0.16–0.47 at n = 3. At M = 25 it is 0.1–1.0 nats/tree.
- Chen–Goodman KN is worse than Katz at n = 2 for M ≤ 50 (+0.1 to +0.5 nats/tree) and matches it
  from M ≈ 100, except for the strong-context truth, where it stays worse up to M = 300. At n = 3
  it wins from M ≈ 50 for three of the four truths.
- The gain is smallest when the truth has strong, sparse context dependence and largest when the
  context dependence is weak. Interpolation with lower orders helps most when they carry most of
  the information.

### 3.5 Discrimination and constant coding

![](../Plots/kn_discrimination.png)

AUC for telling a held-out real equation from same-shape corruptions (leave-one-family-out, root
context, KN with D = 0.8; paired differences):

| corruption | n | Katz | KN | KN − Katz |
|---|---|---|---|---|
| one operator swapped | 2 | 0.899 | 0.913 | +0.014 ± 0.006 |
| | 3 | 0.903 | 0.927 | +0.024 ± 0.006 |
| | 6 | 0.870 | 0.920 | +0.049 ± 0.009 |
| all operators relabelled | 3 | 0.960 | 0.977 | +0.017 ± 0.007 |

With `code_constants=True` (KN, n = 3): constant–variable swaps (x² → 2ˣ, x/2 → 2/x; 93 equations)
go from AUC 0.500 (ties by construction) to 0.989. Operator-swap AUC rises 0.927 → 0.945. Example
priors (KN, n = 3, root ctx): `pow(x0,a0)` −7.5 vs `pow(a0,x0)` −11.4; `x0/a0` −6.6 vs `a0/x0`
−10.2; under the default coding each pair is tied. Held-out NLLs are not comparable across
codings (different event spaces). Within the constant coding KN still beats Katz by margins
similar to §3.1 (`results/cv_grid_summary.csv`).

### 3.6 Zero probabilities and real ESR rankings

Fraction of functions with zero prior (−∞ log-prior):

| set | Katz | KN |
|---|---|---|
| held-out corpus equations (LOO) | 2/161 | 0 |
| Feynman equations, trained on the other 61 | 7/100 | 0 |
| ESR core_maths complexity 5 (all 594 parsed) / 6 (first 2000) | 1.9% / 14.7% | 0 / 0 |
| EoS-DE ESR fits, complexity ≤ 8 | 22.5% (829/3680) | 0 |

Mechanism: Katz backs off to a unigram over the tokens seen in that position. `inv` occurs as a first
child but never as a second child in the corpus, so P(second child = inv) = 0 in every context.

Re-ranking existing ESR fits (−log L + parameter codelength + −log prior, priors trained on the full
corpus; functions with −∞ Katz prior excluded from the comparison):

| comparison | EoS-DE top-10 overlap | same winner | Pantheon+ (cx 5) top-10 overlap |
|---|---|---|---|
| Katz → KN, n = 3, root ctx | 80% | yes | 100% |
| Katz → KN, n = 10, root ctx | 60% | yes | 100% |
| Katz n = 2 original → Katz n = 3 root ctx | 20% | no | 100% |

Ranking each prior over all functions it assigns non-zero probability (functions with zero Katz
probability ranked last under Katz) gives the same top-10 overlaps and winners: none of the KN
top-10 functions has zero Katz probability, and one of the Pantheon+ KN top 50 does
(`results/rerank_summary.csv`, which also lists the overlaps over the shared support).

The smoothing changes per-function log-priors by SD 2.0–3.1 nats (Spearman 0.84–0.92 between
Katz and KN log-priors). That is enough to reorder near-degenerate functions but did not change
either winner. The representation change matters more for ranking than the smoothing.

### 3.7 Runtime

KN evaluates a pre-parsed function in 0.012–0.015 ms against 0.03–0.35 ms for Katz (3–25× faster;
Katz recomputes its back-off weights on every call). Both are negligible next to sympy parsing
(1.6–2.1 ms per function), so generating priors for an ESR library is not noticeably faster.
Building either model takes 0.8 s, dominated by parsing the corpus.

## 4. Recommendations

1. Use `smoothing='kn'` with `root_context=True` and n = 3 or 4, and call `tune_discounts()` on the
   corpus at hand (0.4 s for the Physics corpus), or `tune_discounts('count')` (D1 and D2+, ~5 s),
   which gains a further ~0.1 nats per equation at n ≥ 4. If tuning is not wanted, `smoothing_kwargs={'discounts': 0.8}`
   is a good fixed choice for corpora of ~100–200 equations and is the package default. Avoid the
   Chen–Goodman estimates (`discounts='mkn'`) for corpora of fewer than ~100 equations.
2. Applications that used Katz with long contexts (n = 9–10, as in the halo-profile and inflation
   papers) are where KN helps most: 1.5–3 nats per equation in held-out codelength.
3. Keep the default constant coding unless the application needs the prior to separate power laws
   from exponentials. In that case use `code_constants=True`, bearing in mind that it maps fitted
   parameters onto the corpus's pure numerals (2, π, ½), which is itself an approximation.
4. Make `smoothing='kn'` (D = 0.8) the package default, with a prominent note that published
   results used `smoothing='katz'` (implemented in version 0.2.0; README.rst). Whether
   `root_context=True` should also become the default depends on the ESR benchmark, because of the
   penalty on single-leaf functions.

## 5. Caveats and open issues

- **Constant functions under the root context.** No corpus equation is a bare leaf, so a
  root-context prior makes single-leaf functions more expensive (`a0` or `x`: −3.5 nats Katz,
  −5.2 to −5.9 nats KN at n = 3, vs −0.5 nats in the original representation). This penalises the
  constant/null model in ESR
  (e.g. constant w in an EoS fit). If null models must stay cheap, either keep `root_context=False`
  or add a dedicated term for complexity-1 functions.
- **Grammar leakage.** `y` (a leaf different from its leaf sibling) is only valid after a leaf first
  child. Both models leak some probability to it in other contexts. Renormalising over valid
  continuations would gain a little for both.
- **Corpus size.** All real-corpus conclusions rest on 161 partly overlapping equations, and the
  out-of-source test shows weaker gains for the Feynman → other direction. A second, independent
  corpus (e.g. the Encyclopaedia Inflationaris potentials) would test generality.
- **Downstream SR benchmark in progress.** Whether the improved prior recovers known generating
  functions more often is being tested with ESR fits of the Nguyen-8 and Korns-1/4/6/7 mocks of the
  methods paper (`esr_benchmark.py`): the default MDL prior, Katz n = 2 (the paper's prior), KN n = 2,
  Katz and KN at n = 3 with root context, and KN at n = 3 with root context and `code_constants=True`.
  Functions with equal likelihood are treated as one, counted as the truth if any of them is a listed
  variant of the truth, and the best competing function is checked numerically for equivalence to
  the truth.

## 6. Reproducing

Run from `assessment/` (local runs ≤ 2.5 min each; parsed corpora cached in `assessment/cache/`):

```
python3 cv_corpus.py            # LOO grid (n=1-4, representations)        -> results/cv_grid*.csv
python3 tune.py landscape       # discount landscape                       -> results/discount_landscape.csv
python3 tune.py nested 3 root   # nested 10-fold tuning (any n, base|root) -> results/nested_*.csv
python3 cv_extra.py             # Feynman -> other; LOO at n = 8, 10       -> results/out_of_source.csv, loo_large_n.csv
python3 discrimination.py [const]                                          # -> results/discrimination*.csv
python3 esr_rerank.py           # re-rank EoS-DE and Pantheon+ ESR fits    -> results/rerank_*.csv
python3 runtime.py              #                                          -> results/runtime.csv
python3 make_plots.py           # Plots/kn_*.png
```

Run on a cluster (20 cores, about 1–3 min): `python3 synthetic.py run 10` and `python3 cv_grouped.py`,
with `jobs/run_multiprocessing.sh` (ESR imports `mpi4py`, so in a single-process multiprocessing job
the Slurm/PMI environment variables must be cleared before Python starts). The ESR benchmark uses
`jobs/run_bench_gen.sh`, `jobs/run_bench_fit.sh` and `jobs/run_bench_collect.sh` (paths in
`jobs/env.sh`). `esr_rerank.py` and `runtime.py` read the ESR clone from
`ESR_DIR` (default `./ESR`) and the EoS-DE fits from `EOS_FIT_DIR`. The reverse cross-source
table was produced inline with the code of `cv_extra.out_of_source` with the corpora swapped
(`results/out_of_source_reverse.csv`).
