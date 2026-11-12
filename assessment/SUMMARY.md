# Kneser–Ney smoothing for the Katz function prior: summary of results

Status on 10 November 2026. The full analysis, tables and caveats are in `REPORT.md`, and the plots
in `Plots/kn_*.png`.

## Method

The function prior of Bartlett, Desmond & Ferreira (2023) assigns a probability to the expression
tree of a candidate function with an n-gram model of its operators. Each node is predicted from its
n nearest ancestors, and a second child also from its left sibling. The model is trained on 161
physics equations (100 from AI Feynman and 61 from other sources) and smoothed by Katz back-off with
Good–Turing discounting. We added interpolated Kneser–Ney (KN) smoothing (`smoothing='kn'`), with
one discount per order by default or count-dependent (modified KN) discounts. KN subtracts a fixed discount from each observed count, estimates the
lower-order distributions from the number of distinct contexts in which a symbol occurs, and
interpolates down to a uniform distribution over the basis, so that all basis operators have
non-zero probability in all contexts.

Three further options are independent of the smoothing: an explicit start-of-tree context for the
root node (`root_context`), separate coding of numerical constants and fitted parameters
(`code_constants`) and tuning of the discounts by ten-fold cross-validation over the corpus
(`tune_discounts()`). Constant coding is off by default, so constants are coded like variables as
in the original prior. KN with a fixed discount D = 0.8 is the default. With `smoothing='katz'`
the package gives the Katz model of version 0.2.1, apart from a fix for single-leaf functions
(x, a₀) whose root event the original coder counted twice. Priors computed before version 0.2.0,
including those of Bartlett, Desmond & Ferreira (2023) and Sousa et al. (2024), used version 0.1.0,
which omits the parent from the context of a right child at n = 2. On the physics corpus the two
versions differ by a median of 0.5 nats in the priors of ESR's core_maths functions up to
complexity 5 (n = 2). The configurations of published work below are evaluated with version 0.2.1
and do not reproduce the published priors.

We compared the priors on held-out codelengths (negative log-priors) of corpus equations, under
leave-one-out, leave-one-family-out with nested discount tuning and cross-source splits. We also
used synthetic corpora of 25–3000 trees drawn from known tree grammars, for which the excess
codelength relative to the truth can be computed exactly. The remaining tests measure how well
each prior separates real equations from corrupted versions of the same tree shape, how many
functions in ESR libraries receive zero probability, and how the priors reorder existing ESR fits.
We repeated the held-out, cross-source and discrimination tests on an independent corpus, the
inflationary potentials with which Sousa et al. (2024) trained the prior.

## Findings

(i) KN gives shorter held-out codelengths than Katz at all context lengths tested (n = 2–10), and
the margin grows with n. Holding out whole families of related equations, with the root context
and tuned discounts, the gain is 0.35 ± 0.10, 0.46 ± 0.12 and 0.70 ± 0.18 nats per equation at
n = 2, 3 and 4, and 1.2–1.3 nats at n = 6–10. Katz is best at n = 3 and degrades with longer contexts
while KN stays flat. Comparing each model at its best n, KN wins by 0.48 ± 0.19 nats per equation,
and it is better on 67–97% of individual equations depending on n and representation. All numbers in this
summary use sympy 1.14, which the package now requires (with sympy 1.12, which parses 6 of the 161
corpus equations differently, the gains above were up to 0.2 nats larger).

(ii) The largest single improvement is representational. The original representation predicts the
root from the distribution of all first children, which is dominated by leaves. Giving the root an
explicit start-of-tree context lowers the Katz codelength by 1.51 ± 0.07 nats per equation at
n = 2. Root context and tuned KN together improve on Katz at n = 2 (the methods paper's default) by
2.4 ± 0.2 nats per equation, and on Katz at n = 10 (as used in the halo-profile and inflation
applications) by 3.6 ± 0.3 nats.

(iii) Katz assigns zero probability to a substantial fraction of the functions ESR considers,
while KN assigns it to none. This applies to 22.5% of the 3680 functions fitted in an existing ESR
analysis of the dark energy equation of state (complexity ≤ 8), and to 15% of the first 2000
functions of the ESR `core_maths` library at complexity 6. This is because Katz backs off to a
unigram distribution over the symbols seen in each position. The symbol `inv` never appears as a
second child in the corpus, so functions such as x + 1/x drop out of the MDL ranking altogether.

(iv) The discounts need calibrating for a corpus this small. The standard Chen–Goodman estimates
are noisy for a vocabulary of ~15 symbols, and on synthetic corpora of ≤ 50 equations they make KN
worse than Katz at n = 2 for three of the four synthetic truths. A single discount per model chosen by cross-validation (optimum
D ≈ 0.7–0.9) gives KN a smaller excess codelength than Katz in all 56 combinations of synthetic
truth, corpus size and n. For a corpus the size of the real one the gain is 0.15–0.18 nats per
tree at n = 2, and 0.01 for the truth with strong, sparse context dependence.

(v) Generalisation across sources is weaker. Trained on the AI Feynman equations and tested on the
other 61, tuned KN and Katz differ by at most 2σ at all n tested (−0.10 ± 0.21 nats per equation
at n = 3). In the reverse direction KN gains 0.70 ± 0.16 nats per equation at n = 3.

(vi) KN separates real equations from corrupted ones slightly better: the AUC for a single swapped
operator rises by 0.016 ± 0.005, 0.017 ± 0.006 and 0.042 ± 0.010 at n = 2, 3 and 6. With
`code_constants=True` the prior also distinguishes power laws from exponentials (xᵃ from aˣ),
raising the AUC for constant–variable swaps from 0.50, a tie by construction, to 0.99. We keep the
original coding as the default. In the corpus, physical constants are symbols like any variable,
and which symbols are regressed and which are fitted is a separate modelling choice.

(vii) In real ESR rankings the change of smoothing reorders near-degenerate functions without
changing the winner. The top-10 functions of the dark energy fits overlap by 70–80%, and the
Pantheon+ top 10 at complexity 5 is unchanged. Per-function log-priors shift with a standard
deviation of 2–3 nats, comparable to the description-length gaps between the leading functions.
Changing the representation (Katz at n = 2 to Katz at n = 3 with root context) does change the
dark energy winner.

(viii) The within-corpus results hold on an independent corpus. Sousa et al. (2024) trained the
prior on 45 or 58 single-field potentials from the Encyclopaedia Inflationaris, written in the
operators of two ESR basis sets and drawn from a list of 67. The Katz model is not always
normalised (its right-child conditionals lose up to 10% of the probability), so we compare to a
repaired Katz whose conditionals are renormalised. On all three corpora tuned KN with root context
beats it by 0.4–1.0 nats per potential at n = 2–3 and by 2.1–3.7 nats at n = 6–9. Katz is again
best at n = 3 and the root context alone saves 1.8–1.9 nats. Relative to the configuration of that
application (Katz at n = 9 without root context, evaluated with version 0.2.1), the held-out
codelength falls by 5.1–5.7 nats per
potential, of which the smoothing contributes 0.5–1.0. Trained on the physics corpus and tested on
the potentials, tuned KN with root context does not beat Katz (−0.9 ± 0.4 nats per potential at
n = 3, level at n ≥ 6), as in the Feynman → other test of (v). The data files contain four errors
that matter little for the prior. The package's file reader skips one potential and fails on six
others, which our own loader avoids (REPORT §3.10).

(ix) Tokens learned from the corpus do not shorten the codelength consistently. We merged the most
frequent operator combinations of the training trees, such as `pow(x,y)`, `*(x,_)` and `inv(x)`,
into new tokens (tree byte-pair encoding), with the number of merges chosen by nested
cross-validation and whole families held out. Relative to the prior at n = 3 with root context and
tuned discounts, the held-out codelength of the physics equations changes by −0.05 ± 0.15 nats per
equation at n = 3 and +0.11 ± 0.18 at n = 4. At n = 2 the merged tokens gain 0.65 ± 0.15 nats over
the n = 2 prior but only reach the n = 3 prior. This is because each merged token occupies a single
context slot and so lengthens the effective context. On the inflationary potentials the outcome
depends on the basis in which they are written: at n = 3 the gain is 0.96 ± 0.49 nats per potential
in ESR set A and 0.20 ± 0.45 in set B. Written in the physics basis (the master list), the
potentials lose 1.02 ± 0.29 nats, since cross-validation chose merges that lengthen their held-out
codelength.
Flattening sums and products into n-ary nodes, which removes the bracketing of the binary trees,
gains nothing on any corpus (−0.06 ± 0.15 nats per equation at n = 3 for physics). The package
therefore keeps its fixed operator basis (REPORT §3.11).

Together these results show that tuned KN with a root context is a better-calibrated prior on real
physics equations. Most of the gain over the published configuration comes from the root context
and from the longer contexts that KN can exploit, while the smoothing alone matters most for the
functions to which Katz assigns zero probability.

No corpus equation is a bare leaf, so the root context makes single-leaf functions expensive. At
n = 3 the log-prior of a₀ falls from −0.5 nats in the original representation to −3.5 (Katz) and
−5.2 to −5.9 nats (KN), which penalises constant models in ESR. The conclusions on real equations
rest on two small corpora (161 partly overlapping physics equations and 45–67 inflationary
potentials).

## Recovering known functions with ESR

The tests above concern the prior alone. To find out whether it helps ESR recover the generating
function, we repeated the benchmark of the methods paper. ESR fitted all functions up to
complexity 7 to mock data from Nguyen-8 (√x) and Korns-1, 4, 6 and 7, with N = 10–10⁴ points and
five noise realisations each (175 data sets), and every prior was applied to the same fits. A data
set counts as recovered when no other function has a lower description length; reparametrisations
of the truth, found by checking that each function reproduces the other exactly, count as the truth.

| | MDL (k log n) | Katz n = 2 (paper settings) | KN n = 2 | Katz n = 3 root | KN n = 3 root | KN n = 3 root + const |
|---|---|---|---|---|---|---|
| truth ranked first | 0.52 | 0.42 | 0.47 | 0.61 | 0.54 | 0.56 |

Going from n = 2 to n = 3 with root context raises recovery under both smoothings (32 data sets
gained and none lost for Katz, 12 and none for KN). Under Katz almost all of the gain comes from
the root context, with Katz at n = 3 without it recovering 0.43 of the data sets. Under KN it comes
from n, with KN at n = 3 without root context recovering 0.56. Normalising the
Katz conditionals changes one verdict in 175. The smoothing matters little: KN beats Katz at n = 2 and Katz beats KN at n = 3
with root context, but in both comparisons almost all the data sets that differ are Nguyen-8, which
is decided by margins of about 1 nat on the prior of √x at the root. Against the MDL baseline the
language-model priors win on Korns-4 and Korns-7 and lose on the two square-root functions; on
Korns-6 they prefer a₀ + a₁xᵃ² even at N = 10⁴. Fitting Korns-4 with the offset printed in the
paper (−2.13 rather than −2.3) changes no verdict. The earlier truth test, which merged functions of
equal likelihood, inflated every prior's recovery rate by 4–5 percentage points without changing
their order. Extending Nguyen-8 and Korns-7 to complexity 8 (30 data sets) removes a few recoveries
but changes none of these comparisons. It hurts the MDL baseline most: on Korns-7 its recovery
falls from 0.33 to 0.07, as nested exponentials enter that its simple length penalty does not
discourage.

## The prior in genetic programming

We also ran Operon on the same benchmarks (392 runs) with four bloat treatments and a common final
selection by description length with the KN prior. Using the prior on the literal tree as the only
bloat control recovers the truth most often (0.45, against 0.18 with Operon's usual length
objective), but its searches are worse: its selected models have a higher description length than
those of the length-controlled search in 75% of paired runs, and in every run it recovered that the
standard search did not, the standard search had found a better model. Adding the prior to the
length objective matches the standard search at about 35 times the search time. The prior should
therefore not replace the usual bloat control. We then repeated three of the arms and chose the
final model a second time with ESR's k log n codelength in place of the prior. This recovers the
truth in 0.54 of 298 runs, compared to 0.28 with the prior, including all runs that the prior
recovers. The prior instead picks variants of the truth with a parameter that the data do not
resolve, such as √(a₀ + a₁x) with a₀ ≈ 0, and on Korns-6 a power law with a free exponent. Setting
unresolved parameters to zero, as ESR does, raises its rate to 0.42 and leaves the Korns-6
difference. Under either criterion Korns-4 and Korns-7, the benchmarks on which the prior helps
ESR, are recovered in only two of their 120 runs and hence hardly enter the comparison.

## Recommendation

Use `smoothing='kn'` with `root_context=True` at n = 3 or 4, followed by `tune_discounts()`, and
keep the default constant coding unless the application needs to separate power laws from
exponentials. KN with D = 0.8 is now the package default, with a prominent note that published
priors used `smoothing='katz'` without root context in version 0.1.0. The root context is now the default too: it
gives the largest held-out gain, the n = 3 root-context priors recover more known functions in the
ESR benchmark than the n = 2 priors without it (under Katz because of the root context), and with it a
constant or x was the best function in 1 of 175 data sets under Katz and in none under KN. In genetic programming, keep the usual bloat
control rather than using the prior as the bloat penalty. On the benchmarks that Operon solves the
prior did not improve the final selection either.
