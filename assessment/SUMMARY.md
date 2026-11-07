# Kneser–Ney smoothing for the Katz function prior: summary of results

Status on 7 November 2026. The full analysis, tables and caveats are in `REPORT.md`, and the plots
in `Plots/kn_*.png`.

## Method

The function prior of Bartlett, Desmond & Ferreira (2023) assigns a probability to the expression
tree of a candidate function with an n-gram model of its operators. Each node is predicted from its
n nearest ancestors, and a second child also from its left sibling. The model is trained on 161
physics equations (100 from AI Feynman and 61 from other sources) and smoothed by Katz back-off with
Good–Turing discounting. We added interpolated modified Kneser–Ney (KN) smoothing as an option
(`smoothing='kn'`). KN subtracts a fixed discount from each observed count, estimates the
lower-order distributions from the number of distinct contexts in which a symbol occurs, and
interpolates down to a uniform distribution over the basis, so that all basis operators have
non-zero probability in all contexts.

Three further options are independent of the smoothing: an explicit start-of-tree context for the
root node (`root_context`), separate coding of numerical constants and fitted parameters
(`code_constants`) and tuning of the discounts by ten-fold cross-validation over the corpus
(`tune_discounts()`). Constant coding is off by default, so constants are coded like variables as
in the original prior. KN with a fixed discount D = 0.8 is the default from version 0.2.0, while
`smoothing='katz'` reproduces the published prior, apart from a fix for single-leaf functions
(x, a₀), whose root event the original coder counted twice.

We compared the priors on held-out codelengths (negative log-priors) of corpus equations, under
leave-one-out, leave-one-family-out with nested discount tuning and cross-source splits. We also
used synthetic corpora of 25–3000 trees drawn from known tree grammars, for which the excess
codelength relative to the truth can be computed exactly. The remaining tests measure how well
each prior separates real equations from corrupted versions of the same tree shape, how many
functions in ESR libraries receive zero probability, and how the priors reorder existing ESR fits.

## Findings

(i) KN gives shorter held-out codelengths than Katz at all context lengths tested (n = 2–10), and
the margin grows with n. Holding out whole families of related equations, with the root context
and tuned discounts, the gain is 0.34 ± 0.10, 0.50 ± 0.13 and 0.76 ± 0.18 nats per equation at
n = 2, 3 and 4, and ~1.5 nats at n = 6–10. Katz is best at n = 3 and degrades with longer contexts
while KN stays flat. Comparing each model at its best n, KN wins by 0.57 ± 0.19 nats per equation,
and it is better on 69–97% of individual equations depending on n and representation.

(ii) The largest single improvement is representational. The original representation predicts the
root from the distribution of all first children, which is dominated by leaves. Giving the root an
explicit start-of-tree context lowers the Katz codelength by 1.97 ± 0.19 nats per equation at
n = 2. Root context and tuned KN together improve on Katz at n = 2 (the methods paper's default) by
2.5 ± 0.3 nats per equation, and on Katz at n = 10 (as used in the halo-profile and inflation
applications) by 3.8 ± 0.3 nats.

(iii) Katz assigns zero probability to a substantial fraction of the functions ESR considers,
while KN assigns it to none. This applies to 22.5% of the 3680 functions fitted in an existing ESR
analysis of the dark energy equation of state (complexity ≤ 8), and to 15% of the first 2000
functions of the ESR `core_maths` library at complexity 6. This is because Katz backs off to a
unigram distribution over the symbols seen in each position. The symbol `inv` never appears as a
second child in the corpus, so functions such as x + 1/x drop out of the MDL ranking altogether.

(iv) The discounts need calibrating for a corpus this small. The standard Chen–Goodman estimates
are noisy for a vocabulary of ~15 symbols, and on synthetic corpora of ≤ 50 equations they make KN
worse than Katz at n = 2. A single discount per model chosen by cross-validation (optimum
D ≈ 0.7–0.9) gives KN a smaller excess codelength than Katz in all 56 combinations of synthetic
truth, corpus size and n. For a corpus the size of the real one the gain is 0.09–0.22 nats per
tree at n = 2.

(v) Generalisation across sources is weaker. Trained on the AI Feynman equations and tested on the
other 61, tuned KN and Katz differ by less than 2σ at all n tested (−0.33 ± 0.24 nats per equation
at n = 3). In the reverse direction KN gains 0.68 ± 0.19 nats per equation at n = 3.

(vi) KN separates real equations from corrupted ones slightly better: the AUC for a single swapped
operator rises by 0.014 ± 0.006, 0.024 ± 0.006 and 0.049 ± 0.009 at n = 2, 3 and 6. With
`code_constants=True` the prior also distinguishes power laws from exponentials (xᵃ from aˣ),
raising the AUC for constant–variable swaps from 0.50, a tie by construction, to 0.99. We keep the
original coding as the default. In the corpus, physical constants are symbols like any variable,
and which symbols are regressed and which are fitted is a separate modelling choice.

(vii) In real ESR rankings the change of smoothing reorders near-degenerate functions without
changing the winner. The top-10 functions of the dark energy fits overlap by 60–80%, and the
Pantheon+ top 10 at complexity 5 is unchanged. Per-function log-priors shift with a standard
deviation of 2–3 nats, comparable to the description-length gaps between the leading functions.
Changing the representation (Katz at n = 2 to Katz at n = 3 with root context) does change the
dark energy winner.

Together these results show that tuned KN with a root context is a better-calibrated prior on real
physics equations. Most of the gain over the published configuration comes from the root context
and from the longer contexts that KN can exploit, while the smoothing alone matters most for the
functions to which Katz assigns zero probability.

No corpus equation is a bare leaf, so the root context makes single-leaf functions expensive. At
n = 3 the log-prior of a₀ falls from −0.5 nats in the original representation to −3.5 (Katz) and
−5.2 to −5.9 nats (KN), which penalises constant models in ESR. All conclusions on real equations
also rest on 161 partly overlapping equations; an independent corpus such as the Encyclopaedia
Inflationaris potentials would test their generality.

## ESR benchmark (running)

The tests above concern the prior alone. To find out whether the improved prior helps ESR recover
the generating function, we are repeating the benchmark of the methods paper on Glamdring. ESR fits
mock data generated from Nguyen-8 and Korns-1, 4, 6 and 7, with N = 10–10⁴ points and five noise
realisations each (175 data sets), using the paper's operator basis up to complexity 8. The
functions are ranked by description length under the default ESR prior and five language-model
priors: Katz at n = 2 (the paper's choice), KN at n = 2, Katz and KN at n = 3 with root context,
and KN at n = 3 with root context and separate constant coding. For each data set we record the
description-length difference between the true function and the best alternative, and the rank of
the truth. Results to complexity 7 are expected on the evening of 7 November and the full results
on 8 November.

In a pilot to complexity 5 on one Nguyen-8 data set (N = 100), Katz at n = 2 ranked xᵃ above √x, as
the methods paper found for its language-model prior, while all three root-context priors ranked √x
first.
One data set does not establish a systematic improvement; the benchmark will.

## Recommendation

Use `smoothing='kn'` with `root_context=True` at n = 3 or 4, followed by `tune_discounts()`, and
keep the default constant coding unless the application needs to separate power laws from
exponentials. KN with D = 0.8 is now the package default, with a prominent note that published
results used `smoothing='katz'`. Whether the root context should also become the default awaits
the benchmark, which will show whether the penalty on single-leaf functions outweighs its gains.
