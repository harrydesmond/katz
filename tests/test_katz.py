import itertools
import os
import tempfile
import unittest

import numpy as np

try:  # ESR (and MPI) are only needed for scoring ESR function libraries
    import esr
    import esr.generation.duplicate_checker
    import esr.generation.generator

    from katz.esr_prior import compute_logprior, default_label, get_logconst

    HAVE_ESR = True
except ImportError:
    HAVE_ESR = False

from katz import tree_parser
from katz.back_off import BackOff
from katz.good_turing import GoodTuring
from katz.kneser_ney import KneserNey, chen_goodman_discounts
from katz.prior import KatzPrior
from katz.process_equations import (
    count_variables,
    split_by_punctuation,
    standardise_file,
)


def _require_sympy_114(test):
    """The corpus is parsed with sympy, and sympy < 1.14 writes e.g. sqrt(2*x0**2 + 4*x1)
    as a*sqrt(...) rather than sqrt(...), which changes the trees of 6 of the 161 physics
    and 2 of the 100 Feynman equations and so the prior. Reference values are for sympy
    >= 1.14 (the version required in setup.py)."""
    import sympy

    if tuple(int(v) for v in sympy.__version__.split(".")[:2]) < (1, 14):
        test.skipTest(f"reference values need sympy >= 1.14, not {sympy.__version__}")


class TestKatzPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 2
        basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
            ["+", "-", "*", "/", "pow"],
        ]
        # Reference values below are for the model of Bartlett et al. (2023): Katz, no root context
        cls.kp_feynman = KatzPrior(
            n,
            basis_functions,
            "data/FeynmanEquations.csv",
            "data/NewFeynman.csv",
            smoothing="katz",
            root_context=False,
        )
        cls.kp_feynman_no_input = KatzPrior(
            n, basis_functions, None, "data/NewFeynman.csv", smoothing="katz",
            root_context=False,
        )

        basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
            ["+", "-", "*", "/", "pow"],
        ]
        cls.kp_physics = KatzPrior(
            n,
            basis_functions,
            "data/PhysicsEquations.csv",
            "data/NewPhysics.csv",
            input_delimiter=";",
            smoothing="katz",
            root_context=False,
        )

    def test_logprior_string_equations(self):
        _require_sympy_114(self)
        equations = ["x0**2", "sin(x0) + sin(x1)", "sin(sin(x0+x1))"]

        # Feynman prior
        expected_results = [np.float64(-2.3), np.float64(-18.5), np.float64(-17.3)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_feynman.logprior(eq), expected, places=1)

        # Physics prior
        expected_results = [np.float64(-2.6), np.float64(-19.9), np.float64(-17.4)]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_physics.logprior(eq), expected, places=1)

    def test_logprior_list_equations(self):
        _require_sympy_114(self)
        equations = [
            ["+", "x0", "x0"],
            ["*", "2", "x0"],
            ["+", "x0", "x1"],
            ["+", "sin", "x0", "sin", "x1"],
        ]

        # Feynman prior
        expected_results = [
            np.float64(-9.5),
            np.float64(-4.0),
            np.float64(-5.9),
            np.float64(-18.5),
        ]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_feynman.logprior(eq), expected, places=1)
                self.assertAlmostEqual(
                    self.kp_feynman_no_input.logprior(eq), expected, places=1
                )

        # Physics prior
        expected_results = [
            np.float64(-9.2),
            np.float64(-4.0),
            np.float64(-5.5),
            np.float64(-19.9),
        ]
        for eq, expected in zip(equations, expected_results):
            with self.subTest(eq=eq):
                self.assertAlmostEqual(self.kp_physics.logprior(eq), expected, places=1)

    def test_op2str(self):
        expected = {
            "pi": "a",
            "0.5": "a",
            "Symbol": "x",
            "x0": "x",
            "x1": "x",
            "a0": "x",
            "a1": "x",
            "Add": "+",
            "Sub": "-",
            "Mul": "*",
            "Div": "/",
            "Pow": "pow",
            "sin": "sin",
        }
        for op, string in expected.items():
            with self.subTest(op=op):
                self.assertEqual(self.kp_feynman.coder.op2str(op), string)
                self.assertEqual(self.kp_feynman_no_input.coder.op2str(op), string)
                self.assertEqual(self.kp_physics.coder.op2str(op), string)

        # Check unknown operator raises Exception
        bad_ops = ["unknown", "Cosh", "Sinh", "garbage", "y0", "b2"]
        for op in bad_ops:
            with self.assertRaises(ValueError):
                self.kp_feynman.coder.op2str(op)
            with self.assertRaises(ValueError):
                self.kp_feynman_no_input.coder.op2str(op)
            with self.assertRaises(ValueError):
                self.kp_physics.coder.op2str(op)


class TestBackOff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 1
        cls.data_file = "data/romeoandjuliet.txt"
        with open(cls.data_file, "r") as f:
            cls.data = f.readlines()[80:]
        cls.data = [line.strip().split() for line in cls.data]
        cls.data = list(itertools.chain(*cls.data))
        cls.data = [(word,) for word in cls.data]

    def test_bo(self):
        self.bo = BackOff(self.data)
        words_in_play = [
            "Romeo",
            "Montague",
            "Juliet",
            "Capulet",
            "the",
            "a",
            "and",
            "to",
            "of",
            "in",
        ]
        for word in words_in_play:
            with self.subTest(word=word):
                self.assertIn(word, self.bo.words)
                self.assertTrue(self.bo.get_pbo(word, ()) > 0)
        words_not_in_play = [
            "Macbeth",
            "Hamlet",
            "Othello",
            "Computer",
            "Avocado",
            "Pineapple",
        ]
        for word in words_not_in_play:
            with self.subTest(word=word):
                self.assertNotIn(word, self.bo.words)
                self.assertEqual(self.bo.get_pbo(word, ()), 0)


@unittest.skipUnless(HAVE_ESR, "ESR is not installed")
class TestESRPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.comp = 5
        esr.generation.duplicate_checker.main("core_maths", cls.comp)
        # Location of the installed ESR package's function library (equal to
        # ESR/esr/function_library/core_maths/ when ESR is cloned into the repo root)
        cls.dirname = os.path.join(
            os.path.dirname(esr.__file__), "function_library", "core_maths", ""
        )
        cls.n = 2
        cls.basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
            ["+", "-", "*", "/", "pow"],
        ]
        cls.in_eqfile = "data/FeynmanEquations.csv"
        cls.out_eqfile = "data/NewFeynman.csv"
        cls.input_delimiter = ","

    def test_compute_logprior(self):

        # Ensure the directory exists
        os.makedirs(self.dirname, exist_ok=True)

        # Test get_logconst function
        for overwrite in [True, False]:
            get_logconst(self.comp, self.dirname, overwrite=overwrite)

        # Check if the logconst file is created
        logconst_file = os.path.join(
            self.dirname, f"compl_{self.comp}", f"logconst_{self.comp}.txt"
        )
        self.assertTrue(os.path.isfile(logconst_file))

        for use_tree, (label, prior_kwargs) in itertools.product(
            [False, True], [("katz", {"smoothing": "katz"}), ("kn", None)]
        ):

            # Test compute_logprior function
            for overwrite in [True, False]:
                compute_logprior(
                    self.comp,
                    self.n,
                    self.basis_functions,
                    self.dirname,
                    self.in_eqfile,
                    self.out_eqfile,
                    overwrite=overwrite,
                    input_delimiter=self.input_delimiter,
                    use_tree=use_tree,
                    prior_kwargs=prior_kwargs,
                )

            # Check if equation file is created
            eq_file = os.path.join(
                self.dirname, f"compl_{self.comp}", f"all_equations_{self.comp}.txt"
            )
            self.assertTrue(os.path.isfile(eq_file))

            # Check if the logprior file is created
            logprior_file = os.path.join(
                self.dirname,
                f"compl_{self.comp}",
                f"{label}_logprior_{self.n}_{self.comp}"
                + ("_tree" if use_tree else "")
                + ".txt",
            )
            self.assertTrue(os.path.isfile(logprior_file))

            # Check if the codelen file is created
            codelen_file = os.path.join(
                self.dirname,
                f"compl_{self.comp}",
                f"{label}_codelen_{self.n}_{self.comp}"
                + ("_tree" if use_tree else "")
                + ".txt",
            )
            self.assertTrue(os.path.isfile(codelen_file))

            # Load equations and logprior values
            logprior = np.loadtxt(logprior_file)
            codelen = np.loadtxt(codelen_file)
            with open(eq_file, "r") as f:
                equations = f.readlines()
            equations = [eq.strip() for eq in equations]
            self.assertEqual(len(equations), len(logprior), len(codelen))

            # Check that equations with same string have same katz prior
            if not use_tree:
                unique_strings = {}
                for index, string in enumerate(equations):
                    if string not in unique_strings:
                        unique_strings[string] = []
                    unique_strings[string].append(index)
                for s, indices in unique_strings.items():
                    if np.isfinite(logprior[indices[0]]):
                        self.assertTrue(
                            np.all(logprior[indices] == logprior[indices[0]])
                        )
                    else:
                        print(s, logprior[indices])
                        self.assertTrue(np.all(~np.isfinite(logprior[indices])))

            # Check that at least some values are finite
            self.assertTrue(np.any(np.isfinite(logprior)))

    def test_manifest_and_labels(self):
        """Output is reused only if its manifest matches the requested configuration"""
        import json

        get_logconst(self.comp, self.dirname, overwrite=False)
        self.assertEqual(default_label({}), "kn")
        self.assertEqual(default_label({"smoothing": "katz"}), "katz")
        self.assertEqual(
            default_label({"smoothing": "katz", "root_context": False}), "katz_noroot"
        )
        self.assertEqual(
            default_label({"root_context": True, "code_constants": True}),
            "kn_const",
        )
        base = os.path.join(self.dirname, f"compl_{self.comp}", "")
        prior_file = base + f"clash_logprior_{self.n}_{self.comp}.txt"
        manifest_file = base + f"clash_prior_{self.n}_{self.comp}.json"
        results = []
        for rc in [False, True, True]:
            compute_logprior(
                self.comp,
                self.n,
                self.basis_functions,
                self.dirname,
                self.in_eqfile,
                self.out_eqfile,
                overwrite=False,
                prior_kwargs={"root_context": rc},
                label="clash",
            )
            with open(manifest_file) as f:
                manifest = json.load(f)
            self.assertEqual(manifest["config"]["root_context"], rc)
            self.assertEqual(manifest["config"]["smoothing_kwargs"]["discounts"], 0.8)
            self.assertEqual(manifest["n_failed"], 0)
            results.append((np.loadtxt(prior_file), os.path.getmtime(prior_file)))
        # A different configuration under the same label is recomputed...
        self.assertFalse(np.allclose(results[0][0], results[1][0], equal_nan=True))
        # ... and the same configuration is reused
        self.assertEqual(results[1][1], results[2][1])
        for f in os.listdir(base):
            if f.startswith("clash_"):
                os.remove(base + f)


class TestGoodTuringSingleBucket(unittest.TestCase):
    """Tests for GoodTuring edge cases that previously caused NaN.

    When every token in the corpus is distinct (each appears exactly once),
    Nr has a single frequency bucket.  Before the fix this left Zr[-1] = 0,
    causing log(0) = -inf and linregress returning NaN for all outputs.
    """

    def _all_distinct_corpus(self):
        """Return a corpus where every item appears exactly once."""
        return [(i,) for i in range(5)]

    def test_zr_last_element_finite(self):
        """Zr[-1] must be positive and finite for a single-bucket Nr."""
        gt = GoodTuring(self._all_distinct_corpus())
        self.assertTrue(
            np.all(np.isfinite(gt.Zr)), f"Zr contains non-finite values: {gt.Zr}"
        )
        self.assertTrue(np.all(gt.Zr > 0), f"Zr contains non-positive values: {gt.Zr}")

    def test_slope_intercept_finite(self):
        """slope and intercept must be finite (not NaN) for a single-bucket Nr."""
        gt = GoodTuring(self._all_distinct_corpus())
        self.assertTrue(np.isfinite(gt.slope), f"slope is not finite: {gt.slope}")
        self.assertTrue(
            np.isfinite(gt.intercept), f"intercept is not finite: {gt.intercept}"
        )

    def test_get_S_finite(self):
        """get_S must return a positive, finite value for r >= 1."""
        gt = GoodTuring(self._all_distinct_corpus())
        for r in [1, 2, 3]:
            with self.subTest(r=r):
                s = gt.get_S(r)
                self.assertTrue(
                    np.isfinite(s) and s > 0,
                    f"get_S({r}) = {s} is not positive and finite",
                )

    def test_expected_count_finite(self):
        """expected_count must return a positive, finite value for seen words."""
        gt = GoodTuring(self._all_distinct_corpus())
        for word in self._all_distinct_corpus():
            with self.subTest(word=word):
                ec = gt.expected_count(word)
                self.assertTrue(
                    np.isfinite(ec) and ec > 0,
                    f"expected_count({word}) = {ec} is not positive and finite",
                )

    def test_single_element_corpus(self):
        """A corpus with exactly one element must not raise and must be finite."""
        gt = GoodTuring([(42,)])
        self.assertTrue(np.all(np.isfinite(gt.Zr)))
        self.assertTrue(np.isfinite(gt.slope))
        self.assertTrue(np.isfinite(gt.intercept))
        self.assertTrue(np.isfinite(gt.get_S(1)))


class TestGoodTuringMultiBucket(unittest.TestCase):
    """Sanity checks for GoodTuring with multiple frequency buckets (normal path)."""

    @classmethod
    def setUpClass(cls):
        # Two distinct frequency levels: (1,) appears twice, (2,) appears once.
        cls.corpus = [(1,), (1,), (2,)]
        cls.gt = GoodTuring(cls.corpus)

    def test_zr_finite(self):
        self.assertTrue(np.all(np.isfinite(self.gt.Zr)))

    def test_slope_intercept_finite(self):
        self.assertTrue(np.isfinite(self.gt.slope))
        self.assertTrue(np.isfinite(self.gt.intercept))

    def test_actual_count(self):
        self.assertEqual(self.gt.actual_count((1,)), 2)
        self.assertEqual(self.gt.actual_count((2,)), 1)
        self.assertEqual(self.gt.actual_count((99,)), 0)

    def test_expected_count_positive(self):
        for word in [(1,), (2,)]:
            with self.subTest(word=word):
                self.assertGreater(self.gt.expected_count(word), 0)


class TestSparseKatzPrior(unittest.TestCase):
    """Tests for KatzPrior with a tiny single-equation corpus.

    This is the case that triggered NaN: the equation being evaluated is
    present in the corpus but get_pbo returned NaN because GoodTuring
    failed on single-bucket Nr arrays.

    The right back-off model is now trained on full (parent, left_sibling,
    right_child) tuples so that the query context correctly includes the
    parent node, matching the fix to logprior.
    """

    # Minimal SimpleEquations-style CSV (semicolon-delimited)
    _SIMPLE_EQ_HEADER = (
        "Filename;Number;Output;Formula;# variables;"
        "v1_name;v1_low;v1_high;v2_name;v2_low;v2_high;"
        "v3_name;v3_low;v3_high;v4_name;v4_low;v4_high;"
        "v5_name;v5_low;v5_high;v6_name;v6_low;v6_high;"
        "v7_name;v7_low;v7_high;v8_name;v8_low;v8_high;"
        "v9_name;v9_low;v9_high;v10_name;v10_low;v10_high"
    )
    _SIMPLE_EQ_ROW = "Eq0;2;f;sin(x) + sin(x - y);2;x;1;3;y;1;3;;;;;;;;;;;;;;;;;;;;;;;"

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._in_file = os.path.join(cls._tmpdir, "SimpleEquations.csv")
        cls._out_file = os.path.join(cls._tmpdir, "NewSimple.csv")

        with open(cls._in_file, "w") as f:
            f.write(cls._SIMPLE_EQ_HEADER + "\n")
            f.write(cls._SIMPLE_EQ_ROW + "\n")

        cls.basis_functions = [["x"], ["sin"], ["+", "-"]]
        cls.n = 2
        cls.kp = KatzPrior(
            cls.n,
            cls.basis_functions,
            cls._in_file,
            cls._out_file,
            input_delimiter=";",
            smoothing="katz",
            root_context=False,
        )
        cls.eq = "sin(x0) + sin(x0 - x1)"

    def test_logprior_is_finite(self):
        """logprior must return a finite value, not NaN, for an equation in the corpus."""
        p = self.kp.logprior(self.eq)
        self.assertTrue(np.isfinite(p), f"logprior returned non-finite value: {p}")

    def test_logprior_is_negative(self):
        """Log-probabilities must be non-positive."""
        p = self.kp.logprior(self.eq)
        self.assertLessEqual(p, 0.0)

    def test_logprior_expected_value(self):
        """logprior must match the expected value for the minimal corpus.

        With n=2 and a single training equation the right back-off corpus
        contains two 3-grams, each appearing exactly once.  The Good-Turing
        fallback (d=1, no discounting) means both right-child probabilities
        equal 1, contributing 0 to the log-prior.  The value is therefore
        determined entirely by the left probabilities, which are unchanged
        by this fix.
        """
        p = self.kp.logprior(self.eq)
        self.assertAlmostEqual(p, -3.0, places=1)

    def test_out_file_created(self):
        """standardise_file must write the output CSV."""
        self.assertTrue(os.path.isfile(self._out_file))

    def test_out_file_contains_standardised_equation(self):
        """The output CSV must contain the standardised equation."""
        import pandas as pd

        df = pd.read_csv(self._out_file)
        self.assertIn("sin(x0)+sin(x0-x1)", df["New Formula"].tolist())


class TestKneserNey(unittest.TestCase):

    def test_worked_example(self):
        """Two bigrams seen once each: an unseen bigram keeps a share of the mass"""
        kn = KneserNey([("sin", "x"), ("cos", "y")], discounts=0.75)
        self.assertAlmostEqual(kn.prob("x", ("sin",)), 0.625)
        self.assertAlmostEqual(kn.prob("y", ("sin",)), 0.375)

    def test_hand_computed_trigram(self):
        """Compare against a hand calculation with a fixed discount D = 0.5"""
        corpus = [
            ("a", "b", "c"),
            ("a", "b", "c"),
            ("d", "b", "c"),
            ("a", "b", "e"),
            ("b", "e"),  # shorter event: raw count at order 2
        ]
        kn = KneserNey(corpus, order=3, vocab=["z"], discounts=0.5)
        # Order 1: continuation counts c=1, e=1; gamma = 0.5; uniform over {c, e, z}
        p1 = {"c": 0.25 + 0.5 / 3, "e": 0.25 + 0.5 / 3, "z": 0.5 / 3}
        for w, p in p1.items():
            self.assertAlmostEqual(kn.prob(w, ()), p)
        # Order 2, context b: adjusted counts (b,c)=2 [types a,d], (b,e)=2 [raw 1 + type a]
        p2 = {w: (2 - 0.5) / 4 * (w != "z") + 0.25 * p1[w] for w in p1}
        for w, p in p2.items():
            self.assertAlmostEqual(kn.prob(w, ("b",)), p)
        # Order 3, context (a, b): raw counts c=2, e=1
        p3 = {"c": 1.5 / 3 + p2["c"] / 3, "e": 0.5 / 3 + p2["e"] / 3, "z": p2["z"] / 3}
        for w, p in p3.items():
            self.assertAlmostEqual(kn.prob(w, ("a", "b")), p)
        self.assertAlmostEqual(kn.prob("c", ("d", "b")), 0.5 + 0.5 * p2["c"])
        # Unseen context backs off to the longest seen suffix
        self.assertAlmostEqual(kn.prob("c", ("q", "b")), p2["c"])
        # Contexts longer than order - 1 are truncated
        self.assertAlmostEqual(kn.prob("c", ("q", "a", "b")), p3["c"])
        self.assertEqual(kn.get_pbo("c", ("a", "b")), kn.prob("c", ("a", "b")))

    def test_normalisation_and_positivity(self):
        rng = np.random.default_rng(42)
        V = list("abcdef")
        corpus = [tuple(rng.choice(V, size=rng.integers(1, 5))) for _ in range(200)]
        for discounts in ["mkn", "kn", 0.3, [0.2, 0.5, 0.9, 0.7]]:
            kn = KneserNey(corpus, order=4, vocab=V + ["g"], discounts=discounts)
            for L in range(4):
                for h in itertools.product(V + ["g"], repeat=L):
                    p = np.array([kn.prob(w, h) for w in kn.vocab])
                    with self.subTest(discounts=discounts, h=h):
                        self.assertAlmostEqual(p.sum(), 1.0, places=12)
                        self.assertTrue(np.all(p > 0))
        self.assertEqual(kn.prob("unknown", ()), 0.0)

    def test_continuation_counts(self):
        """A frequent word seen in a single context gets less lower-order mass than a
        rarer word seen in many contexts (and vice versa for raw unigram frequency)"""
        corpus = [("s", "f")] * 10 + [(c, "g") for c in "abcde"]
        kn = KneserNey(corpus, order=2, discounts=0.5)
        self.assertLess(kn.prob("f", ()), kn.prob("g", ()))
        self.assertLess(kn.prob("f", ("q",)), kn.prob("g", ("q",)))
        self.assertGreater(kn.prob("f", ("s",)), 0.9)

    def test_chen_goodman_discounts(self):
        D = chen_goodman_discounts({1: 10, 2: 5, 3: 3, 4: 2})
        np.testing.assert_allclose(D, [0.5, 1.1, 3 - 4 * 0.5 * 2 / 3])
        # Undefined estimates fall back to 0.5 and estimates are clipped to [1e-3, r]
        np.testing.assert_allclose(chen_goodman_discounts({3: 1}), [0.5, 0.5, 0.5])
        D = chen_goodman_discounts({1: 1, 2: 1, 3: 100, 4: 1})
        self.assertTrue(np.all(D >= 1e-3) and np.all(D <= [1, 2, 3]))

    def test_discount_options(self):
        corpus = [("a", "b"), ("a", "c"), ("b", "c"), ("c",)]
        kn = KneserNey(corpus, discounts=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        np.testing.assert_allclose(kn.discounts[1:], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        kn = KneserNey(corpus, discounts=0.9, discount_scale=2.0)
        np.testing.assert_allclose(kn.discounts[1:], [[1.0, 1.8, 1.8]] * 2)
        with self.assertRaises(ValueError):
            KneserNey(corpus, discounts="bad")
        with self.assertRaises(ValueError):
            KneserNey([])


class TestKatzPriorOptions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
            ["+", "-", "*", "/", "pow"],
        ]
        cls.kp_kn = KatzPrior(
            2, cls.basis_functions, None, "data/NewPhysics.csv", smoothing="kn"
        )

    def test_kn_finite_and_normalised(self):
        eqs = list(self.kp_kn.all_eq[:20]) + [
            "arccos(arccos(arccos(x0)))",
            "tanh(tanh(x0))",
            "sin(x0) + sin(x1)",
        ]
        for eq in eqs:
            with self.subTest(eq=eq):
                p = self.kp_kn.logprior(eq)
                self.assertTrue(np.isfinite(p) and p < 0)
        for model, vocab in [
            (self.kp_kn.backoff_left, self.kp_kn.vocab_left),
            (self.kp_kn.backoff_right, self.kp_kn.vocab_right),
        ]:
            contexts = set()
            for k in range(1, model.order + 1):
                contexts.update(model.total[k].keys())
            for h in contexts:
                self.assertAlmostEqual(sum(model.prob(w, h) for w in vocab), 1.0, 12)

    def test_equations_argument(self):
        """Building from a list of equations matches building from the file"""
        kp = KatzPrior(
            2, self.basis_functions, equations=self.kp_kn.all_eq, smoothing="kn"
        )
        for eq in ["x0**2", "sin(x0)*exp(x1)", "x0 + 3*x1"]:
            self.assertEqual(kp.logprior(eq), self.kp_kn.logprior(eq))
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=["x0"], smoothing="bad")
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=["x0"], right_backoff="bad")

    def test_code_constants(self):
        eqs = ["pow(x0, a0)", "pow(a0, x0)"]
        for flag in [False, True]:
            kp = KatzPrior(
                2, self.basis_functions, equations=["x0**2"], code_constants=flag
            )
            t = [kp.coder.process_all_equations(3, [eq], 1) for eq in eqs]
            with self.subTest(code_constants=flag):
                if flag:
                    self.assertNotEqual(t[0], t[1])
                    self.assertIn(kp.coder.code["a"], t[0][1])
                else:
                    self.assertEqual(t[0], t[1])

    def test_root_context(self):
        n = 2
        kp = KatzPrior(
            n,
            self.basis_functions,
            equations=self.kp_kn.all_eq[:30],
            root_context=True,
            smoothing="kn",
        )
        none = kp.coder.code["None"]
        t = kp.coder.process_all_equations(n + 1, ["sin(sin(x0 + x1))"], 2)
        self.assertEqual(t[0], (none, none, kp.coder.code["sin"]))
        self.assertTrue(all(len(tt) == n + 2 for tt in t[1:]))
        left, right = kp.split_events(t)
        self.assertTrue(all(len(tt) == n + 1 for tt in left))
        self.assertTrue(np.isfinite(kp.logprior("sin(sin(x0 + x1))")))

    def test_right_backoff(self):
        n = 2
        kp = KatzPrior(
            n,
            self.basis_functions,
            equations=self.kp_kn.all_eq[:30],
            right_backoff="parent",
        )
        code = kp.coder.code
        t = kp.coder.process_all_equations(n + 1, ["x0 + sin(x1)"], 2)
        _, right = kp.split_events(t)
        # (parent, left sibling, right) -> (left sibling, parent, right)
        self.assertEqual(right[0], (code["x"], code["+"], code["sin"]))
        self.assertTrue(np.isfinite(kp.logprior("x0 + sin(x1)")))

    def test_tune_discounts(self):
        kp = KatzPrior(
            2,
            self.basis_functions,
            equations=self.kp_kn.all_eq,
            smoothing="kn",
            root_context=True,
        )
        res = kp.tune_discounts("single", k=5, grid=np.linspace(0.1, 1.0, 10))
        for side in ("left", "right"):
            self.assertTrue(0.1 <= res[side]["discounts"] <= 1.0)
        # Chosen discounts are applied to the refitted models
        self.assertAlmostEqual(
            kp.backoff_left.discounts[1, 0], res["left"]["discounts"]
        )
        self.assertAlmostEqual(
            kp.backoff_right.discounts[2, 2], res["right"]["discounts"]
        )
        # Minimum of the CV curve is at the chosen value
        j = np.argmin(res["cv_nll"][0])
        self.assertAlmostEqual(res["grid"][j], res["left"]["discounts"])
        res = kp.tune_discounts("scale", k=3, grid=[0.5, 1.0])
        self.assertIn(res["right"]["discount_scale"], [0.5, 1.0])
        # Count-dependent discounts (D1, D2+), the same at every order
        grid = [(0.5, 0.5), (0.8, 0.8), (0.6, 1.2), (0.9, 1.5)]
        res = kp.tune_discounts("count", k=3, grid=grid)
        for side, model in (("left", kp.backoff_left), ("right", kp.backoff_right)):
            D = np.asarray(res[side]["discounts"])
            self.assertEqual(D.shape, (3, 3))
            self.assertIn(tuple(D[0, :2]), grid)
            self.assertTrue(np.all(D == D[0]) and D[0, 1] == D[0, 2])
            self.assertTrue(np.allclose(model.discounts[1:], D))
        j = np.argmin(res["cv_nll"][1])
        self.assertEqual(tuple(res["grid"][j]), tuple(res["right"]["discounts"][0][:2]))
        self.assertTrue(np.isfinite(kp.logprior("sin(x0) + x1")))
        with self.assertRaises(ValueError):
            kp.tune_discounts("count", grid=[0.5, 0.8])
        with self.assertRaises(ValueError):
            KatzPrior(
                2, self.basis_functions, equations=["x0+x1"], smoothing="katz"
            ).tune_discounts()

    def test_default_is_kn(self):
        """Kneser-Ney with a fixed discount is the default; FunctionPrior is an alias"""
        from katz.prior import DEFAULT_KN_DISCOUNT, FunctionPrior

        eqs = self.kp_kn.all_eq[:40]
        kp = FunctionPrior(2, self.basis_functions, equations=eqs)
        self.assertEqual(kp.smoothing, "kn")
        self.assertEqual(kp.smoothing_kwargs["discounts"], DEFAULT_KN_DISCOUNT)
        explicit = KatzPrior(
            2,
            self.basis_functions,
            equations=eqs,
            smoothing="kn",
            smoothing_kwargs={"discounts": DEFAULT_KN_DISCOUNT},
        )
        mkn = KatzPrior(
            2,
            self.basis_functions,
            equations=eqs,
            smoothing="kn",
            smoothing_kwargs={"discounts": "mkn"},
        )
        for eq in ["x0**2", "sin(x0) + x1"]:
            self.assertEqual(kp.logprior(eq), explicit.logprior(eq))
            self.assertNotEqual(kp.logprior(eq), mkn.logprior(eq))

    def test_single_leaf(self):
        """A single-leaf function contributes exactly one (root) event"""
        for rc, sm in itertools.product([False, True], ["katz", "kn"]):
            kp = KatzPrior(
                2,
                self.basis_functions,
                equations=self.kp_kn.all_eq[:40],
                root_context=rc,
                smoothing=sm,
            )
            for eq in ["x0", "a0", ["x0"], ["a0"]]:
                with self.subTest(root_context=rc, smoothing=sm, eq=eq):
                    t = kp.coder.process_all_equations(3, [eq], kp.maxvar)
                    self.assertEqual(len(t), 1)
                    pl, pr = kp.event_probs(t)
                    self.assertEqual((len(pl), len(pr)), (1, 0))
                    root = t[0]
                    p = kp.backoff_left.get_pbo(root[-1], root[:-1])
                    self.assertAlmostEqual(kp.logprior(eq), np.log(p))




class TestEdgeCases(unittest.TestCase):
    """Edge cases found by an independent audit of the code"""

    basis_functions = [
        ["a", "x"],
        ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
        ["+", "-", "*", "/", "pow"],
    ]

    def test_bare_leaf_and_maxvar(self):
        self.assertEqual(split_by_punctuation("x0"), ["x0"])
        self.assertEqual(split_by_punctuation("a0"), ["a0"])
        self.assertEqual(split_by_punctuation("x0+x1"), ["x0", "+", "x1"])
        self.assertEqual(count_variables(["x10 + 1"]), 11)
        self.assertEqual(count_variables(["sin(x0)", "a0", "exp(x2)"]), 3)
        self.assertEqual(count_variables(["2*a0"]), 1)
        with tempfile.TemporaryDirectory() as td:
            fname = os.path.join(td, "eq.csv")
            with open(fname, "w") as f:
                f.write("New Formula\nx10 + 1\nx0\n")
            eqs, maxvar = standardise_file(None, fname, ",")
            self.assertEqual((eqs, maxvar), (["x10 + 1", "x0"], 11))
        for fname, expected in [("data/NewPhysics.csv", 11), ("data/NewFeynman.csv", 9)]:
            with self.subTest(fname=fname):
                self.assertEqual(standardise_file(None, fname, ",")[1], expected)

    def test_kn_parameter_validation(self):
        corpus = [("r", "a"), ("r", "b")]
        for kw in [
            dict(discounts=0.0),
            dict(discounts=float("nan")),
            dict(discounts=-0.5),
            dict(discount_scale=-1.0),
            dict(discount_scale=float("inf")),
            dict(discounts=[0.5, 0.5, 0.5]),  # wrong shape for order 2
            dict(order=0),
        ]:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    KneserNey(corpus, vocab=["c"], **kw)
        m = KneserNey(corpus, vocab=["c"], discounts=[0.5, 0.7])
        self.assertTrue(m.prob("c", ("r",)) > 0)
        # Discounts above their maximum D_r = r are capped
        m = KneserNey(corpus, discounts=0.9, discount_scale=3.0)
        self.assertTrue(np.all(m.discounts[1:] == [1.0, 2.0, 2.7]))

    def test_kn_empty_corpus(self):
        m = KneserNey([], order=2, vocab=["a", "b", "c", "d"])
        for w in "abcd":
            self.assertAlmostEqual(m.prob(w, ("z",)), 0.25)
        with self.assertRaises(ValueError):
            KneserNey([])
        m = KneserNey([("r", "a")] * 5, cache_size=2)
        p = [m.prob(w, ("r",)) for w in "a" * 3]
        self.assertTrue(len(m._cache) <= 2 and p[0] == p[2])

    def test_corpus_without_binary_operators(self):
        eqs = ["sin(x0)", "exp(sqrt(x0))", "x0"]
        kp = KatzPrior(2, self.basis_functions, equations=eqs)
        self.assertEqual(len(kp.backoff_right.counts[1]), 0)
        for eq in ["sin(x0)", "x0 * sin(x0)", "x0"]:
            self.assertTrue(np.isfinite(kp.logprior(eq)))
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=eqs, smoothing="katz")

    def test_tune_discounts_validation(self):
        eqs = ["sin(x0) + x1", "x0*x1", "exp(x0) - x1", "sqrt(x0)/x1"]
        with self.assertRaises(ValueError):
            KatzPrior(2, self.basis_functions, equations=eqs[:1]).tune_discounts()
        kp = KatzPrior(2, self.basis_functions, equations=eqs)
        before = kp.logprior("sin(x0) * x1")
        for kw in [dict(k=1), dict(grid=[]), dict(grid=[0.5, np.nan]), dict(grid=[0.0])]:
            with self.subTest(**{k: str(v) for k, v in kw.items()}):
                with self.assertRaises(ValueError):
                    kp.tune_discounts(**kw)
        self.assertEqual(kp.logprior("sin(x0) * x1"), before)
        res = kp.tune_discounts(k=2)  # two equations per fold
        self.assertTrue(np.all(np.isfinite(res["cv_nll"])))

    def test_basis_order_is_reproducible(self):
        kp = KatzPrior(2, self.basis_functions, equations=["x0 + x1"])
        self.assertEqual(kp.basis_functions[0], ["a", "x", "x0", "x1"])

    def test_symbols_beyond_corpus_range(self):
        # x_i and a_i outside the corpus's variable range are real symbols and leaves
        eqs = ["x0*sin(x0) + x0", "sqrt(x0**2 + x0)", "x0/(x0 + 1)", "exp(-x0**2)"]
        kp = KatzPrior(2, self.basis_functions, equations=eqs)
        self.assertEqual(kp.maxvar, 1)
        for e0, e1 in [("sqrt(x0**2)", "sqrt(x3**2)"), ("sqrt(a0**2)", "sqrt(a7**2)"),
                       ("x0*a2 + a5", "x3*a2 + a5")]:
            with self.subTest(eq=e1):
                self.assertAlmostEqual(kp.logprior(e1), kp.logprior(e0), places=12)

    def test_normalised_katz(self):
        # the Katz conditionals need not sum to one; the opt-in normalised comparator does
        import pandas as pd

        eqs = list(pd.read_csv("data/NewPhysics.csv")["New Formula"])
        raw = KatzPrior(3, self.basis_functions, equations=eqs, smoothing="katz")
        norm = KatzPrior(3, self.basis_functions, equations=eqs, smoothing="katz",
                         smoothing_kwargs={"normalise": True})
        m, r = norm.backoff_right, raw.backoff_right
        contexts = {tuple(e[-(norm.n + 1):-1]) for t in norm.eq_tuples
                    for e in norm.split_events(t)[1]}
        raw_sums = [sum(r.get_pbo(w, h) for w in r.words) for h in contexts]
        self.assertLess(min(raw_sums), 0.99)  # the original model loses mass
        for h in contexts:
            p = [m.get_pbo(w, h) for w in m.words]
            self.assertTrue(min(p) >= 0)
            self.assertAlmostEqual(sum(p), 1.0, places=12)
        # the default Katz model is unchanged
        self.assertEqual(raw.logprior("sin(x0) + x1"),
                         KatzPrior(3, self.basis_functions, equations=eqs,
                                   smoothing="katz").logprior("sin(x0) + x1"))

    def test_cache_size(self):
        for bad in (-1, 2.8, True, "10"):
            with self.subTest(cache_size=bad), self.assertRaises(ValueError):
                KneserNey([("a", "b")], order=2, cache_size=bad)
        corpus = [("r", "a", "b"), ("r", "b", "a"), ("r", "a", "a")]
        ref = KneserNey(corpus, vocab=["a", "b", "c"])
        for size in (0, 1, 3):
            m = KneserNey(corpus, vocab=["a", "b", "c"], cache_size=size)
            for w, h in itertools.product("abcz", [(), ("a",), ("r", "a")]):
                self.assertEqual(m.prob(w, h), ref.prob(w, h))
            self.assertLessEqual(len(m._cache), size)

    def test_random_normalisation(self):
        # every conditional distribution is a probability distribution over the vocabulary
        rng = np.random.default_rng(420)
        for j in range(200):
            order = int(rng.integers(1, 6))
            corpus = [tuple(rng.integers(0, 4, size=int(rng.integers(1, order + 1))))
                      for _ in range(int(rng.integers(0, 40)))]
            discounts = ["mkn", "kn", 0.8, np.tile([0.8, 1.3, 1.7], (order, 1))][j % 4]
            m = KneserNey(corpus, order=order, vocab=range(4), discounts=discounts)
            for k in range(min(order, 4)):
                h = tuple(rng.integers(0, 4, size=k))
                p = np.array([m.prob(w, h) for w in range(4)])
                self.assertTrue(np.all(np.isfinite(p)) and np.all(p >= 0))
                self.assertAlmostEqual(p.sum(), 1.0, places=12)

    def test_corpus_parse_is_reproducible(self):
        # the parsed corpus defines the prior: fail if sympy changes how it writes it
        import json

        import pandas as pd
        import sympy

        _require_sympy_114(self)
        with open(os.path.join(os.path.dirname(__file__), "corpus_labels_sympy1.14.json")) as f:
            ref = json.load(f)
        for name in ("NewPhysics", "NewFeynman"):
            eqs = list(pd.read_csv(f"data/{name}.csv")["New Formula"])
            kp = KatzPrior(2, ref["basis_functions"], equations=eqs, smoothing="katz")
            locs = {f"{c}{i}": sympy.Symbol(f"{c}{i}", real=True)
                    for c in "xa" for i in range(kp.maxvar)}
            for i, eq in enumerate(eqs):
                with self.subTest(corpus=name, eq=eq):
                    self.assertEqual(kp.coder.equation2labels(eq, locs), ref["labels"][name][i])


try:
    from katz import esr_prior as _esr_prior

    HAVE_MPI = True
except ImportError:
    HAVE_MPI = False


@unittest.skipUnless(HAVE_MPI, "katz.esr_prior needs mpi4py")
class TestPriorCache(unittest.TestCase):
    """compute_logprior reuses saved scores only if every input and output is unchanged"""

    corpus = ["x0*sin(x0) + x0", "sin(x0)**2", "x0/(x0 + 1)", "sqrt(x0) + sin(x0)"]
    basis = [["a", "x"], ["sqrt", "sin", "square", "inv"], ["+", "-", "*", "/", "pow"]]

    def _library(self, td, fcn, tree):
        d = os.path.join(td, "compl_1")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "all_equations_1.txt"), "w") as f:
            f.write(fcn + "\n")
        with open(os.path.join(d, "trees_1.txt"), "w") as f:
            f.write(repr(tree) + "\n")
        _esr_prior.get_logconst(1, td)

    def _run(self, td, **kw):
        _esr_prior.compute_logprior(1, 2, self.basis, td, None, None,
                                    prior_kwargs={"equations": self.corpus}, **kw)
        base = os.path.join(td, "compl_1", "kn_")
        return (float(np.loadtxt(base + "logprior_2_1.txt")),
                os.stat(base + "prior_2_1.json").st_mtime_ns)

    def test_reuse_and_invalidation(self):
        kp = KatzPrior(2, self.basis, equations=self.corpus)
        with tempfile.TemporaryDirectory() as td:
            self._library(td, "x0", ["x0"])
            v0, t0 = self._run(td)
            self.assertAlmostEqual(v0, -kp.logprior("x0"), places=10)
            self.assertEqual(self._run(td), (v0, t0))  # unchanged inputs: reused
            # new candidate library (the audit's example)
            self._library(td, "sin(x0)", ["sin", "x0"])
            v1, t1 = self._run(td)
            self.assertAlmostEqual(v1, -kp.logprior("sin(x0)"), places=10)
            # changed constant penalties
            lc = os.path.join(td, "compl_1", "logconst_1.txt")
            np.savetxt(lc, [1.5])
            v2, t2 = self._run(td)
            self.assertNotEqual(t2, t1)
            code = float(np.loadtxt(os.path.join(td, "compl_1", "kn_codelen_2_1.txt")))
            self.assertAlmostEqual(code, v1 + 1.5, places=10)
            # tampered output
            np.savetxt(os.path.join(td, "compl_1", "kn_logprior_2_1.txt"), [0.0])
            v3, t3 = self._run(td)
            self.assertAlmostEqual(v3, v1, places=10)
            self.assertNotEqual(t3, t2)
            # logconst follows the tree file
            self._library(td, "sqrt(x0)", ["sqrt", "x0"])
            self.assertEqual(float(np.loadtxt(lc)), 0.0)


class TestTreeParser(unittest.TestCase):
    """The built-in (sympy-only) tree parser"""

    basis_functions = [
        ["a", "x"],
        ["sqrt", "exp", "log", "sin", "cos", "arcsin", "arccos", "tanh", "inv"],
        ["+", "-", "*", "/", "pow"],
    ]
    equations = [
        "x0**2",
        "sin(x0) + x1",
        "x0 - x1",
        "x0/x1",
        "1/x0",
        "sqrt(x0)*exp(-x1)",
        "pow(Abs(a0), x0) + a1",
        "a0*x0**3 + 2.5",
        "x0",
        "arccos(tanh(x0))/log(x1)",
    ]

    def test_labels(self):
        kp = KatzPrior(2, self.basis_functions, equations=["x0 + x1"])
        import sympy

        x = {f"x{i}": sympy.Symbol(f"x{i}", real=True) for i in range(2)}
        a = {f"a{i}": sympy.Symbol(f"a{i}", real=True) for i in range(2)}
        labels = kp.coder.equation2labels("x0 - x1", {**x, **a})
        self.assertEqual(labels, ["-", "x0", "x1"])
        labels = kp.coder.equation2labels("sqrt(x0)/x1", {**x, **a})
        self.assertEqual(labels, ["/", "sqrt", "x0", "x1"])
        labels = kp.coder.equation2labels("pow(Abs(a0), x0)", {**x, **a})
        self.assertEqual(labels, ["pow", "a0", "x0"])

    @unittest.skipUnless(HAVE_ESR, "ESR is not installed")
    def test_same_as_esr(self):
        import katz.process_equations as pe

        import pandas as pd

        eqs = self.equations + list(pd.read_csv("data/NewPhysics.csv")["New Formula"])
        kp = KatzPrior(3, self.basis_functions, equations=eqs)  # basis x0 ... x10
        for rc, cc in [(False, False), (True, True)]:
            coder = pe.SymbolCoder(kp.basis_functions, code_constants=cc, root_context=rc)
            try:
                pe.generator = esr.generation.generator
                ref = [coder.process_all_equations(4, [e], 11) for e in eqs]
            finally:
                pe.generator = tree_parser
            new = [coder.process_all_equations(4, [e], 11) for e in eqs]
            self.assertEqual(new, ref)


def _load_assessment(name):
    import importlib.util
    import sys

    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assessment")
    if here not in sys.path:
        sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBenchmarkTruth(unittest.TestCase):
    """Truth identification in the ESR benchmark (assessment/esr_benchmark.py)"""

    @classmethod
    def setUpClass(cls):
        cls.bench = _load_assessment("esr_benchmark")

    def test_likelihood_does_not_decide_equivalence(self):
        # the second audit's counterexamples: near-equal likelihoods are not the truth
        tp = self.bench.truth_preference
        r = tp(["truth", "different_model"], [5.0, 1.0], [1e5, 1e5 + 0.08], ["truth"])
        self.assertEqual((r["pref"], r["top"], r["rank_strict"]), (-4.0, False, 2))
        r = tp(["truth", "alternative_a", "alternative_b", "far_other"], [5.0, 1.0, 2.0, 6.0],
               [1e4, 1e4 + 0.009, 1e4 + 0.018, 1e4 + 0.1], ["truth"])
        self.assertEqual((r["pref"], r["rank_strict"], r["best_other"]),
                         (-4.0, 3, "alternative_a"))
        self.assertTrue(np.isnan(tp(["x", "a0"], [1.0, 2.0], [1.0, 1.0], ["sqrt(x)"])["pref"]))

    def test_offset_does_not_hide_a_difference(self):
        r = self.bench.truth_preference(
            ["a0 + sin(x)", "a0"], [5.0, 4.0], [5.0, 6.0], ["a0 + sin(x)"],
            params=np.array([[1e6], [1e6]]), xr=[0, 6.2], sig=0.5)
        self.assertEqual((r["pref"], r["top"], r["best_other"], r["n_equiv"]),
                         (-1.0, False, "a0", 0))
        self.assertAlmostEqual(r["best_other_maxdev"], 2.0, places=2)  # |sin| <= 1, sig 0.5
        self.assertTrue(r["best_is_null"])

    def test_same_family(self):
        sf = self.bench.same_family
        self.assertTrue(sf("sqrt(x)", "x**0.5", [], [], [0, 4]))
        self.assertFalse(sf("sqrt(x)", "pow(x,a0)", [], [0.5], [0, 4]))  # extra parameter
        self.assertTrue(sf("a0 + a1*x", "a0*(a1 + x)", [1.57, 2.43], [2.43, 0.646], [-50, 50]))
        self.assertFalse(sf("a0 + a1*x", "a0*x", [1.57, 2.43], [2.43], [-50, 50]))
        self.assertTrue(sf("a0 + a1*sin(x)", "a0 - sin(x)*cos(a1)", [-2.3, 0.13],
                           [-2.3, np.arccos(-0.13)], [-50, 50]))
        self.assertFalse(sf("a0 + a1*sin(x)", "a0/(a1 + sin(x))", [-2.3, 0.13],
                            [-41.1, 17.7], [-50, 50]))

    def test_same_family_as_truth(self):
        # tested at the generating function, so independent of noisy fitted parameters
        sft = self.bench.same_family_as_truth
        for bench, g, expected in [
            ("korns_6", "a0 + sqrt(a1)*sqrt(x)", True),  # a1 -> a1**2 (amplitude > 0)
            ("korns_6", "(a0 + sqrt(x))/sqrt(a1)", True),
            ("korns_6", "a0 + a1*pow(x,a2)", False),  # a larger family
            ("korns_4", "a0*(a1 + sin(x))", True),
            ("korns_7", "a0 - pow(Abs(a1),x)", False),  # cannot represent the truth
            # represents the truth only at sin(a2) = 1, so not the same family near it
            ("korns_7", "a0*(-sin(a2) + pow(Abs(a1),x))", False),
            ("nguyen_8", "x**0.5", True),
            ("nguyen_8", "x**2*pow(x,a0)", False),
            ("korns_1", "x/a0 + a1", True),
        ]:
            with self.subTest(bench=bench, g=g):
                self.assertEqual(sft(g, bench, np.array([np.nan, -1.0, 3.0])), expected)

    def test_unlisted_representation_counts_as_truth(self):
        fun = ["sqrt(x)", "x**0.5", "pow(x,a0)", "x"]
        par = np.array([[0.0], [0.0], [0.5], [0.0]])
        r = self.bench.truth_preference(fun, [3.0, 2.0, 2.5, 4.0], [1.0, 1.0, 0.99, 5.0],
                                        ["sqrt(x)"], params=par, xr=[0, 4])
        self.assertEqual((r["n_equiv"], r["best_other"], r["pref"], r["top"]),
                         (1, "pow(x,a0)", 0.5, True))
        self.assertEqual((r["rank_strict"], r["pref_strict"]), (3, -1.0))

    def test_truth_variants_reproduce_the_truth(self):
        # every listed variant represents the generating function exactly on its domain
        import re

        import scipy.optimize
        import sympy

        b, rng = self.bench, np.random.default_rng(1)
        # random starts rarely reach this one (a0 ~ -214, a1 = exp(-0.547), a2 = pi/2)
        hints = {"a0*(-sin(a2) + pow(Abs(a1),x))": [-200.0, 0.5, 1.4]}
        for name, (f, xr, variants) in b.BENCHMARKS.items():
            xs = np.linspace(xr[0], xr[1], 200)
            T = sympy.lambdify(sympy.Symbol("x"), sympy.sympify(f), "numpy")(xs) * np.ones(200)
            for v in variants:
                k = b._lambda(v)[1]

                def resid(p, v=v):
                    r = (b.fitted_curve(v, p, xs) - T) / np.ptp(T)
                    return np.where(np.isfinite(r), r, 1e6)

                best = np.max(np.abs(resid(np.zeros(0)))) if k == 0 else np.inf
                bases = [int(j) for j in re.findall(r"pow\(Abs\(a(\d+)\)", v)]
                for i in range(60 if k else 0):
                    p0 = rng.standard_normal(k) * rng.choice([0.01, 1, 10, 100, 300], k)
                    p0[bases] = rng.uniform(0.3, 1.8, len(bases))  # bases of exponentials
                    p0 = np.array(hints[v]) if (v in hints and i == 0) else p0
                    try:
                        fit = scipy.optimize.least_squares(resid, p0, xtol=1e-15, ftol=1e-15,
                                                           gtol=1e-15, max_nfev=2000)
                    except ValueError:
                        continue
                    best = min(best, np.max(np.abs(fit.fun)))
                    if best < 1e-8:
                        break
                with self.subTest(benchmark=name, variant=v):
                    self.assertLess(best, 1e-6)


class TestRerank(unittest.TestCase):
    """Ranking comparisons (assessment/esr_rerank.py)"""

    @classmethod
    def setUpClass(cls):
        cls.compare = staticmethod(_load_assessment("esr_rerank").compare)

    def test_top_k_with_ties_and_few_candidates(self):
        import pandas as pd

        df = pd.DataFrame(dict(fcn=list("abcdefghijklmnopqrst"), DL_a=[1.0] * 20,
                               DL_b=[1.0] * 20))
        r = self.compare(df, "a", "b", top=(10,))
        self.assertEqual((r["top10_overlap"], r["top10_ties"], r["same_winner"]), (1.0, 20, True))
        df = pd.DataFrame(dict(fcn=["x", "y"], DL_a=[1.0, 2.0], DL_b=[1.5, 2.0]))
        self.assertEqual(self.compare(df, "a", "b", top=(10,))["top10_overlap"], 1.0)
        df = pd.DataFrame(dict(fcn=["x", "y"], DL_a=[np.inf, np.nan], DL_b=[1.0, 2.0]))
        r = self.compare(df, "a", "b", top=(10,))
        self.assertTrue(np.isnan(r["top10_overlap"]))
        self.assertEqual((r["winner_a"], r["winner_b"], r["same_winner"]), (None, "x", False))


if __name__ == "__main__":
    unittest.main()
