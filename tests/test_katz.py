import itertools
import os
import tempfile
import unittest

import esr
import esr.generation.duplicate_checker
import numpy as np

from katz.back_off import BackOff
from katz.esr_prior import compute_logprior, get_logconst
from katz.good_turing import GoodTuring
from katz.kneser_ney import KneserNey, chen_goodman_discounts
from katz.prior import KatzPrior


class TestKatzPrior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        n = 2
        basis_functions = [
            ["a", "x"],
            ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
            ["+", "-", "*", "/", "pow"],
        ]
        # Reference values below are for the Katz model of Bartlett et al. (2023)
        cls.kp_feynman = KatzPrior(
            n,
            basis_functions,
            "data/FeynmanEquations.csv",
            "data/NewFeynman.csv",
            smoothing="katz",
        )
        cls.kp_feynman_no_input = KatzPrior(
            n, basis_functions, None, "data/NewFeynman.csv", smoothing="katz"
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
        )

    def test_logprior_string_equations(self):
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


if __name__ == "__main__":
    unittest.main()
