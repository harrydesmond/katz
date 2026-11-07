import re

import numpy as np

from katz.back_off import BackOff
from katz.kneser_ney import KneserNey
from katz.process_equations import SymbolCoder, standardise_file


# Discount used by the Kneser-Ney model unless smoothing_kwargs or tune_discounts() set one.
# Cross-validation on the Physics corpus gives optima D = 0.6-0.93 (depending on n and on
# the left/right model), with the held-out codelength within ~0.07 nats per equation of its
# minimum for D within +-0.15 of the optimum (assessment/REPORT.md).
DEFAULT_KN_DISCOUNT = 0.8


def _count_variables(all_eq):
    """Number of variables x0, x1, ... needed to parse a list of equation strings"""
    idx = [int(m) for eq in all_eq for m in re.findall(r"(?<![A-Za-z0-9_])x(\d+)", eq)]
    return max(idx) + 1 if len(idx) > 0 else 1


class KatzPrior:

    def __init__(
        self,
        n,
        basis_functions,
        in_eqfile=None,
        out_eqfile=None,
        input_delimiter=",",
        smoothing="kn",
        smoothing_kwargs=None,
        code_constants=False,
        root_context=False,
        right_backoff="sibling",
        equations=None,
    ):
        """Class to evaluate the probability of a function based on an n-gram language model
        (interpolated Kneser-Ney by default, or the Katz back-off model of Bartlett et al. 2023)

        Note: before version 0.2.0 the default was smoothing='katz'. Published results that
        use this package (Bartlett, Desmond & Ferreira 2023 and its applications) used the
        Katz model; pass smoothing='katz' to reproduce them.

        Args:
            :n (int): The length of the n-tuples to consider
            :basis_functions (list): List of basis functions to consider. Entries 0, 1 and 2 are lists of nullary, unary, and binary operators, respectively.
            :in_eqfile (str): Name of file containing the equations to study. If None, then equations read from out_eqfile
            :out_eqfile (str): Name of file to output the standardised equations to
            :input_delimiter (str): The delimiter used in the input csv file
            :smoothing (str, default='kn'): Language model used for the conditional probabilities: 'kn' (interpolated Kneser-Ney) or 'katz' (Katz back-off with Good-Turing discounting, as in Bartlett et al. 2023)
            :smoothing_kwargs (dict): Extra keyword arguments for the language model. For 'kn': discounts (default DEFAULT_KN_DISCOUNT = 0.8 at all orders; 'mkn' for the Chen-Goodman modified Kneser-Ney estimates) and discount_scale
            :code_constants (bool, default=False): Code numerical constants and parameters as "a" rather than as variables (see SymbolCoder)
            :root_context (bool, default=False): Condition nodes near the root on an explicit start-of-tree context (see SymbolCoder)
            :right_backoff (str, default='sibling'): Order in which the context of a right child is shortened when backing off. 'sibling' (original behaviour) drops the most distant ancestor first, so the last context kept is the left sibling. 'parent' drops the left sibling first, so the last context kept is the parent.
            :equations (list of str): Equations (with variables x0, x1, ...) to use as the corpus instead of reading a file. If given, in_eqfile and out_eqfile are ignored.

        Returns:
            KatzPrior: Prior model to find prior of a function given a previous set of equations

        """

        self.n = n
        if smoothing not in ("katz", "kn"):
            raise ValueError(f"Unknown smoothing method: {smoothing}")
        if right_backoff not in ("sibling", "parent"):
            raise ValueError(f"Unknown right_backoff option: {right_backoff}")
        self.smoothing = smoothing
        self.smoothing_kwargs = {} if smoothing_kwargs is None else dict(smoothing_kwargs)
        if smoothing == "kn":
            self.smoothing_kwargs.setdefault("discounts", DEFAULT_KN_DISCOUNT)
        self.side_kwargs = [{}, {}]  # left/right overrides, e.g. set by tune_discounts
        self.right_backoff = right_backoff

        if equations is not None:
            self.all_eq = list(equations)
            self.maxvar = _count_variables(self.all_eq)
        else:
            self.all_eq, self.maxvar = standardise_file(
                in_eqfile, out_eqfile, input_delimiter
            )
        self.basis_functions = [
            list(
                set(basis_functions[0] + ["a"] + [f"x{i}" for i in range(self.maxvar)])
            ),  # type0
            basis_functions[1],  # type1
            basis_functions[2],
        ]  # type2
        self.coder = SymbolCoder(
            self.basis_functions,
            code_constants=code_constants,
            root_context=root_context,
        )
        self.none_code = self.coder.code["None"]

        # Words which may be predicted (used to normalise the Kneser-Ney model)
        leaves = ["a", "x"] if code_constants else ["x"]
        ops = [
            o
            for o in basis_functions[1] + basis_functions[2]
            if o in self.coder.code
        ]
        self.vocab_left = [self.coder.code[s] for s in leaves + ops]
        self.vocab_right = self.vocab_left + [self.coder.code["y"]]

        self.eq_tuples = [
            self.coder.process_all_equations(n + 1, [eq], self.maxvar)
            for eq in self.all_eq
        ]
        self.fit(self.eq_tuples)

    def _right_tuple(self, tt):
        """Right-child prediction event (context..., word) from a full sibling tuple"""
        if self.right_backoff == "sibling":
            return tt
        ctx = tt[-(self.n + 1) : -1]
        return (ctx[-1],) + ctx[:-1] + (tt[-1],)

    def split_events(self, t):
        """
        Split the n-tuples of one equation into the prediction events of the left
        (first child / root) and right (second child) models

        Args:
            :t (list): The n-tuples of one equation, as returned by SymbolCoder.process_all_equations

        Returns:
            :left (list): Tuples whose final element is a root or first child and whose other elements are its ancestors
            :right (list): Tuples whose final element is a second child and whose other elements are its context
        """
        left = [t[0]] + [tt[:-1] for tt in t[1:]]
        right = [self._right_tuple(tt) for tt in t[1:] if tt[-1] != self.none_code]
        return left, right

    def fit(self, eq_tuples):
        """
        (Re)build the left and right language models from the n-tuples of a set of equations

        Args:
            :eq_tuples (list): List (one entry per equation) of lists of n-tuples

        Returns:
            :None
        """
        data_left = []
        data_right = []
        for t in eq_tuples:
            left, right = self.split_events(t)
            data_left += left
            data_right += right
        if self.smoothing == "katz":
            self.backoff_left = BackOff(data_left)
            self.backoff_right = BackOff(data_right)
        else:
            order = self.n + 1
            self.backoff_left = KneserNey(
                data_left,
                order=order,
                vocab=self.vocab_left,
                **{**self.smoothing_kwargs, **self.side_kwargs[0]},
            )
            self.backoff_right = KneserNey(
                [tt[-order:] for tt in data_right],
                order=order,
                vocab=self.vocab_right,
                **{**self.smoothing_kwargs, **self.side_kwargs[1]},
            )

    def tune_discounts(self, method="single", k=10, seed=0, grid=None):
        """
        Choose the Kneser-Ney discounts of the left and right models by k-fold
        cross-validation over the corpus equations, minimising the held-out negative
        log-probability, then refit on the full corpus with the chosen discounts.

        Args:
            :method (str, default='single'): 'single' (one discount D used at every order) or 'scale' (factor multiplying the Chen-Goodman estimates)
            :k (int, default=10): Number of folds
            :seed (int, default=0): Seed of the random assignment of equations to folds
            :grid (array, default=None): Values to try. Defaults to 0.05-1.0 ('single') or 0.1-2.0 ('scale')

        Returns:
            :result (dict): The chosen left and right settings and the cross-validated NLL (nats per equation) on the grid
        """
        if self.smoothing != "kn":
            raise ValueError("tune_discounts requires smoothing='kn'")
        if method not in ("single", "scale"):
            raise ValueError(f"Unknown tuning method: {method}")
        if grid is None:
            grid = (
                np.linspace(0.05, 1.0, 39)
                if method == "single"
                else np.linspace(0.1, 2.0, 39)
            )
        grid = np.asarray(grid, dtype=float)
        N = len(self.eq_tuples)
        k = min(k, N)
        folds = np.array_split(np.random.default_rng(seed).permutation(N), k)
        nll = np.zeros((2, len(grid)))
        for f in folds:
            test = set(f.tolist())
            self.fit([self.eq_tuples[i] for i in range(N) if i not in test])
            models = (self.backoff_left, self.backoff_right)
            events = ([], [])
            for i in f:
                left, right = self.split_events(self.eq_tuples[i])
                events[0].extend((tt[-1], tt[:-1]) for tt in left)
                events[1].extend((tt[-1], tt[-(self.n + 1) : -1]) for tt in right)
            for j, g in enumerate(grid):
                for side in (0, 1):
                    if method == "single":
                        models[side].set_discounts(float(g))
                    else:
                        models[side].set_discounts("mkn", float(g))
                    nll[side, j] -= sum(
                        np.log(models[side].prob(w, h)) for w, h in events[side]
                    )
        best = [float(grid[np.argmin(nll[side])]) for side in (0, 1)]
        if method == "single":
            self.side_kwargs = [{"discounts": b} for b in best]
        else:
            self.side_kwargs = [
                {"discounts": "mkn", "discount_scale": b} for b in best
            ]
        self.fit(self.eq_tuples)
        return dict(
            left=self.side_kwargs[0],
            right=self.side_kwargs[1],
            grid=grid,
            cv_nll=nll / N,
        )

    def event_probs(self, t):
        """
        Conditional probabilities of every node of an equation

        Args:
            :t (list): The n-tuples of one equation, as returned by SymbolCoder.process_all_equations

        Returns:
            :pleft (np.ndarray): Probabilities of the root and first children
            :pright (np.ndarray): Probabilities of the second children given the first
        """
        tleft, tright = self.split_events(t)
        pleft = np.array([self.backoff_left.get_pbo(tt[-1], tt[:-1]) for tt in tleft])
        pright = np.array(
            [
                self.backoff_right.get_pbo(tt[-1], tt[-(self.n + 1) : -1])
                for tt in tright
            ]
        )
        return pleft, pright

    def logprior_tuples(self, t):
        """
        Natural logarithm of the prior of an equation already converted to n-tuples

        Args:
            :t (list): The n-tuples of one equation, as returned by SymbolCoder.process_all_equations

        Returns:
            :p (float): The natural logarithm of the prior
        """
        pleft, pright = self.event_probs(t)
        with np.errstate(divide="ignore"):
            return np.sum(np.log(pleft)) + np.sum(np.log(pright))

    def logprior(self, eq):
        """
        Compute the natural logarithm of the prior of a given equation

        Args:
            :eq (str): The equation to find the prior probability of

        Returns:
            :p (float): The natural logarithm of the prior of the supplied equation
        """

        t = self.coder.process_all_equations(self.n + 1, [eq], self.maxvar)
        return self.logprior_tuples(t)


# Name that does not refer to a particular smoothing method
FunctionPrior = KatzPrior
