Katz
----

:Katz: Function prior for symbolic regression with an n-gram language model
:Authors: Deaglan J. Bartlett
:Homepage: https://github.com/DeaglanBartlett/katz 
:Documentation: https://katz.readthedocs.io

.. image:: https://readthedocs.org/projects/katz/badge/?version=latest
  :target: https://katz.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/cs.LG-arXiv%32304.06333-B31B1B.svg
  :target: https://arxiv.org/abs/2304.06333

.. image:: https://github.com/DeaglanBartlett/katz/actions/workflows/build.yml/badge.svg
  :target: https://github.com/DeaglanBartlett/katz/actions/workflows/build.yml
  :alt: Build Status

.. image:: https://img.shields.io/codecov/c/github/DeaglanBartlett/katz
  :target: https://app.codecov.io/gh/DeaglanBartlett/katz
  :alt: Coverage

.. important::

   Since version 0.2.0 the default model is Kneser-Ney smoothing (``smoothing='kn'``),
   which predicts held-out physics equations better than the Katz back-off model
   (see ``assessment/REPORT.md``). **All results published before this change**, including
   `Bartlett, Desmond & Ferreira (2023) <https://arxiv.org/abs/2304.06333>`_ and the
   applications of the prior, **used the Katz model**. To reproduce them, pass
   ``smoothing='katz'`` to ``KatzPrior`` (or ``prior_kwargs={'smoothing': 'katz'}`` to
   ``katz.esr_prior.compute_logprior``).

About
=====

Symbolic Regression algorithms attempt to find analytic expressions which accurately
fit a given data set. In many cases, one wants these to interpretable and thus one
tries to find the simplest possible expression which can accurately fit the data. 
Even if an expression is simple, if one is seeking a physical law, one may discard
an expression because it does not look like one you would expect to see in this context.
To automate this process, we produce a prior on functions based on an n-gram language
model of the operators in a function's tree: by studying a corpus of previously used
equations in a given field, the model returns a probability for each function based on the
order of operators appearing in the function. The conditional probabilities are smoothed
with interpolated Kneser-Ney by default, or with the Katz back-off model of the original
paper.

Installation
=============

This repository depends on the `ESR Package <https://github.com/DeaglanBartlett/ESR>`_ to produce tree representations of equations.
To download and install the relevant code and dependencies in a new virtual environment, run

.. code:: bash

	python3 -m venv katz_env
	source katz_env/bin/activate
	git clone git@github.com:DeaglanBartlett/ESR.git
	pip install -e ESR
	git clone git@github.com:DeaglanBartlett/katz.git
	pip install -e katz


Example
========

To generate a 2-gram prior from the `AI Feynman function set <https://space.mit.edu/home/tegmark/aifeynman.html>`_ 
(`arXiv:1905.11481 <https://arxiv.org/abs/1905.11481>`_) and evaluate this on some trial functions, run the following

.. code-block:: python

	from katz.prior import KatzPrior
	
	n = 2
	basis_functions = [["a", "x"],
                ["sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"],
                ["+", "-", "*", "/", "pow"]]
    
    	kp = KatzPrior(n, basis_functions, 'data/FeynmanEquations.csv', 'data/NewFeynman.csv')
    	
	for eq in ['x0**2', 'sin(x0) + sin(x1)', 'sin(sin(x0+x1))']:
            p = kp.logprior(eq)
            print(eq, p)


Smoothing and representation options
------------------------------------

By default (``smoothing='kn'``) the conditional probabilities are estimated with
interpolated Kneser-Ney smoothing, with a single absolute discount ``D = 0.8`` at all
orders. This assigns a non-zero probability to every operator of the basis set in every
context and is less sensitive to the choice of ``n`` than the Katz model. The discounts can
instead be tuned by cross-validation on the corpus with ``tune_discounts()``, set with
``smoothing_kwargs={'discounts': D}``, or estimated with the Chen-Goodman formulae
(``smoothing_kwargs={'discounts': 'mkn'}``, not recommended for corpora of fewer than
~100 equations). ``smoothing='katz'`` gives the Katz back-off model of Bartlett et al. (2023)
and reproduces its priors exactly, apart from functions consisting of a single leaf (``x0``,
``a0``), whose root event was previously counted twice.

Two optional changes to the tree representation are available. ``root_context=True``
conditions the root and the nodes near it on an explicit start-of-tree context; this gives
the largest improvement in held-out tests but makes single-leaf functions considerably less
probable, since no corpus equation is a bare leaf. ``code_constants=True`` codes constants
and parameters as ``a`` rather than as variables (by default they are treated like
variables, as physical constants are in the corpus, so e.g. ``pow(x0, a0)`` and
``pow(a0, x0)`` have the same prior).

.. code-block:: python

	kp = KatzPrior(3, basis_functions, 'data/PhysicsEquations.csv', 'data/NewPhysics.csv',
	               input_delimiter=';', root_context=True)
	kp.tune_discounts()
	print(kp.logprior('sin(x0) + sin(x1)'))

``FunctionPrior`` is an alias of ``KatzPrior``. An assessment of the models is given in
``assessment/REPORT.md``.


Licence and Citation
====================

Users are  required to cite the katz `Paper <https://arxiv.org/abs/2304.06333>`_

.. code:: bibtex

  @ARTICLE{2023arXiv2304.06333,
       author = {{Bartlett}, D.~J. and {Desmond}, H. and {Ferreira}, P.~G.},
        title = "{Priors for symbolic regression}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning},
         year = 2023,
        month = apr,
          eid = {arXiv:2304.06333},
        pages = {arXiv:2304.06333},
  archivePrefix = {arXiv},
       eprint = {2304.06333},
  primaryClass = {cs.LG},
          url = {https://arxiv.org/abs/2304.06333},
  }


Since this depends on the ESR package, 
users are required to cite the ESR `Paper <https://arxiv.org/abs/2211.11461>`_
for which the following bibtex can be used

.. code:: bibtex

  @ARTICLE{2022arXiv2211.11461,
       author = {{Bartlett}, D.~J. and {Desmond}, H. and {Ferreira}, P.~G.},
        title = "{Exhaustive Symbolic Regression}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2022,
        month = nov,
          eid = {arXiv:2211.11461},
        pages = {arXiv:2211.11461},
  archivePrefix = {arXiv},
       eprint = {2211.11461},
  primaryClass = {astro-ph.CO},
	  url = {https://arxiv.org/abs/2211.11461},
  }

The software is available on the MIT licence:

Copyright 2023 Deaglan J. Bartlett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contributors
============

Below is a list of contributors to this repository. 

`Deaglan Bartlett <https://github.com/DeaglanBartlett>`_ (CNRS & Sorbonne Université, Institut d’Astrophysique de Paris)

Documentation
=============

The documentation for this project can be found
`at this link <https://katz.readthedocs.io/>`_

Acknowledgements
================
DJB is supported by the Simons Collaboration on `Learning the Universe <https://www.learning-the-universe.org/>`_.

