import ast
import itertools
import os
import re
import sys
import warnings

import numpy as np
from mpi4py import MPI

from katz.prior import KatzPrior

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_indices(n):
    """
    Find the indices of an array of length n which should be considered
    by this rank. For output data_start, data_end, the rank considers the
    entries array[data_start:data_end] where len(array) = n.

    Args:
        :n (int): The length of the array we wish to split among ranks.

    Returns:
        :data_start (int): The first index of the array to be considered by the rank
        :data_end (int): The final index (+1) to be considered by the rank.

    """

    nLs = int(np.ceil(n / float(size)))  # Number of lines per file for given thread

    while nLs * (size - 1) > n:
        if rank == 0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank == 0:
        print("Total number of functions: ", n, flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank * nLs
    data_end = (rank + 1) * nLs

    if rank == size - 1:
        data_end = n

    return data_start, data_end


def get_functions(comp, dirname):
    """
    Load all functions from a given directory at a certain complexity
    and divide these among the ranks.

    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the function lists

    Returns:
        :fcn_list (list[str]): The list of functions considered by this rank
        :data_start (int): The first index of the array to be considered by the rank
        :data_end (int): The final index (+1) to be considered by the rank

    """

    recursion_limits = {8: 2000, 9: 2500, 10: 3000}
    sys.setrecursionlimit(recursion_limits.get(comp, sys.getrecursionlimit()))

    fname = dirname + f"/compl_{comp}/all_equations_{comp}.txt"

    if rank == 0:
        print("Number of cores:", size, flush=True)

    with open(fname, "r") as f:
        fcn_list = f.readlines()

    data_start, data_end = get_indices(len(fcn_list))

    return fcn_list[data_start:data_end], data_start, data_end


def get_trees(comp, dirname):
    """
    Load the tree files for ESR for a given complexity and convert to
    lists of strings. These are split among ranks.

    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the ESR functions

    Returns:
        :tree_list (list): The list of lists giving trees to be considered by this rank

    """

    fname = dirname + f"/compl_{comp}/trees_{comp}.txt"
    with open(fname, "r") as f:
        tree_list = f.read().splitlines()
    data_start, data_end = get_indices(len(tree_list))
    tree_list = tree_list[data_start:data_end]
    for i in range(len(tree_list)):
        # Remove np.str_(...) wrappers
        tree = re.sub(r"np\.str_\((['\"])(.*?)\1\)", r"\1\2\1", tree_list[i])
        # Add commas between adjacent quoted strings
        tree = re.sub(r"(['\"])\s+(['\"])", r"\1, \2", tree)
        tree_list[i] = ast.literal_eval(tree)

    return tree_list


def get_logconst(comp, dirname, overwrite=False):
    """
    Determine the sum(log(c_i)) for constants c_i appearning in all equations
    and save results to file.

    Args:
        :comp (int): The complexity of function to consider
        :dirname (str): The directory containing the function lists
        :overwrite (bool): Whether to overwrite the constants file if it already exists

    Returns:
        :None

    """

    outname = dirname + f"/compl_{comp}/logconst_{comp}.txt"
    if os.path.isfile(outname) and (not overwrite):
        return

    tree_list = get_trees(comp, dirname)
    logconst = [None] * len(tree_list)
    for i, tree in enumerate(tree_list):
        n = np.array([int(tt) for tt in tree if tt.lstrip("-").isdigit()])  # Integers
        n[n == 0] = 1  # So we have log(1) for 0 instead of log(0)
        logconst[i] = np.sum(np.log(np.abs(n)))

    logconst = comm.gather(logconst, root=0)
    if rank == 0:
        logconst = np.array(list(itertools.chain(*logconst)))
        np.savetxt(outname, logconst)

    return


def _prior_config(n, basis_functions, prior_kwargs):
    """Complete (JSON-serialisable) configuration of the KatzPrior built by compute_logprior"""
    import inspect

    from katz.prior import DEFAULT_KN_DISCOUNT

    defaults = {
        k: v.default
        for k, v in inspect.signature(KatzPrior.__init__).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    cfg = {
        k: prior_kwargs.get(k, defaults[k])
        for k in ("smoothing", "code_constants", "root_context", "right_backoff")
    }
    skw = dict(prior_kwargs.get("smoothing_kwargs") or {})
    if cfg["smoothing"] == "kn":
        skw.setdefault("discounts", DEFAULT_KN_DISCOUNT)
    cfg["smoothing_kwargs"] = {
        k: np.asarray(v).tolist() if not isinstance(v, str) else v for k, v in skw.items()
    }
    cfg["n"] = int(n)
    cfg["basis_functions"] = [list(b) for b in basis_functions]
    return cfg


def default_label(prior_kwargs):
    """Output-file prefix for a prior configuration: the smoothing method plus the
    non-default representation options, e.g. 'kn', 'katz', 'kn_root', 'kn_root_const'"""
    label = prior_kwargs.get("smoothing", "kn")
    if prior_kwargs.get("root_context", False):
        label += "_root"
    if prior_kwargs.get("code_constants", False):
        label += "_const"
    if prior_kwargs.get("right_backoff", "sibling") == "parent":
        label += "_parent"
    return label


def compute_logprior(
    comp,
    n,
    basis_functions,
    dirname,
    in_eqfile,
    out_eqfile,
    overwrite=False,
    input_delimiter=",",
    use_tree=False,
    prior_kwargs=None,
    label=None,
):
    """
    Compute the log of the function prior for all functions at a give complexity
    given a corpus of equations. Saves results to two files: {label}_logprior_{n}_{comp}.txt
    contains just -log(prior) values, and {label}_codelen_{n}_{comp}.txt also contains
    the sum(log(c_i)) terms for constants c_i which appear in the equations. A manifest
    {label}_prior_{n}_{comp}.json records the prior configuration, a hash of the corpus,
    the package version and the number of functions that could not be scored (NaN in the
    output; a function with zero prior probability has -log(prior) = inf). Existing output
    is reused only if its manifest matches the requested configuration.

    Args:
        :comp (int): The complexity of function to consider
        :n (int): The length of the n-tuples to use for back-off model
        :basis_functions (list): List of basis functions to consider. Entries 0, 1 and 2 are lists of nullary, unary, and binary operators, respectively.
        :dirname (str): The directory containing the function lists
        :in_eqfile (str): Name of file containing the equations to use as corpus
        :out_eqfile (str): Name of file to output standardised corpus equations to
        :overwrite (bool): Whether to overwrite files if they already exist
        :input_delimiter (str): The delimiter used in the input csv file
        :use_tree (bool, default=False): Whether to compute the prior using the tree representation from file (True) or using the string representation (False) and convert this to a tree automatically.
        :prior_kwargs (dict, default=None): Extra keyword arguments passed to KatzPrior (e.g. {'smoothing': 'katz'} for the model of Bartlett et al. 2023; the default is Kneser-Ney)
        :label (str, default=None): Prefix of the output files, so that priors from different models are not overwritten. If None, default_label(prior_kwargs) is used: the smoothing method ('kn' or 'katz') followed by '_root', '_const' and '_parent' for non-default representation options.

    Returns:
        :None

    """
    import hashlib
    import json

    from katz.prior import __version__
    from katz.process_equations import standardise_file

    prior_kwargs = {} if prior_kwargs is None else dict(prior_kwargs)
    if label is None:
        label = default_label(prior_kwargs)
    suffix = f"_{n}_{comp}" + ("_tree" if use_tree else "")
    outname_code = dirname + f"/compl_{comp}/{label}_codelen{suffix}.txt"
    outname_prior = dirname + f"/compl_{comp}/{label}_logprior{suffix}.txt"
    outname_manifest = dirname + f"/compl_{comp}/{label}_prior{suffix}.json"

    # The corpus is read (and the standardised file written) by rank 0 only
    if "equations" in prior_kwargs:
        all_eq = list(prior_kwargs.pop("equations"))
    elif rank == 0:
        all_eq, _ = standardise_file(in_eqfile, out_eqfile, input_delimiter)
    else:
        all_eq = None
    all_eq = comm.bcast(all_eq, root=0)

    manifest = {
        "config": _prior_config(n, basis_functions, prior_kwargs),
        "corpus_sha256": hashlib.sha256("\n".join(all_eq).encode()).hexdigest(),
        "corpus_size": len(all_eq),
        "comp": int(comp),
        "use_tree": bool(use_tree),
        "katz_version": __version__,
    }
    reuse = False
    if rank == 0 and not overwrite:
        if os.path.isfile(outname_code) and os.path.isfile(outname_prior):
            old = None
            if os.path.isfile(outname_manifest):
                with open(outname_manifest) as f:
                    old = json.load(f)
            keys = ("config", "corpus_sha256", "comp", "use_tree")
            if old is not None and all(old.get(k) == manifest[k] for k in keys):
                reuse = True
            else:
                print(
                    f"Recomputing {outname_prior}: existing output has "
                    + ("no manifest" if old is None else "a different configuration"),
                    flush=True,
                )
    if comm.bcast(reuse, root=0):
        return

    if use_tree:
        fcn_list_proc = get_trees(comp, dirname)
    else:
        fcn_list_proc, _, _ = get_functions(comp, dirname)

    kp = KatzPrior(n, basis_functions, equations=all_eq, **prior_kwargs)
    logprior_proc = np.empty(len(fcn_list_proc))
    nbad = np.zeros(2, dtype=int)  # zoo (complex infinity), failed to parse/score

    for i, eq in enumerate(fcn_list_proc):
        if "zoo" in eq:
            logprior_proc[i] = np.nan
            nbad[0] += 1
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if use_tree:
                        logprior_proc[i] = kp.logprior(eq)
                    else:
                        logprior_proc[i] = kp.logprior(eq.strip())
            except Exception as e:  # noqa: BLE001
                if use_tree:
                    print("BAD EQ:", eq, e)
                else:
                    print("BAD EQ:", eq.strip(), e)
                logprior_proc[i] = np.nan
                nbad[1] += 1

    logprior = comm.gather(logprior_proc, root=0)
    nbad = comm.reduce(nbad, op=MPI.SUM, root=0)
    if rank == 0:
        logprior = np.array(list(itertools.chain(*logprior)))
        np.savetxt(outname_prior, -logprior)
        logconst = np.loadtxt(dirname + f"/compl_{comp}/logconst_{comp}.txt")
        np.savetxt(
            outname_code, -logprior + logconst
        )  # will use as replacement to aifeyn term, so need to sum these
        manifest.update(
            n_functions=int(len(logprior)),
            n_zoo=int(nbad[0]),
            n_failed=int(nbad[1]),
            n_zero_prior=int(np.sum(np.isneginf(logprior))),
        )
        with open(outname_manifest, "w") as f:
            json.dump(manifest, f, indent=1)
        if nbad[1] > 0:
            print(
                f"WARNING: {nbad[1]} of {len(logprior)} functions at complexity {comp} "
                "could not be scored (NaN in the output)",
                flush=True,
            )
    comm.Barrier()

    return
