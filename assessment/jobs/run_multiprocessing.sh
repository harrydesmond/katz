#!/bin/bash
# Single-node multiprocessing job, e.g.
#   sbatch -N 1 --ntasks-per-node=20 run_multiprocessing.sh synthetic.py run 10
#   sbatch -N 1 --ntasks-per-node=20 run_multiprocessing.sh cv_grouped.py
# ESR imports mpi4py, so the Slurm/PMI variables are cleared and MPI_Init runs as a singleton.
source "${KATZ_DIR:-$HOME/katz}/assessment/jobs/env.sh"  # sbatch runs a copy of this script
export NPROC=$SLURM_CPUS_ON_NODE
for v in $(env | grep -o -E '^(SLURM|PMI|PMIX|OMPI)_[A-Za-z0-9_]*'); do unset $v; done
python3 "$@"
