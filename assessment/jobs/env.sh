#!/bin/bash
# Environment for the Slurm job scripts in this directory. Edit for your cluster:
#   KATZ_DIR  repository root;  ESR_DIR  ESR clone (with the 'new_osc_maths' basis for the
#   benchmark);  PYTHON_ROOT  Python installation;  SRUN  MPI launcher (srun --mpi=pmix
#   works with mpi4py built on OpenMPI 4).
export KATZ_DIR=${KATZ_DIR:-$HOME/katz}
export ESR_DIR=${ESR_DIR:-$KATZ_DIR/ESR}
export PYTHON_ROOT=${PYTHON_ROOT:-/usr/local/shared/python/3.11.4}
export PATH=$PYTHON_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_ROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ESR_DIR:$PYTHONPATH
export OMP_NUM_THREADS=1
SRUN=${SRUN:-"srun -u --mpi=pmix"}
cd $KATZ_DIR/assessment
