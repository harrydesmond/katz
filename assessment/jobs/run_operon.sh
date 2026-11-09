#!/bin/bash
# Operon bloat experiment: run all lines of a task list (operon_bloat.py tasks pilot|full)
# in parallel on one node; finished runs are skipped, so the job can be requeued, e.g.
#   sbatch --requeue -N 1 --ntasks-per-node=16 run_operon.sh tasks_pilot.txt
# Needs a Python with pyoperon >= 0.6 (OPERON_PYTHON), sympy, scipy and pandas.
source "${KATZ_DIR:-$HOME/katz}/assessment/jobs/env.sh"
PY=${OPERON_PYTHON:-python3}
NT=$(wc -l < "$1")
echo "=== $NT runs on $SLURM_CPUS_ON_NODE cores $(date)"
# sh -c keeps xargs going when one run is killed (xargs stops on a signal-killed child)
seq 0 $((NT - 1)) | xargs -P "$SLURM_CPUS_ON_NODE" -I{} \
  sh -c "$PY operon_bloat.py batch $1 {} || echo FAILED task {}"
echo "=== done $(date)"; $PY operon_bloat.py collect
