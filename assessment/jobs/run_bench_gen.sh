#!/bin/bash
# ESR benchmark, step 1: function library (complexity 1-8), mock data and prior files, e.g.
#   sbatch -N 1 --ntasks-per-node=28 run_bench_gen.sh
# Complexities <= 5 run on one rank: ESR's index splitting fails with more ranks than functions.
source "${KATZ_DIR:-$HOME/katz}/assessment/jobs/env.sh"  # sbatch runs a copy of this script
export BENCH_CMAX=${BENCH_CMAX:-8}
for c in $(seq 1 $BENCH_CMAX); do
  NP=$SLURM_NTASKS; [ $c -le 5 ] && NP=1
  echo "=== gen $c $(date) ranks $NP"; $SRUN -n $NP python3 esr_benchmark.py gen $c || exit 1
done
echo "=== mocks $(date)"; $SRUN -n 1 python3 esr_benchmark.py mocks || exit 1
echo "=== priors $(date)"; $SRUN -n $SLURM_NTASKS python3 esr_benchmark.py priors || exit 1
echo "=== done $(date)"
