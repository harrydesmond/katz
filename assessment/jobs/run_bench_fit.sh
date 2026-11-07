#!/bin/bash
# ESR benchmark, step 2: fit mocks (indices given as arguments) up to complexity BENCH_CMAX
# and apply every prior; complexities already fitted are skipped, so the job can be requeued.
#   for g in $(seq 0 34); do a=$((5*g)); sbatch --requeue -N 1 --ntasks-per-node=16 \
#     --export=ALL,BENCH_CMAX=8 run_bench_fit.sh $a $((a+1)) $((a+2)) $((a+3)) $((a+4)); done
source "${KATZ_DIR:-$HOME/katz}/assessment/jobs/env.sh"  # sbatch runs a copy of this script
for i in "$@"; do
  echo "=== fit $i $(date)"
  $SRUN -n 1 python3 esr_benchmark.py fit $i 1 4 || exit 1
  $SRUN -n $SLURM_NTASKS python3 esr_benchmark.py fit $i 5 $BENCH_CMAX || exit 1
  echo "=== combine $i $(date)"
  $SRUN -n 1 python3 esr_benchmark.py combine $i 1 4 || exit 1
  $SRUN -n $SLURM_NTASKS python3 esr_benchmark.py combine $i 5 $BENCH_CMAX || exit 1
done
echo "=== done $(date)"
