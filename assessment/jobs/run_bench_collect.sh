#!/bin/bash
# ESR benchmark, step 3: rank the truth under every prior for all finished mocks
#   sbatch -N 1 --ntasks-per-node=1 --export=ALL,BENCH_CMAX=8 run_bench_collect.sh
source "${KATZ_DIR:-$HOME/katz}/assessment/jobs/env.sh"  # sbatch runs a copy of this script
for v in $(env | grep -o -E '^(SLURM|PMI|PMIX|OMPI)_[A-Za-z0-9_]*'); do unset $v; done
python3 esr_benchmark.py collect || exit 1
cp bench_work/benchmark_results.csv bench_work/benchmark_results_c${BENCH_CMAX:-8}.csv
