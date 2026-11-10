#!/bin/bash
# ESR benchmark: apply further priors (BENCH_PRIORS) to mocks that are already fitted, without
# refitting. Mode 'priors' writes their prior files (MPI over the job's tasks); mode 'mocks'
# applies them to mocks FIRST..LAST and ranks the truth (one core per mock: ESR's combine_DL
# makes every rank read the whole codelength file, so more ranks per mock do not help).
#   export BENCH_CMAX=7 BENCH_PRIORS=katz3,kn3   # in the submitting shell: --export would
#                                                # split the comma-separated list
#   sbatch -N 1 --ntasks-per-node=8 --export=ALL run_bench_rescore.sh priors
#   sbatch -N 1 --ntasks-per-node=8 --dependency=afterok:<priors job> --export=ALL \
#       run_bench_rescore.sh mocks 0 23        # or: run_bench_rescore.sh list 3 17 42
# Mocks that already have their table are skipped, so a job can simply be resubmitted.
# Per-mock tables: bench_work/rescore/results_<i>_<priors>.csv (merge them afterwards).
source "${KATZ_DIR:-$HOME/katz}/assessment/jobs/env.sh"  # sbatch runs a copy of this script
: "${BENCH_PRIORS:?set BENCH_PRIORS}" "${BENCH_CMAX:?set BENCH_CMAX}"
echo "=== $1 $BENCH_PRIORS c<=$BENCH_CMAX on $(hostname), $SLURM_NTASKS tasks $(date)"
if [ "$1" = priors ]; then
  $SRUN -n $SLURM_NTASKS python3 esr_benchmark.py priors || exit 1
  # the current code must reproduce the existing prior files of the published comparison
  $SRUN -n $SLURM_NTASKS python3 esr_benchmark.py checkpriors katz3r,kn3r 5 || exit 1
else
  R=${BENCH_DIR:-$KATZ_DIR/assessment/bench_work}/rescore
  L=$KATZ_DIR/logs/rescore
  tag=$(echo $BENCH_PRIORS | tr , -)
  mkdir -p $R $L
  if [ "$1" = list ]; then shift; idx="$*"; else idx=$(seq $2 $3); fi
  printf "%s\n" $idx | xargs -P $SLURM_NTASKS -I{} sh -c "
    [ -s $R/results_{}_$tag.csv ] && { echo skip {}; exit 0; }
    $SRUN -n 1 -c 1 --exact python3 esr_benchmark.py combine {} 1 $BENCH_CMAX > $L/combine-{}-$tag.log 2>&1 &&
    $SRUN -n 1 -c 1 --exact python3 esr_benchmark.py collect {} $R/results_{}_$tag.csv > $L/collect-{}-$tag.log 2>&1 &&
    echo done {} || echo FAILED mock {}"
fi
echo "=== finished $(date)"
