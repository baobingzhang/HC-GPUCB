#!/bin/bash
#SBATCH --job-name=hcg_T1000
#SBATCH --partition=core32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=${WORKDIR}/results/logs/speedTpin_%A_%a.out
#SBATCH --error=${WORKDIR}/results/logs/speedTpin_%A_%a.err
#SBATCH --array=60-89

# Phase B5: speedup-vs-T re-run with node pinning + rank1.
#   1 func (hartmann3) * 3 T values (200, 500, 1000) * 3 methods * 10 seeds = 90
# --exclusive: don't share the node with other jobs
# --nodelist: restrict to a contiguous block of homogeneous nodes

WORKDIR=${WORKDIR:-$(pwd)}
source ~/.bashrc
conda activate hcgpucb

WORKDIR=${WORKDIR}
cd $WORKDIR
mkdir -p results/logs results/raw/exp_speedup_pinned

FUNC=hartmann3
TS=(200 500 1000)
METHODS=(gpucb rank1_gpucb hc_adaptive)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

ID=$SLURM_ARRAY_TASK_ID
N_T=${#TS[@]}
N_M=${#METHODS[@]}
N_S=${#SEEDS[@]}
N_PER_T=$((N_M * N_S))
TI=$((ID / N_PER_T))
REM=$((ID % N_PER_T))
M=$((REM / N_S))
S=$((REM % N_S))

OUT_DIR=results/raw/exp_speedup_pinned/T${TS[$TI]}
mkdir -p $OUT_DIR

echo "[speedT-pin] node=$(hostname) T=${TS[$TI]} method=${METHODS[$M]} seed=${SEEDS[$S]}"
python experiments/exp1_main.py \
    --func $FUNC --method ${METHODS[$M]} --seed ${SEEDS[$S]} \
    --T ${TS[$TI]} --out $OUT_DIR
