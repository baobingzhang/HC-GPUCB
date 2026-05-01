#!/bin/bash
#SBATCH --job-name=hcg_D3
#SBATCH --partition=core96
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=${WORKDIR}/results/logs/D3_%A_%a.out
#SBATCH --error=${WORKDIR}/results/logs/D3_%A_%a.err
#SBATCH --array=0-119%16

# Phase D3 — HC-GPUCB with rank-1 inference integrated into Phase 2.
# 3 funcs (branin, hartmann3, hartmann6) * 4 budgets {500, 1000, 2000, 3000}
#   * 2 methods (hc_fixed_rank1, hc_adaptive_rank1) * 5 seeds = 120 cells
# Throttle 16 to share CPU politely with geta.

WORKDIR=${WORKDIR:-$(pwd)}
source ~/.bashrc
conda activate hcgpucb

WORKDIR=${WORKDIR}
cd $WORKDIR
mkdir -p results/logs

FUNCS=(branin hartmann3 hartmann6)
TS=(500 1000 2000 3000)
METHODS=(hc_fixed_rank1 hc_adaptive_rank1)
SEEDS=(0 1 2 3 4)

ID=$SLURM_ARRAY_TASK_ID
N_T=${#TS[@]}
N_M=${#METHODS[@]}
N_S=${#SEEDS[@]}
N_PER_F=$((N_T * N_M * N_S))
N_PER_T=$((N_M * N_S))

F=$((ID / N_PER_F))
REM=$((ID % N_PER_F))
TI=$((REM / N_PER_T))
REM2=$((REM % N_PER_T))
MI=$((REM2 / N_S))
S=$((REM2 % N_S))

OUT_DIR=results/raw/exp_D3_hc_rank1/T${TS[$TI]}
mkdir -p $OUT_DIR

echo "[D3] FUNC=${FUNCS[$F]} T=${TS[$TI]} METHOD=${METHODS[$MI]} SEED=${SEEDS[$S]} node=$(hostname)"
python experiments/exp1_main.py \
    --func ${FUNCS[$F]} --method ${METHODS[$MI]} --seed ${SEEDS[$S]} \
    --T ${TS[$TI]} --out $OUT_DIR
