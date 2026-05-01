#!/bin/bash
#SBATCH --job-name=hcg_T2k3k
#SBATCH --partition=core96
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=${WORKDIR}/results/logs/T2k3k_%A_%a.out
#SBATCH --error=${WORKDIR}/results/logs/T2k3k_%A_%a.err
#SBATCH --array=0-89

# Phase B2 (revised): T=2000 + T=3000 large-budget runs.
#   Vanilla GP-UCB at T>=2000 is infeasible (>=21h, exceeds SLURM limits).
#   T=5000 with rank1 is borderline (>4h benchmark timeout). T=3000 is the
#   sweet spot: well above vanilla's last feasible budget (T=1000 ~ 80 min)
#   while reliably finishing in <2h on rank1 / HC.
#
# 3 funcs * 2 budgets * 3 methods (rank1, hc_fixed, hc_adaptive) * 5 seeds = 90 runs.

WORKDIR=${WORKDIR:-$(pwd)}
source ~/.bashrc
conda activate hcgpucb

WORKDIR=${WORKDIR}
cd $WORKDIR
mkdir -p results/logs

FUNCS=(branin hartmann3 hartmann6)
T_LIST=(2000 3000)
METHODS=(rank1_gpucb hc_fixed hc_adaptive)
SEEDS=(0 1 2 3 4)

ID=$SLURM_ARRAY_TASK_ID
N_F=${#FUNCS[@]}
N_T=${#T_LIST[@]}
N_M=${#METHODS[@]}
N_S=${#SEEDS[@]}
N_PER_F=$((N_T * N_M * N_S))
N_PER_T=$((N_M * N_S))

FI=$((ID / N_PER_F))
REM=$((ID % N_PER_F))
TI=$((REM / N_PER_T))
REM2=$((REM % N_PER_T))
MI=$((REM2 / N_S))
SI=$((REM2 % N_S))

FUNC=${FUNCS[$FI]}
T=${T_LIST[$TI]}
METHOD=${METHODS[$MI]}
SEED=${SEEDS[$SI]}

OUT_DIR=results/raw/exp_T2k3k/T${T}
mkdir -p $OUT_DIR

echo "[T2k3k] FUNC=$FUNC METHOD=$METHOD SEED=$SEED T=$T  node=$(hostname)"
python experiments/exp1_main.py \
    --func $FUNC --method $METHOD --seed $SEED \
    --T $T --out $OUT_DIR
