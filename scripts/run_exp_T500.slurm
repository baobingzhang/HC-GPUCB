#!/bin/bash
#SBATCH --job-name=hcgpucb_T500
#SBATCH --partition=core32
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/exp_T500_%A_%a.out
#SBATCH --error=results/logs/exp_T500_%A_%a.err
#SBATCH --array=0-39

# Adjust the source/conda-activate lines to your environment.
# Example below assumes a conda env named `hcgpucb` with requirements.txt installed.
source ~/.bashrc
conda activate hcgpucb

# WORKDIR is the repo root; default to the current dir if unset.
WORKDIR=${WORKDIR:-$(pwd)}
cd "$WORKDIR"
mkdir -p results/logs results/raw/exp_T500

# Decode (func, method, seed) from SLURM_ARRAY_TASK_ID:
#   4 funcs * 2 methods (gpucb, hc_adaptive) * 5 seeds = 40 array tasks
FUNCS=(branin hartmann3 hartmann6 ackley5)
METHODS=(gpucb hc_adaptive)
SEEDS=(0 1 2 3 4)

N_METHODS=${#METHODS[@]}
N_SEEDS=${#SEEDS[@]}
N_PER_FUNC=$((N_METHODS * N_SEEDS))

ID=$SLURM_ARRAY_TASK_ID
F_IDX=$((ID / N_PER_FUNC))
REM=$((ID % N_PER_FUNC))
M_IDX=$((REM / N_SEEDS))
S_IDX=$((REM % N_SEEDS))

echo "[$(date)] node=$(hostname) func=${FUNCS[$F_IDX]} method=${METHODS[$M_IDX]} seed=${SEEDS[$S_IDX]}"

python experiments/exp1_main.py \
    --func ${FUNCS[$F_IDX]} \
    --method ${METHODS[$M_IDX]} \
    --seed ${SEEDS[$S_IDX]} \
    --T 500 \
    --out results/raw/exp_T500

echo "[$(date)] done"
