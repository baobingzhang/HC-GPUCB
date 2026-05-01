#!/bin/bash
#SBATCH --job-name=hcgpucb_exp1
#SBATCH --partition=core32
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=${WORKDIR}/results/logs/exp1_%A_%a.out
#SBATCH --error=${WORKDIR}/results/logs/exp1_%A_%a.err
#SBATCH --array=0-79

# 4 funcs * 4 methods * 5 seeds = 80 array tasks
WORKDIR=${WORKDIR:-$(pwd)}
source ~/.bashrc
conda activate hcgpucb

WORKDIR=${WORKDIR}
cd $WORKDIR
mkdir -p results/logs results/raw/exp1_T80

echo "[$(date)] node=$(hostname) array_id=$SLURM_ARRAY_TASK_ID"

python experiments/exp1_main.py --array-id "$SLURM_ARRAY_TASK_ID" --T 80 --n-seeds 5 --out results/raw/exp1_T80

echo "[$(date)] done"
