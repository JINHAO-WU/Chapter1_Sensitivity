#!/bin/bash
#SBATCH --job-name=enso_main
#SBATCH --partition=gpumig
#SBATCH --account=orchid
#SBATCH --qos=gpumig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="/home/users/jwu028/study/Study_timeseies"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

source ~/nb_envs/pytorch/bin/activate

PYTHON_BIN="$(readlink -f "$(which python)")"
PYTHON_ENV_DIR="$(dirname "$(dirname "${PYTHON_BIN}")")"
export LD_LIBRARY_PATH="${PYTHON_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "MPLBACKEND: ${MPLBACKEND}"
nvidia-smi

python Sensitivity_CNN_scripts/A_Main_Run.py

echo "Job finished: $(date)"
