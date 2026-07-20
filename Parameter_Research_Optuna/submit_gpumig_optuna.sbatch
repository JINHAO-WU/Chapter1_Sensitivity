#!/bin/bash
#SBATCH --job-name=optuna_gpumig
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

PROJECT_DIR="/home/users/jwu028/study/Parameter_Research"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

source ~/nb_envs/pytorch/bin/activate

PYTHON_BIN="$(readlink -f "$(which python)")"
PYTHON_ENV_DIR="$(dirname "$(dirname "${PYTHON_BIN}")")"
export LD_LIBRARY_PATH="${PYTHON_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
nvidia-smi

python optuna_tpe_search.py

echo "Job finished: $(date)"
