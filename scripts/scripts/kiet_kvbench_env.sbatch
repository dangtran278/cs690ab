#!/bin/bash
#SBATCH --job-name=kvbench_env_kiet
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=vram40
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=1:59:00
#SBATCH --chdir=/work/pi_sunghoonlee_umass_edu/Kiet/danqtran_umass_edu-690ab_kv
#SBATCH --output=logs/kiet_kvbench_env-%j.out
#SBATCH --error=logs/kiet_kvbench_env-%j.err

set -euo pipefail

REPO_ROOT="${KVBENCH_REPO_ROOT:-/work/pi_sunghoonlee_umass_edu/Kiet/danqtran_umass_edu-690ab_kv}"
cd "${REPO_ROOT}"

module load conda/latest
module load cuda/12.4.1

mkdir -p logs

bash scripts/setup_conda_env_kiet.sh
