#!/bin/bash
#SBATCH --job-name=kvbench-matrix
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=03:59:00
#SBATCH --chdir=/work/pi_sunghoonlee_umass_edu/Kiet/danqtran_umass_edu-690ab_kv
#SBATCH --output=logs_a100/kvbench-matrix-%j.out
#SBATCH --error=logs_a100/kvbench-matrix-%j.err

set -euo pipefail

mkdir -p logs_a100

python scripts/run_matrix.py --probe_max_batch --profile_max_batch --output_dir logs_a100
