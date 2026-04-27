#!/bin/bash
#SBATCH --job-name=kvbench-logs-kiet
#SBATCH --partition=gpu
#SBATCH --gpus=1
# Request a higher-VRAM GPU class (Unity-style feature constraint).
# Override at submit time if your site uses different feature names, e.g.:
#   sbatch --constraint=a100 scripts/submit_kvbench_logs_kiet.sbatch
#SBATCH --constraint=vram40
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
# Always start in the repo so relative paths like logs/ resolve consistently.
#SBATCH --chdir=/work/pi_sunghoonlee_umass_edu/Kiet/danqtran_umass_edu-690ab_kv
#SBATCH --output=logs/kvbench-logs-kiet-%j.out
#SBATCH --error=logs/kvbench-logs-kiet-%j.err

set -euo pipefail

# Hard-anchor to the repo root (Slurm copies batch scripts; don't infer paths from $0).
REPO_ROOT="${KVBENCH_REPO_ROOT:-/work/pi_sunghoonlee_umass_edu/Kiet/danqtran_umass_edu-690ab_kv}"
cd "${REPO_ROOT}"

# Writable env created by scripts/setup_conda_env_kiet.sh
CONDA_ENV="${CONDA_ENV:-/work/pi_sunghoonlee_umass_edu/Kiet/danqtran_umass_edu-690ab_kv/.conda/envs/kvbench-kiet}"

# Override when submitting, e.g.:
#   sbatch --export=ALL,EXTRA_ARGS='--run_ppl --methods fp16' scripts/submit_kvbench_logs_kiet.sbatch
#
# Default avoids 32K passkey contexts (often OOM on 40–48GB cards); extend if you have headroom.
DEFAULT_EXTRA_ARGS="\
--run_ppl --run_passkey --run_memory --run_latency --run_throughput --run_scaling \
--methods fp16 kivi4 kivi2 kvquant_nuq4_1p kvquant_nuq3_1p \
--ppl_tokens 4096 --ppl_prefill_tokens 512 \
--passkey_contexts 2048 4096 8192 16384 \
--latency_context_tokens 4096 --latency_decode_tokens 64 --latency_warmup_tokens 8 \
--throughput_context_tokens 4096 --throughput_batch_sizes 1 2 4 8 \
--chunked_prefill_tokens 0 \
--output_dir logs"

EXTRA_ARGS="${EXTRA_ARGS:-$DEFAULT_EXTRA_ARGS}"

module load conda/latest
module load cuda/12.4.1

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

mkdir -p logs

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

python scripts/run_matrix_new.py ${EXTRA_ARGS}
