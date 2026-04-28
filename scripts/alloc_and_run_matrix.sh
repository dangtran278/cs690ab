#!/usr/bin/env bash
set -euo pipefail

# Request an interactive GPU allocation and run KVBench matrix inside it.
#
# Usage:
#   bash scripts/alloc_and_run_matrix.sh
#   CONDA_ENV=kvbench bash scripts/alloc_and_run_matrix.sh --output_dir logs_a100
#   bash scripts/alloc_and_run_matrix.sh --probe_max_batch --profile_max_batch --output_dir logs_a100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-kvbench}"

# Match the user's allocation request by default.
PARTITION="${PARTITION:-gpu}"
GPUS="${GPUS:-1}"
NODES="${NODES:-1}"
CPUS="${CPUS:-4}"
CONSTRAINT="${CONSTRAINT:-vram80}"
TIME_LIMIT="${TIME_LIMIT:-03:59:00}"
MEM_MB="${MEM_MB:-50000}"

mkdir -p logs

echo "Requesting allocation:"
echo "salloc -p ${PARTITION} -G ${GPUS} -N ${NODES} -c ${CPUS} --constraint=${CONSTRAINT} --time=${TIME_LIMIT} --mem ${MEM_MB}"

salloc \
  -p "${PARTITION}" \
  -G "${GPUS}" \
  -N "${NODES}" \
  -c "${CPUS}" \
  --constraint="${CONSTRAINT}" \
  --time="${TIME_LIMIT}" \
  --mem "${MEM_MB}" \
  srun bash -lc "
    set -euo pipefail
    cd '${REPO_ROOT}'
    module load conda/latest
    module load cuda/12.4.1
    eval \"\$(conda shell.bash hook)\"
    conda activate '${CONDA_ENV}'
    export PYTHONNOUSERSITE=\${PYTHONNOUSERSITE:-1}
    python scripts/run_matrix.py $*
  "
