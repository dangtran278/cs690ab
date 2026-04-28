#!/usr/bin/env bash
set -euo pipefail

# Create a writable conda environment for this repo (avoids shared /scratch3 envs).
#
# Usage (interactive):
#   bash scripts/setup_conda_env_kiet.sh
#
# Optional overrides:
#   ENV_PREFIX=/path/to/env bash scripts/setup_conda_env_kiet.sh
#   PYTHON_VERSION=3.10 bash scripts/setup_conda_env_kiet.sh
#   SKIP_TORCH=1 bash scripts/setup_conda_env_kiet.sh   # if you install torch separately

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_PREFIX="${ENV_PREFIX:-${REPO_ROOT}/.conda/envs/kvbench-kiet}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
SKIP_TORCH="${SKIP_TORCH:-0}"

echo "Repo:        ${REPO_ROOT}"
echo "Env prefix:  ${ENV_PREFIX}"
echo "Python:      ${PYTHON_VERSION}"

if ! command -v module >/dev/null 2>&1; then
  echo "WARN: 'module' not found. If you are on the cluster, run: module load conda/latest (and cuda) first." >&2
fi

if command -v module >/dev/null 2>&1; then
  module load conda/latest || true
  module load cuda/12.4.1 || true
  # If your site provides cuDNN as a module, uncomment:
  # module load cudnn/9 || true
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH after module load." >&2
  exit 1
fi

# Make conda activation work in non-interactive bash scripts.
eval "$(conda shell.bash hook)"

if [[ -d "${ENV_PREFIX}" ]]; then
  echo "Env already exists at: ${ENV_PREFIX}"
  echo "Activate with:"
  echo "  conda activate ${ENV_PREFIX}"
  echo "If you want a clean rebuild, remove that directory and re-run this script."
  exit 0
fi

mkdir -p "$(dirname "${ENV_PREFIX}")"
conda create -y -p "${ENV_PREFIX}" "python=${PYTHON_VERSION}"

conda activate "${ENV_PREFIX}"

# Prevent accidental imports from ~/.local shadowing the env.
export PYTHONNOUSERSITE=1

python -m pip install -U pip setuptools wheel

if [[ "${SKIP_TORCH}" != "1" ]]; then
  python -m pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

python -m pip install "transformers>=4.43" datasets huggingface_hub accelerate safetensors

cd "${REPO_ROOT}"
python -m pip install -e .

echo
echo "Done."
echo "Activate this environment in future shells with:"
echo "  conda activate ${ENV_PREFIX}"
echo "Recommended for stability:"
echo "  export PYTHONNOUSERSITE=1"
