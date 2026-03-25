#!/bin/bash
#SBATCH --job-name=kivi_env
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --constraint="vram40"
#SBATCH --time=0:59:00
#SBATCH --qos=short
#SBATCH --output=/scratch3/workspace/danqtran_umass_edu-690ab_kv/logs/kivi_env-%j.out
#SBATCH --error=/scratch3/workspace/danqtran_umass_edu-690ab_kv/logs/kivi_env-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=danqtran@umass.edu

module load conda/latest
module load cuda/12.4.1

cd /scratch3/workspace/danqtran_umass_edu-690ab_kv
conda activate .conda/envs/kivi

cd KIVI
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.4.3+cu124torch2.4-cp310-cp310-linux_x86_64.whl
pip install -e .

cd quant
pip install -e . --no-build-isolation