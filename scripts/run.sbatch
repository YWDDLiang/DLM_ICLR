#!/usr/bin/env bash
#SBATCH --job-name=dlm-iclr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=dlm-iclr-%j.log
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -m dlm_iclr "$@"
