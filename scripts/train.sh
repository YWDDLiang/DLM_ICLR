#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
if [[ "${TRAIN_GPUS:-1}" -gt 1 ]]; then
  exec "${PYTHON:-python}" -m torch.distributed.run --standalone --nproc_per_node="$TRAIN_GPUS" -m dlm_iclr train "$@"
fi
exec "${PYTHON:-python}" -m dlm_iclr train "$@"
