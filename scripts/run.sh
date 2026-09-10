#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export TOKENIZERS_PARALLELISM=false
exec "${PYTHON:-python}" -m dlm_iclr "$@"
