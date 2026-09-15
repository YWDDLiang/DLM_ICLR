#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python}" -m dlm_iclr run "$@"
