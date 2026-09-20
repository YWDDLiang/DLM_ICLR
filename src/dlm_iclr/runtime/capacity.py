"""Dataset representation capacity, fixed for each configured worker process.

The launcher sets these before model/data imports. MP-20 remains the default,
so previously saved checkpoints retain their original interpretation.
"""
import os
MAX_ATOMS=int(os.environ.get('DLM_MAX_ATOMS','20'))
MIN_ATOMS=int(os.environ.get('DLM_MIN_ATOMS','1'))
DATASET_LABEL=os.environ.get('DLM_DATASET_LABEL','MP-20')
ATOM_RANGE_TEXT=f'{MIN_ATOMS} to {MAX_ATOMS}' if MIN_ATOMS!=MAX_ATOMS else f'exactly {MAX_ATOMS}'

LENGTH_MAX_BIN=int(os.environ.get('DLM_LENGTH_MAX_BIN','500'))
