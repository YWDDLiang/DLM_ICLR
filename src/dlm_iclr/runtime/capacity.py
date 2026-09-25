"""Representation limits fixed before imports in each dataset worker process."""
import os

MAX_ATOMS = int(os.environ.get("DLM_MAX_ATOMS", "20"))
MIN_ATOMS = int(os.environ.get("DLM_MIN_ATOMS", "1"))
LENGTH_MAX_BIN = int(os.environ.get("DLM_LENGTH_MAX_BIN", "500"))
DATASET_LABEL = os.environ.get("DLM_DATASET_LABEL", "MP-20")
ATOM_RANGE_TEXT = (
    f"exactly {MAX_ATOMS}" if MIN_ATOMS == MAX_ATOMS else f"{MIN_ATOMS} to {MAX_ATOMS}"
)
