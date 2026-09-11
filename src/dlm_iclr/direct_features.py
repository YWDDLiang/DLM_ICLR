"""Optional CrysLLMGen Direct fingerprints and bounded-memory coverage."""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
from pathlib import Path
import numpy as np
from .io import file_hash, fingerprint, read_json, write_json

_FEATURIZERS = None


def _initialize():
    global _FEATURIZERS
    from matminer.featurizers.composition.composite import ElementProperty
    from matminer.featurizers.site.fingerprint import CrystalNNFingerprint

    _FEATURIZERS = (
        ElementProperty.from_preset("magpie", impute_nan=False),
        CrystalNNFingerprint.from_preset("ops"),
    )


def _featurize(payload):
    from pymatgen.core import Structure
    from ._vendor.crysllmgen.direct_constants import CompScalerMeans, CompScalerStds

    if _FEATURIZERS is None:
        _initialize()
    composition, sites = _FEATURIZERS
    try:
        structure = Structure.from_dict(payload)
        raw = np.asarray(composition.featurize(structure.composition), dtype=float)
        scaled = (raw - np.asarray(CompScalerMeans)) / np.asarray(CompScalerStds)
        # Preserve the upstream StandardScaler's replace_nan_token=0 rule.
        scaled = np.where(np.isnan(scaled), 0.0, scaled)
        structural = np.asarray([sites.featurize(structure, i) for i in range(len(structure))]).mean(0)
        if not np.isfinite(scaled).all() or not np.isfinite(structural).all():
            raise ValueError("nonfinite_fingerprint")
        return {"comp_fp": scaled.tolist(), "struct_fp": structural.tolist(), "error": None}
    except Exception as error:
        return {"comp_fp": None, "struct_fp": None, "error": f"{type(error).__name__}: {error}"}


def compute_features(structures, *, cache, workers=1):
    """Cache exact geometry and featurizer identity; retry failed fingerprints."""
    if workers < 1:
        raise ValueError("Direct fingerprint workers must be positive")
    _initialize()  # Missing optional dependencies fail before any row is scored.
    identity = {
        "schema": "crysllmgen_direct_features_v1",
        "packages": {key: importlib.metadata.version(key) for key in ("matminer", "pymatgen", "numpy")},
        "implementation": file_hash(__file__),
        "scaler": file_hash(Path(__file__).parent / "_vendor/crysllmgen/direct_constants.py"),
        "composition": "ElementProperty.magpie,impute_nan=False;frozen_scaler;NaN->0",
        "structure": "CrystalNNFingerprint.ops,mean_over_sites",
    }
    cache = Path(cache) / fingerprint(identity)
    keys, pending, values = [], {}, {}
    hits = 0
    for structure in structures:
        if structure is None:
            keys.append(None)
            continue
        payload = structure.as_dict()
        key = fingerprint(payload)
        keys.append(key)
        path = cache / key[:2] / (key + ".json")
        if key in values or key in pending:
            hits += 1
        elif path.exists():
            values[key] = read_json(path)
            hits += 1
        else:
            pending[key] = payload

    def retain(results):
        for key, value in zip(pending, results, strict=True):
            values[key] = value
            if value["error"] is None:
                write_json(cache / key[:2] / (key + ".json"), value)

    if workers == 1:
        retain(map(_featurize, pending.values()))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize) as pool:
            retain(pool.map(_featurize, pending.values()))
    return (
        [
            values[key] if key else {"comp_fp": None, "struct_fp": None, "error": "not_reconstructed"}
            for key in keys
        ],
        {"identity": identity, "cache_hits": hits, "computed": len(pending)},
    )


def coverage(predicted, reference, *, struc_cutoff, comp_cutoff, requested, block_size=256):
    """Upstream independent structural/composition minima, in distance blocks."""
    from scipy.spatial.distance import cdist

    if requested < 1 or not reference or block_size < 1:
        raise ValueError("Coverage requires a positive denominator, reference set and block size")
    valid = [row for row in predicted if row["error"] is None]
    if not valid:
        return {"cov_recall": 0.0, "cov_precision": 0.0}
    pred_s = np.asarray([row["struct_fp"] for row in valid])
    pred_c = np.asarray([row["comp_fp"] for row in valid])
    ref_s = np.asarray([row["struct_fp"] for row in reference])
    ref_c = np.asarray([row["comp_fp"] for row in reference])
    recall_s, recall_c = np.full(len(reference), np.inf), np.full(len(reference), np.inf)
    precision_s, precision_c = np.full(len(valid), np.inf), np.full(len(valid), np.inf)
    for i in range(0, len(valid), block_size):
        left = slice(i, i + block_size)
        for j in range(0, len(reference), block_size):
            right = slice(j, j + block_size)
            distances_s = cdist(pred_s[left], ref_s[right])
            distances_c = cdist(pred_c[left], ref_c[right])
            recall_s[right] = np.minimum(recall_s[right], distances_s.min(axis=0))
            recall_c[right] = np.minimum(recall_c[right], distances_c.min(axis=0))
            precision_s[left] = np.minimum(precision_s[left], distances_s.min(axis=1))
            precision_c[left] = np.minimum(precision_c[left], distances_c.min(axis=1))
    return {
        "cov_recall": float(np.mean((recall_s <= struc_cutoff) & (recall_c <= comp_cutoff))),
        "cov_precision": float(
            np.sum((precision_s <= struc_cutoff) & (precision_c <= comp_cutoff)) / requested
        ),
    }
