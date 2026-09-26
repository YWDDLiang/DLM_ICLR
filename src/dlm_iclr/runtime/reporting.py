"""Readable terminal results from saved evaluation summaries."""

import math
from pathlib import Path


DIRECT_LABELS = (
    ("struct_valid", "Structural validity", True),
    ("comp_valid", "Compositional validity", True),
    ("valid", "Validity", True),
    ("cov_precision", "Coverage precision", True),
    ("cov_recall", "Coverage recall", True),
    ("wdist_density", "Density W1", False),
    ("wdist_num_elems", "Element-count W1", False),
)


def _format(value, *, percent=False, bounds=None, pending=0):
    suffix = "%" if percent else ""
    if bounds is not None:
        lower, upper = bounds
        if pending or lower != upper:
            detail = f"{pending} unresolved" if pending else "unresolved"
            return f"[{lower:.2f}, {upper:.2f}]{suffix} ({detail})"
        value = lower
    if value is None or not math.isfinite(value):
        detail = f" ({pending} unresolved)" if pending else ""
        return "unavailable" + detail
    return f"{value:.2f}{suffix}"


def format_results(report, *, dataset):
    """Keep unresolved rates as intervals and distinguish distances from percentages."""
    lines = [f"CrystalDLM results | {dataset} | {report['requests']} requests", "", "Direct"]
    direct = report["direct"]
    for key, label, percent in DIRECT_LABELS:
        if key not in direct["metrics"]:
            continue
        value = direct["metrics"][key]
        pending = direct.get("unknown_counts", {}).get(key, 0)
        bounds = direct.get("coverage_bounds_percent", {}).get(key) if value is None else None
        if pending and key in direct.get("count_bounds", {}):
            total = direct["counts"]["requests"]
            bounds = [100 * v / total for v in direct["count_bounds"][key]] if total else None
        lines.append(f"  {label:<24} {_format(value, percent=percent, bounds=bounds, pending=pending)}")
    lines.extend(["", "Physical evaluation"])
    for key in ("SUN", "MSUN", "VUN", "V", "U", "N", "Stable", "MetaStable"):
        metric = report["sun"]["metrics"].get(key)
        if metric is None:
            continue
        value = _format(None, percent=True, bounds=metric["percent_bounds"], pending=metric["pending"])
        lines.append(f"  {key:<24} {value}")
    if report.get("structures"):
        lines.extend(["", f"Saved results: {Path(report['structures']).parent}"])
    return "\n".join(lines)
