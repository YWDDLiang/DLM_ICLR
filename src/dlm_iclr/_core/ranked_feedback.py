"""Retained crystal DLM implementation; see docs/reproduction.md for the public workflow."""

import math


COUNTS = (1, 2, 4, 8)


def fixed_slots_match(target, reference):
    return len(target) == len(reference) and target[0] == reference[0] and target[7::4] == reference[7::4]


def validate_action_target(current, target, positions):
    if not fixed_slots_match(target, current):
        raise ValueError("off-support fixed element/count target")
    allowed = set(positions)
    if any(a != b and i not in allowed for i, (a, b) in enumerate(zip(current, target))):
        raise ValueError("target edits positions outside its conditioned action")


def joint_stop_status(forces, stresses_GPa, *, fmax=0.1, stress_tolerance=0.5):
    force = max((math.sqrt(sum(float(v) ** 2 for v in row)) for row in forces), default=float("inf"))
    stress = max((abs(float(v)) for row in stresses_GPa for v in row), default=float("inf"))
    if not math.isfinite(force) or not math.isfinite(stress):
        raise ValueError("nonfinite physical convergence observation")
    return dict(
        force_max_eV_A=force,
        stress_max_GPa=stress,
        physical_converged=force <= fmax and stress <= stress_tolerance,
    )
