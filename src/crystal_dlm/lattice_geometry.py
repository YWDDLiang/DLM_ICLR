"""Retained crystal DLM implementation; see docs/method.md for the public workflow."""

from __future__ import annotations
import math


def lattice_angle_rad(alpha: int, beta: int, gamma: int) -> float:
    """Return the lattice volume angle factor for degree-valued angles."""

    alpha_rad = math.radians(float(alpha))
    beta_rad = math.radians(float(beta))
    gamma_rad = math.radians(float(gamma))
    cos_a = math.cos(alpha_rad)
    cos_b = math.cos(beta_rad)
    cos_g = math.cos(gamma_rad)
    return 1.0 + 2.0 * cos_a * cos_b * cos_g - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g
