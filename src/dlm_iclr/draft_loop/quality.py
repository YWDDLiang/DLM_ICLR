"""Raw-draft objectives and reliable same-composition comparisons.

Predicted CHGNet energy relative to the repository MP hull is a proxy. No label
is called a DFT validation. Missing measurements are never converted into failure.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from statistics import median

BAD = {"generation_failure", "invalid_raw"}
TERMINAL_BAD = BAD | {"invalid_terminal"}
UNKNOWN = {"unknown", "worker_error", "evaluation_error", "timeout"}


def finite(x) -> bool:
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass(frozen=True)
class Measurement:
    record_key: str
    protocol_key: str
    composition: str
    status: str
    verified: bool
    raw_energy: float | None
    force_rms: float | None
    force_max: float | None
    stress_max: float | None
    raw_hull: float | None
    terminal_hull: float | None
    actual_steps: int | None = None

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    def has_raw(self, require_hull=True):
        fields = [self.raw_energy, self.force_rms, self.force_max, self.stress_max]
        if require_hull:
            fields.append(self.raw_hull)
        return all(finite(v) for v in fields)

    def trustworthy(self, require_hull=True):
        return (self.verified and self.status == "verified" and self.has_raw(require_hull)
                and (not require_hull or finite(self.terminal_hull)))


def raw_utility(m: Measurement, *, require_hull=True) -> float | None:
    """A bounded raw-state proxy, NOT SUN and NOT relaxed endpoint reward."""
    if m.status in BAD:
        return -12.0
    if m.status in UNKNOWN or not m.has_raw(require_hull):
        return None
    terms = [0.35 * math.log1p(max(m.force_rms, 0) / 0.1),
             0.15 * math.log1p(max(m.force_max, 0) / 0.2),
             0.15 * math.log1p(max(m.stress_max, 0) / 0.5)]
    if finite(m.raw_hull):
        # No bonus for unphysically extreme negative proxy hull values.
        terms.append(0.35 * math.log1p(max(m.raw_hull, 0) / 0.1))
    return -min(12.0, sum(terms))


def refinability_utility(m: Measurement) -> float | None:
    """Shared C2/feedback target; confirmed F+verification hull quality.

    This changes no Stable/MetaStable definition and assigns no value to unknown
    outcomes. Nonpositive hull values share the maximum reward without bonuses
    for extreme negative model energies.
    """
    return -math.log1p(max(m.terminal_hull, 0.)/.1) if m.trustworthy(True) else None


def raw_nonregression(a, b, settings):
    return (a.raw_energy <= b.raw_energy + settings['energy_slack']
            and a.force_rms <= b.force_rms + settings['force_slack']
            and a.force_max <= b.force_max + settings['force_slack']
            and a.stress_max <= b.stress_max + settings['stress_slack'])


def prefer(candidate: Measurement, current: Measurement, settings: dict, *, require_hull=True):
    """Conservative Pareto-slack rule. Time order and learned scores never label teachers."""
    if candidate.composition != current.composition or candidate.protocol_key != current.protocol_key:
        raise ValueError("Teacher pair changes composition or measurement protocol")
    if not candidate.trustworthy(require_hull):
        return False, "unverified_teacher"
    if require_hull and candidate.terminal_hull > settings["max_teacher_terminal_hull"]:
        return False, "teacher_terminal_hull_too_high"
    if current.status in BAD:
        return (candidate.force_max <= settings["anchor_force"]
                and candidate.stress_max <= settings["anchor_stress"]), "valid_teacher_for_failed_draft"
    if not current.trustworthy(require_hull):
        return False, "unknown_or_unverified_pair"
    a, b = candidate, current
    if require_hull and b.terminal_hull <= 0 < a.terminal_hull:
        return False, "strict_stability_regression"
    regress = (
        not raw_nonregression(a,b,settings)
        or (require_hull and a.terminal_hull > b.terminal_hull + settings["terminal_hull_slack"])
    )
    gain = (b.raw_energy - a.raw_energy >= settings["energy_gain"]
            or b.force_rms - a.force_rms >= settings["force_gain"]
            or b.stress_max - a.stress_max >= settings["stress_gain"])
    return bool(not regress and gain), "raw_physical_improvement" if not regress and gain else "no_clear_improvement"


def is_anchor(m: Measurement, settings: dict, *, require_hull=True) -> bool:
    return (m.trustworthy(require_hull)
            and m.force_max <= settings["anchor_force"]
            and m.stress_max <= settings["anchor_stress"]
            and (not require_hull or m.terminal_hull <= settings["max_teacher_terminal_hull"]))


def teacher_rank(m: Measurement, *, require_hull=True):
    """Within already qualified candidates, preserve strict stability first."""
    strict = bool(require_hull and finite(m.terminal_hull) and m.terminal_hull <= 0)
    return strict, raw_utility(m,require_hull=require_hull)


def publication_gate(before: dict[str, Measurement], after: dict[str, Measurement], settings: dict,
                     *, require_hull=True) -> dict:
    if set(before) != set(after) or not before:
        raise ValueError("Development gate must preserve every requested source")
    common = [k for k in before if raw_utility(before[k], require_hull=require_hull) is not None
              and raw_utility(after[k], require_hull=require_hull) is not None]
    reasons = []
    if len(common) / len(before) < settings["min_common_fraction"]:
        reasons.append("too_many_unknown_development_labels")
    failed_b = sum(m.status in BAD for m in before.values())
    failed_a = sum(m.status in BAD for m in after.values())
    if failed_a > failed_b + settings["max_invalid_increase"]:
        reasons.append("raw_validity_regression")
    delta = (sum(raw_utility(after[k], require_hull=require_hull)
                 - raw_utility(before[k], require_hull=require_hull) for k in common) / len(common)
             if common else None)
    if delta is None or delta < settings["min_mean_gain"]:
        reasons.append("no_raw_quality_gain")
    stable_common = [k for k in common if before[k].has_raw(require_hull) and after[k].has_raw(require_hull)]
    for field, ratio in [("force_rms", settings["max_force_ratio"]),
                         ("stress_max", settings["max_stress_ratio"])]:
        if stable_common:
            b = median(getattr(before[k], field) for k in stable_common)
            a = median(getattr(after[k], field) for k in stable_common)
            if a > max(b * ratio, b + 1e-8):
                reasons.append(field + "_regression")
    hull_keys = [k for k in before if before[k].trustworthy(True) and after[k].trustworthy(True)]
    hull_delta = None
    if require_hull:
        # A measured terminal failure is a known negative, not missing data.
        # Numerical hull means still use only actual verified energies.
        known_terminal = lambda m: m.status in TERMINAL_BAD or m.trustworthy(True)
        terminal_common = [k for k in before if known_terminal(before[k]) and known_terminal(after[k])]
        if len(terminal_common)/len(before) < settings["min_common_fraction"]:
            reasons.append("insufficient_known_terminal_coverage")
        if sum(m.status in TERMINAL_BAD for m in after.values()) > sum(m.status in TERMINAL_BAD for m in before.values()):
            reasons.append("terminal_validity_regression")
        if len(hull_keys)/len(before) < settings.get("min_verified_hull_fraction",0.25):
            reasons.append("insufficient_verified_hull_pairs")
        if hull_keys:
            hull_delta = sum(max(after[k].terminal_hull, 0) - max(before[k].terminal_hull, 0)
                             for k in hull_keys) / len(hull_keys)
            if hull_delta > settings["max_hull_mean_increase"]:
                reasons.append("terminal_hull_regression")
    return {"accepted": not reasons, "reasons": reasons, "requests": len(before),
            "common_known": len(common), "mean_raw_gain": delta, "terminal_hull_delta": hull_delta,
            "verified_hull_pairs":len(hull_keys),
            "known_terminal_pairs":len(terminal_common) if require_hull else None,
            "failed_before": failed_b, "failed_after": failed_a,
            "test_set_used": False, "comparison": "same_requests_pure_draft_no_F_no_verifier"}
