"""Source-level opportunity audit of measured TRAIN candidate pools.

Oracle choices here read outcomes and are diagnostics only. They cannot be used
as an online verifier, a final-panel selection rule, or evidence of learned
gain. Stable opportunity does not include Novel/Unique and is not SUN.
"""
from collections import Counter, defaultdict
import math

from .quality import Measurement, TERMINAL_BAD, finite, raw_utility


def terminal_pass(measurement, threshold):
    if measurement.status in TERMINAL_BAD:
        return False
    if measurement.verified and measurement.status == 'verified' and finite(measurement.terminal_hull):
        return measurement.terminal_hull <= threshold
    return None


def _bounds(statuses):
    return {'passed': sum(x is True for x in statuses),
            'unknown': sum(x is None for x in statuses), 'requests': len(statuses)}


def summarize_candidate_opportunities(outcomes, *, candidates_per_source=4,
                                      stable_threshold=0., metastable_threshold=.1):
    """Compare first, same-pool random expectation, and label-using oracle bounds.

    Outcomes must contain every attempt, including failed/unknown candidates.
    Pair counts are reported separately and never treated as independent Plans.
    Thresholds are passed from the existing evaluation configuration.
    """
    groups = defaultdict(list)
    ids = set()
    for row in outcomes:
        if row['candidate_id'] in ids:
            raise ValueError('Candidate outcome duplicated')
        ids.add(row['candidate_id']); groups[row['parent_source_id']].append(row)
    if not groups or candidates_per_source < 2:
        raise ValueError('Complete multi-candidate source groups required')
    per_source = []
    pairs = Counter()
    for source, rows in groups.items():
        rows = sorted(rows, key=lambda r: r['candidate_index'])
        if [r['candidate_index'] for r in rows] != list(range(candidates_per_source)):
            raise ValueError('Missing/duplicate candidate slots; do not drop failures')
        if len({r['refiner_noise_seed'] for r in rows}) != 1:
            raise ValueError('Within-Plan refinement noise differs')
        raw = [Measurement.from_dict(r['raw_measurement']) for r in rows]
        refined = [Measurement.from_dict(r['F_measurement']) for r in rows]
        if len({m.composition for m in raw+refined}) != 1:
            raise ValueError('Candidate pool changed composition')
        bound_protocols = set()
        for row, a, b in zip(rows, raw, refined, strict=True):
            if row['raw_record_key'] != a.record_key or row['F_record_key'] != b.record_key:
                raise ValueError('Candidate measurement binding differs')
            bound_protocols.update((a.protocol_key, b.protocol_key))
        # These keys bind a hull reference for this chemistry, not just the
        # evaluator. Comparing all compositions against one key is incorrect.
        if len(bound_protocols) != 1:
            raise ValueError('Mixed physical protocol/reference within the same Plan')
        rq = [raw_utility(m) for m in raw]
        fq = [-math.log1p(max(m.terminal_hull, 0.)/.1) if m.trustworthy(True) else None for m in refined]
        pareto, refinability_pairs = [], []
        for i in range(candidates_per_source):
            for j in range(i+1, candidates_per_source):
                if fq[i] is not None and fq[j] is not None and fq[i] != fq[j]:
                    good, bad = (j, i) if fq[j] > fq[i] else (i, j)
                    refinability_pairs.append({'better': good, 'worse': bad, 'F_gain': abs(fq[j]-fq[i])})
                if any(x is None for x in (rq[i], rq[j], fq[i], fq[j])):
                    pairs['unknown_raw_or_refined_quality'] += 1
                    continue
                dr, df = rq[j]-rq[i], fq[j]-fq[i]
                pairs['jointly_known'] += 1
                if abs(dr) >= .05 and abs(df) >= .05:
                    pairs['agree' if dr*df > 0 else 'raw_refinement_tradeoff'] += 1
                if ((dr >= 0 and df >= 0) or (dr <= 0 and df <= 0)) and max(abs(dr), abs(df)) >= .05:
                    good, bad = (j, i) if dr >= 0 and df >= 0 else (i, j)
                    if refined[good].terminal_hull <= metastable_threshold:
                        pareto.append({'better': good, 'worse': bad})
        status = {name: [terminal_pass(m, threshold) for m in refined] for name, threshold in
                  [('Stable', stable_threshold), ('MetaStable', metastable_threshold)]}
        entry = {'source_id': source, 'candidate_ids': [r['candidate_id'] for r in rows],
                 'raw_quality': rq, 'F_verified_hull_quality': fq,
                 'physical_Pareto_pairs': pareto, 'refinability_preference_pairs': refinability_pairs}
        for name, values in status.items():
            oracle = True if True in values else None if None in values else False
            entry[name] = {'candidate_status': values, 'first': values[0], 'oracle': oracle,
                           'confirmed_rescue_over_first': values[0] is False and oracle is True,
                           'random_expected_success_lower': sum(v is True for v in values)/candidates_per_source,
                           'random_expected_success_upper': sum(v is not False for v in values)/candidates_per_source}
        per_source.append(entry)
    summary = {}
    for name in ('Stable', 'MetaStable'):
        summary[name] = {
            'first': _bounds([r[name]['first'] for r in per_source]),
            'oracle_same_pool': _bounds([r[name]['oracle'] for r in per_source]),
            'confirmed_rescue_sources': sum(r[name]['confirmed_rescue_over_first'] for r in per_source),
            'random_expected_pass_count_bounds': [
                sum(r[name]['random_expected_success_lower'] for r in per_source),
                sum(r[name]['random_expected_success_upper'] for r in per_source)],
        }
    return {
        'schema': 'TRAIN_candidate_opportunity_audit_v1',
        'source_requests': len(groups), 'candidate_attempts': len(outcomes),
        'candidates_per_source': candidates_per_source, 'metrics': summary,
        'sources_with_joint_raw_F_improvement': sum(bool(r['physical_Pareto_pairs']) for r in per_source),
        'sources_with_refinability_preference': sum(bool(r['refinability_preference_pairs']) for r in per_source),
        'refinability_preference_pair_count': sum(len(r['refinability_preference_pairs']) for r in per_source),
        'within_source_pairs': dict(pairs), 'per_source': per_source,
        'uses_true_outcomes_for_diagnosis_only': True, 'learned_verifier_gain_established': False,
        'includes_Novel_Unique': False, 'is_SUN_report': False,
        'evaluation_thresholds': {'Stable': stable_threshold, 'MetaStable': metastable_threshold},
        'new_physics_calls': 0,
    }


def measurement_artifact(folder):
    """Read producer metadata without dropping unresolved, genuinely unknown rows."""
    from pathlib import Path
    from .common import read_json
    folder = Path(folder)
    # The producer keeps an old unresolved artifact for provenance after a
    # successful recovery. Its completed result is then authoritative.
    for path in (folder/'result.json', folder/'unresolved.json'):
        if path.exists():
            return read_json(path)
    raise ValueError('Missing measurement artifact')


def check_base_physical_protocol(*artifacts):
    """Check evaluator identity globally, then chemistry-bound keys per Plan."""
    identities = [a['identity']['physics'] for a in artifacts]
    if not identities or any(not x for x in identities) or len(set(identities)) != 1:
        raise ValueError('Different base physical evaluator protocols')
    return identities[0]
