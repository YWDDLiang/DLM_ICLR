from copy import deepcopy

import pytest

from dlm_iclr.draft_loop.candidate_diagnostics import (
    summarize_candidate_opportunities, check_base_physical_protocol, measurement_artifact,
)


def outcome(cid, force, hull, source, status):
    def measurement(key, value, terminal, state):
        return {'record_key': key, 'protocol_key': 'fixed_physics', 'composition': 'O2',
                'status': state, 'verified': state == 'verified', 'raw_energy': -2.,
                'force_rms': value, 'force_max': 2*value, 'stress_max': value,
                'raw_hull': .2, 'terminal_hull': terminal, 'actual_steps': 10}
    return {'parent_source_id': source, 'candidate_id': cid, 'raw_record_key': cid,
            'F_record_key': 'F:'+cid, 'raw_measurement': measurement(cid, force, .05, 'verified'),
            'F_measurement': measurement('F:'+cid, .1, hull, status)}


def pool():
    result = []
    for source, hulls in [('a', [.2, -.01, .05, None]), ('b', [.04, .03, .02, .01])]:
        for i, hull in enumerate(hulls):
            r = outcome(f'{source}:{i}', 10-i, hull, source, 'unknown' if hull is None else 'verified')
            r.update(candidate_index=i, refiner_noise_seed=17)
            result.append(r)
    return result


def test_source_denominator_oracle_rescue_and_random_bounds():
    report = summarize_candidate_opportunities(pool())
    assert report['source_requests'] == 2 and report['candidate_attempts'] == 8
    stable = report['metrics']['Stable']
    assert stable['first'] == {'passed': 0, 'unknown': 0, 'requests': 2}
    assert stable['oracle_same_pool']['passed'] == 1
    assert stable['confirmed_rescue_sources'] == 1
    assert stable['random_expected_pass_count_bounds'] == [.25, .5]
    assert report['metrics']['MetaStable']['random_expected_pass_count_bounds'] == [1.5, 1.75]
    assert report['learned_verifier_gain_established'] is False
    assert report['is_SUN_report'] is False


def test_unknown_pools_do_not_become_failed_oracle_sources():
    rows = pool()[:4]
    for row in rows:
        row['F_measurement'].update(terminal_hull=None, verified=False, status='unknown')
    report = summarize_candidate_opportunities(rows)
    assert report['metrics']['Stable']['oracle_same_pool']['unknown'] == 1
    assert report['metrics']['Stable']['confirmed_rescue_sources'] == 0
    assert report['sources_with_joint_raw_F_improvement'] == 0


def test_different_chemistry_references_are_valid_but_within_plan_mismatch_is_not():
    rows = pool()
    for row in rows[4:]:
        for kind in ('raw_measurement', 'F_measurement'):
            row[kind].update(protocol_key='other_chemistry_bound_reference', composition='Li2')
    assert summarize_candidate_opportunities(rows)['source_requests'] == 2
    rows[4]['F_measurement']['protocol_key'] = 'changed_within_this_plan'
    with pytest.raises(ValueError, match='within the same Plan'):
        summarize_candidate_opportunities(rows)


def test_global_evaluator_identity_is_checked_separately_and_unknowns_are_kept(tmp_path):
    from dlm_iclr.draft_loop.common import write_json
    artifact = {'identity': {'physics': 'checkpoint_and_thresholds'}, 'rows': [{'status': 'unknown'}]}
    write_json(tmp_path/'unresolved.json', artifact)
    assert measurement_artifact(tmp_path) == artifact
    assert check_base_physical_protocol(artifact, artifact) == 'checkpoint_and_thresholds'
    with pytest.raises(ValueError, match='base physical'):
        check_base_physical_protocol(artifact, {'identity': {'physics': 'changed_thresholds'}})
    recovered = {'identity': artifact['identity'], 'rows': [{'status': 'verified'}]}
    write_json(tmp_path/'result.json', recovered)
    assert measurement_artifact(tmp_path) == recovered


@pytest.mark.parametrize('change', ['drop_slot', 'wrong_seed', 'duplicate', 'wrong_key'])
def test_pair_identity_and_all_attempts_are_required(change):
    rows = deepcopy(pool())
    if change == 'drop_slot': rows.pop()
    elif change == 'wrong_seed': rows[0]['refiner_noise_seed'] = 18
    elif change == 'duplicate': rows[-1]['candidate_id'] = rows[0]['candidate_id']
    else: rows[0]['F_record_key'] = 'unrelated'
    with pytest.raises(ValueError): summarize_candidate_opportunities(rows)
