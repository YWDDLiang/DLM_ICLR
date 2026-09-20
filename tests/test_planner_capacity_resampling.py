from dlm_iclr.planner.sampling import _collect_plan_rows


def plan(formula):
    return (f'formula: {formula}\nanion: oxide\ncharge: neutral_plausible\n'
            'lattice: cubic\nspacegroup: sg_195_230\nvolume: volpa_010_014\nend: plan')


def test_rejected_draws_are_replaced_and_accounted_without_dropping_other_failures():
    outputs = iter([plan('Si21'), plan('Si2O3'), plan('Si22'),
                    'formula: ???\nend: plan', plan('Si20')])
    widths = []

    def draw(width):
        widths.append(width)
        return [next(outputs) for _ in range(width)]

    rows, rejected, attempted = _collect_plan_rows(
        draw, requests=3, batch_size=2, seed=17, usage_role='evaluation',
        max_atoms=20, resample_over_capacity=True,
    )
    assert widths == [2, 2, 1]
    assert attempted == len(rows) + len(rejected) == 5
    assert [x['sampling_attempt'] for x in rows] == [1, 3, 4]
    assert [x['original_ordinal'] for x in rows] == [0, 1, 2]
    assert [x['atom_count'] for x in rejected] == [21, 22]
    assert rows[1]['body_eligible'] is False
    assert len({x['source_id'] for x in rows + rejected}) == 5


def test_dataset_specific_limit_and_legacy_switch():
    for capacity, expected in [(5, [6]), (52, [])]:
        outputs = iter([plan('Si6'), plan('Si5')])
        rows, rejected, attempted = _collect_plan_rows(
            lambda width: [next(outputs) for _ in range(width)],
            requests=1, batch_size=1, seed=17, usage_role='evaluation',
            max_atoms=capacity, resample_over_capacity=True,
        )
        assert [x['atom_count'] for x in rejected] == expected
        assert len(rows) == 1 and attempted == 1 + len(expected)
    rows, rejected, attempted = _collect_plan_rows(
        lambda width: [plan('Si21')], requests=1, batch_size=1, seed=17,
        usage_role='evaluation', max_atoms=20, resample_over_capacity=False,
    )
    assert len(rows) == 1 and not rejected and attempted == 1
