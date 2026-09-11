# Existing experiment data

These files organize already completed experiments. No additional G/F/E rollouts,
training experiments or CHGNet relaxations were performed for this publication.

The current full-pipeline experiment completed **run 1 / S0 and S1** on the same
1050 saved Plans. The planned three runs with three updates each were cancelled
by the user because of runtime cost. Round 2 was interrupted during TRAIN
physical feedback, before a second model update or S2. No best-of-three result
or complete S0–S3 trajectory exists for this experiment.

## Data groups

| Directory | Contents | Interpretation |
|---|---|---|
| [completed_1050](completed_1050/) | 2100 structure/score records, common Plans, paired CSV, runtime, model hashes and first-round training records | One completed before/after comparison; not a multi-run result |
| [components](components/autonomous_keep_edit.json) | Earlier autonomous value/editor study, model-selection panels, unused-composition panel and diagnostic cases | Development and 96-composition component evidence, separate from the 1050 experiment |
| [historical](historical/ledger.json) | 57 historical rows and 15 source capsules with group, endpoint, denominator and source pointers | Different estimands; no cross-cohort ranking |
| [development](development/index.json) | Earlier clean-1000, T2T, acceptance and current-state-assisted ranking records | Archived development evidence; additional physical information and TRAIN use are labelled explicitly |

[Detailed Chinese framework/report](../../docs/FRAMEWORK_AND_EXPERIMENTS_ZH.md)
and [English result summary](../../docs/results.md) describe the methods and limits.

## Read the completed comparison

- [Per-request score CSV](completed_1050/per_request_scores.csv): 2100 rows, labelled S0/S1.
- [Paired request CSV](completed_1050/paired_requests.csv): 1050 matched requests.
- [Paired count summary](completed_1050/paired_summary.json): gains, losses and both-known denominator.
- [Runtime CSV](completed_1050/runtime.csv): G/E calls, F source/fallback and E selection for every request.
- [S0 summary](completed_1050/S0/summary.json) and [S1 summary](completed_1050/S1/summary.json).
- S0/S1 `structures.jsonl.gz` preserve selected continuous structures, failures, IDs and declared compositions.
- S0/S1 `scores.jsonl.gz` preserve all original per-request scientific score fields, including raw/terminal force and stress diagnostics.
- [Common Plans](completed_1050/plans.jsonl.gz), [run configuration](completed_1050/run_configuration.json),
  [source split](completed_1050/source_split.json), [model identities](completed_1050/model_identities.json)
  and [physical protocol](completed_1050/physics_protocol.json).
- [Training summary](completed_1050/training/summary.json), G/E/value training curves,
  parameter deltas, and G/E source exposure records.
- [Actual KEEP/EDIT examples](completed_1050/keep_edit_examples.json): two inference
  decisions with all candidate scores, and positive/negative TRAIN rows with their
  value comparisons and physical labels. [Mechanism walkthrough](../../docs/KEEP_EDIT_ZH.md).
- [Actual experiment status](completed_1050/experiment_status.json) and
  [source artifact hashes](completed_1050/source_artifacts.json).

CSV uses the literal `unknown` for missing values; JSON uses `null`. A missing
predicate is not zero. `count_bounds` is `[confirmed, confirmed + unknown]`,
not a statistical confidence interval. Complete point counts/percentages remain
null if any required labels are unresolved. Every original request stays in the
denominator, including failed generation.

```python
import gzip
import json

with gzip.open("data/experiments/completed_1050/S0/structures.jsonl.gz", "rt", encoding="utf-8") as stream:
    structures = [json.loads(line) for line in stream]
```

Use Python or another JSON implementation that preserves nonnegative 63-bit
integer Plan seeds. Avoid converting the seeds through floating point.

To pass compressed structures to the CLI, decompress them to an ordinary JSONL
first; the evaluator's `--structures` input is a plain JSONL file.

## Verify the published projection

From the repository root:

```bash
python scripts/verify_experiment_data.py
```

This standard-library-only command checks [manifest.json](manifest.json), all
2100 structure/score bindings, Plan composition/order, counts and unknown bounds,
paired changes, runtime budgets, and recorded training exposure. It does not
recompute energies or assert physical truth independently of CHGNet.
It also replays the two example decisions from saved value gains and checks
continuous KEEP and unchanged-coordinate preservation.

Infrastructure paths have been removed from configuration and historical
metadata. Original source hashes and published hashes are distinct where the
projection changed serialization or removed paths. The generated S0/S1 structure
and scientific score values are retained without rounding or outcome filtering.
Historical source capsules are compressed JSON/JSONL/text; their numeric data are
retained, with infrastructure locations redacted.

Large G/E/Planner/refiner weights and the original MP-20/hull reference data remain
external assets. Publishing their identities does not imply download availability.
See [assets](../../docs/assets.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md).
