# Direct, SUN and MSUN evaluation

Evaluation reads saved structures after generation and selection. Every input row
remains in the requested denominator, including failed generations and structures
that cannot be reconstructed. Neither command generates replacement samples.

## Direct: full metrics and a validity-only option

Install the CPU evaluation dependencies:

```bash
python -m pip install -e '.[direct]'
```

On Windows with a non-UTF-8 system locale, set `$env:PYTHONUTF8 = "1"` in
PowerShell before starting Python. SMACT's packaged tables require UTF-8 text
decoding. The same setting should be used for SUN scoring.

Run the full CrysLLMGen generation metric set on an inference directory:

```bash
dlm-iclr evaluate-direct --run outputs/h1a2-1050 \
  --metrics full --reference /data/mp20/test.csv --dataset mp20 --workers 8
```

This writes `evaluation/direct/full/scores.jsonl` and
`evaluation/direct/full/summary.json` under the run directory. The reference is
the held-out dataset split used for the distribution and coverage comparisons.
It is supplied explicitly; the command does not substitute the training set used
for SUN novelty.

For a fast check of **only `comp_valid` and `struct_valid`**:

```bash
python -m pip install -e '.[validity]'
dlm-iclr evaluate-direct --run outputs/h1a2-1050 --metrics comp_struct
```

This writes `evaluation/direct/comp_struct/`. It does not load matminer or create
fingerprints, read the reference dataset, compute distances/coverage, load a
neural model, or run physical relaxation. It requires no configuration file,
reference data or GPU. The two validity definitions are identical in both modes.
Uncomputed metrics are listed in `omitted_metrics`, rather than assigned zeros.

The complete Direct metric set is:

| Metric | Definition |
|---|---|
| `comp_valid` | SMACT charge neutrality and Pauling electronegativity checks on reduced composition; single elements and all-metal alloys retain the upstream acceptance rules |
| `struct_valid` | Upstream periodic pair-distance check at 0.5 Å and minimum volume of 0.1 Å³ |
| `valid` | Upstream full evaluator's composition AND structure validity, also requiring successful composition and CrystalNN fingerprints |
| `wdist_density` | Wasserstein-1 distance between valid generated densities and all reference densities, in g/cm³ |
| `wdist_num_elems` | Wasserstein-1 distance between valid generated and reference numbers of distinct elements |
| `cov_recall` | Fraction of reference structures covered under both fingerprint-distance cutoffs |
| `cov_precision` | Number of covered generated structures divided by **all requested generations**, including failures |

Validity and coverage values in `metrics` are percentages rounded to four decimal
places. Wasserstein distances are rounded to four decimal places and are not
percentages. Exact validity counts, unrounded percentages, known/unknown counts
and count bounds are also saved. A validity implementation error is recorded as
unknown; a failed generation or failed reconstruction is invalid.

Full-mode `valid` follows the original fingerprint availability rule. Therefore,
a fingerprint failure can make `valid` false while leaving both basic validity
checks true. This also controls which generated structures enter the Wasserstein
comparisons. When none qualify, both Wasserstein distances are `null` with an
explicit reason. Coverage with no generated fingerprints is zero; its generated
denominator is still the complete request count.

Composition fingerprints use `ElementProperty.from_preset("magpie",
impute_nan=False)`, the bundled frozen 132-feature scaler and the upstream
standardized-NaN-to-zero rule. Structure fingerprints are the mean across sites
of `CrystalNNFingerprint.from_preset("ops")`. For each reference/prediction,
the structural and composition nearest distances are minimized **independently**,
as in the retained CrysLLMGen implementation. The two nearest witnesses need not
be the same structure. Coverage considers all fingerprintable generated
structures, including those that fail composition validity.

The MP-20 cutoffs are 0.4 for structure and 10.0 for standardized composition.
The upstream `carbon` and `perovskite` presets both use 0.2 and 4.0. Their
availability does not establish benchmark performance on those datasets.
Reference reconstruction or fingerprint errors stop full evaluation with the
affected row numbers; reference rows are never silently removed.

`--workers` controls independent CPU fingerprint processes. Successful
fingerprints are cached by exact structure, package versions, scaler and
implementation hashes. Repeating an evaluation reuses those features; failed
fingerprints are retried. `--cache /path/to/cache` shares the cache across output
directories. Coverage is computed in distance blocks to bound peak memory.

## SUN and MSUN

The explicit command `evaluate-sun` reports both SUN and MSUN. The existing
`evaluate` command remains an alias for the same evaluation:

```bash
python -m pip install -e '.[physics]'
dlm-iclr evaluate-sun --config configs/local.json \
  --run outputs/h1a2-1050 --gpus 5 --physics-workers 4 --nu-workers 4
```

Only three configuration assets are needed for this standalone evaluation:
`chgnet`, `hull_cache`, and `novelty_reference`. Generator, editor, value and
refiner checkpoints are not loaded. Use `--gpus 0` to run physical evaluation on
CPU. Shell and Slurm entry points accept the same arguments:

```bash
sbatch --partition YOUR_PARTITION --gres=gpu:5 --cpus-per-task=20 \
  scripts/run.sbatch evaluate-sun --config configs/local.json \
  --run outputs/h1a2-1050 --gpus 5 --physics-workers 4 --nu-workers 4
```

Let \(e_h\) be the terminal CHGNet energy per atom minus the reference hull
energy per atom for the same composition. The output fields are:

| Report name | Per-request field | Predicate |
|---|---|---|
| SUN | `strict_sun` | \(e_h \leq 0\) AND novel AND unique representative |
| MSUN | `meta_sun` | \(e_h \leq 0.1\,\mathrm{eV/atom}\) AND novel AND unique representative |

MSUN includes strict SUN. Novelty compares the **saved output geometry before
CHGNet relaxation** against same-formula training structures. Uniqueness compares
each output to every earlier same-formula output in the original input order;
an earlier structure remains a witness regardless of its stability or novelty.
The directed pymatgen `StructureMatcher` uses `ltol=0.2`, `stol=0.3`, and
`angle_tol=5`. Pair results are cached with their geometry and matcher identity.

Physical evaluation retains CHGNet 0.3.0 weights, FIRE with cell relaxation,
joint atomic force/stress stopping at 0.1 eV/Å and 0.5 GPa, and at most 1000
steps. Main SUN/MSUN use terminal energy. The separate `verified_strict_sun` and
`verified_meta_sun` fields additionally require successful terminal verification.
The raw/terminal statuses, energies and verification details remain in each row.

Results are saved in `evaluation/scores.jsonl`, `evaluation/summary.json`, and
`evaluation/physics/`. In the summary, `metrics.SUN` and `metrics.MSUN` are
percentages; the corresponding underlying fields remain `strict_sun` and
`meta_sun` in `counts`, `percent`, `known_counts`, `unknown_counts`, and
`count_bounds`.

If references or necessary matching/physical work are unresolved, affected
counts and percentages are `null`. For example, 6 confirmed SUN structures and
2 unresolved possibilities among 100 requests yield `count_bounds.strict_sun =
[6, 8]`, not an asserted SUN rate of 6%. Missing reference systems and errors
are retained in the report. Failed generation remains a confirmed failure in
the full denominator. Validity is reported alongside SUN, but is not an extra
gate inserted into the retained SUN predicate.

## Reuse physical labels or evaluate a standalone file

Saved geometry-bound labels can be scored again without CHGNet or PyTorch:

```bash
python -m pip install -e '.[validity]'
dlm-iclr evaluate-sun --config configs/local.json \
  --run outputs/h1a2-1050 \
  --labels outputs/h1a2-1050/evaluation/physics/labels.jsonl \
  --output outputs/h1a2-rescored
```

This path requires only the hull and novelty reference assets. The label count,
source IDs, input order and exact structure keys are checked before scoring.
Reuse the labels produced for these exact outputs under the intended physical
protocol; a different geometry or reordered label list is rejected. The summary
records the hashes of both the structure input and reused label file.

Both commands also accept `--structures FILE.jsonl --output DIRECTORY` instead
of `--run`. For example:

```bash
dlm-iclr evaluate-direct --structures outputs/generated.jsonl \
  --output outputs/direct-full --reference /data/mp20/test.csv --workers 8

dlm-iclr evaluate-direct --structures outputs/generated.jsonl \
  --output outputs/direct-fast --metrics comp_struct

dlm-iclr evaluate-sun --config configs/local.json \
  --structures outputs/generated.jsonl --output outputs/sun --gpus 1
```

Use the pipeline's `structures.jsonl` format: `source_id`, `ordinal`, boolean
`success`, and a pymatgen `structure` dictionary or supported crystal-token
`body`. Legacy `attempt_id`/`status="succeeded"` JSONL is also accepted. If IDs
or ordinals are omitted, they are assigned from row position; duplicates are
rejected and rows are never sorted. Supply the complete original request list,
including failed rows, to retain the intended denominator.

Full Direct references accept a CIF CSV or JSONL containing `structure`,
`target_structure` from `prepare-data`, or `cif`. SUN novelty references accept
the documented CIF CSV or `structure`/`cif` JSONL formats. Direct references
normally use a held-out split; SUN novelty uses the training split.

## Python interfaces

```python
from dlm_iclr.config import load_config
from dlm_iclr.direct import evaluate_direct
from dlm_iclr.evaluation import evaluate_sun
from dlm_iclr.evaluation_inputs import load_records

records = load_records("outputs/sample/structures.jsonl")
rows, fast = evaluate_direct(records, "outputs/direct-fast", metrics="comp_struct")
rows, full = evaluate_direct(
    records, "outputs/direct-full", reference="/data/mp20/test.csv", workers=8,
)
rows, sun = evaluate_sun(
    load_config("configs/local.json"), "outputs/sample/structures.jsonl", "outputs/sun",
    labels="outputs/sample/evaluation/physics/labels.jsonl", nu_workers=4,
)
```

For Python programs using multiple workers, place the calling code inside an
`if __name__ == "__main__":` guard so process spawning also works on Windows.

The Direct formulas and scaler derive from CrysLLMGen commit
`94bb287751cd20a882c7c1df7ca736633d78e5e1`, specifically `GenEval`, the fingerprint
construction in `compute_metrics.py`, and `compute_cov` in `eval_utils.py`.
The standalone interface preserves failed requests and reports undefined values
explicitly. See [third-party notices](../THIRD_PARTY_NOTICES.md) and the recorded
[environment](environment.md) for attribution and reproducibility.
