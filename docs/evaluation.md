# Evaluation

## Direct

The fast path reports composition and structure validity on saved structures. Composition validity is SMACT 3.1 screening plus a mixed-valence supplement using the same oxidation-state table. Structure validity follows the CrysLLMGen geometric criterion. `V` denotes their conjunction.

```bash
bash scripts/evaluate.sh direct --config configs/local.json \
  --structures outputs/mp20/samples/edited.jsonl --output outputs/mp20/direct
```

`--full` also computes fingerprints, density/element-count distribution distances and structural/composition coverage. Supply a reference split with `--reference`. Full Direct's fingerprint-based `valid` is reported separately from the two basic validity predicates.

## Materials Project hull

After F refinement, retrieve corrected GGA/GGA+U competitor entries for each chemical system and all its subsystems:

```bash
export MP_API_KEY='YOUR_MATERIALS_PROJECT_API_KEY'
bash scripts/query_hull.sh --config configs/local.json \
  --structures outputs/mp20/samples/refined.jsonl
```

The downloader records the database version, caches exact-system queries and assembles phase-diagram inputs. It reuses completed queries when extending a dataset. The candidate's energy is computed by CHGNet; MP entries define the competing-phase hull energy at that composition.

## SUN, MSUN and VUN

Physical evaluation uses CHGNet model 0.3.0, joint atomic/cell FIRE relaxation, up to 1000 steps, force tolerance 0.1 eV/Å and stress tolerance 0.5 GPa. A fresh terminal energy check validates the resulting label.

$$E_{\mathrm{hull}}(x)=e_{\mathrm{CHGNet}}(\mathrm{relax}(x))-e_{\mathrm{MP\ hull}}(\mathrm{composition}(x)).$$

Stable means verified `E_hull ≤ 0`; MetaStable means verified `E_hull ≤ 0.1 eV/atom`, including Stable. Novelty compares the submitted structure to training structures of the same formula. Uniqueness compares each structure to earlier same-formula outputs in saved request order. These comparisons use the retained directed StructureMatcher protocol.

- `SUN = Stable ∧ Unique ∧ Novel`.
- `MSUN = MetaStable ∧ Unique ∧ Novel`.
- `VUN = composition_valid ∧ structure_valid ∧ Unique ∧ Novel`.

```bash
bash scripts/evaluate.sh sun --config configs/local.json \
  --structures outputs/mp20/samples/edited.jsonl --output outputs/mp20/sun
```

Each metric stores `passed`, `pending`, `requests`, count bounds and percentage bounds. Every requested output has a score row. Physics caches are keyed by structure and evaluation protocol; matching caches preserve the comparison direction and reference. Increasing the panel reuses existing labels while computing uniqueness in the full saved order.

The default generation workflow evaluates F before C2 so confirmed SUN inputs can be retained. Final C2 scores use the same physics and matching protocol. Computation and result files remain separate for these two evaluation stages.

Code: [Direct](../src/dlm_iclr/evaluation/direct.py), [MP queries](../src/dlm_iclr/evaluation/hull.py), [physics](../src/dlm_iclr/evaluation/physics.py), [metrics](../src/dlm_iclr/evaluation/workflow.py).
