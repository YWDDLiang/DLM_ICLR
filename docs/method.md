# Method and default training recipe

## Representation and generation

A rich Plan specifies atom count, species counts and categorical material context. The dynamic body has `7 + 4N` tokens: one count, six lattice parameters, and `element / x / y / z` for each site. The retained vocabulary covers 1–20 sites and elements through Pu.

G fixes the count and species tokens. It reveals the lattice, then species-grouped X, Y and Z coordinates using the retained paired random streams. Periodic coordinate aliases are combined and the existing numerical geometry support is applied. Recovery reopens failed sites, neighboring sites, then the full numeric body. A final Z continuation can relax inter-site distance support while retaining the original duplicate-coordinate guard. A lattice failure can reopen gamma conditional on alpha/beta. These attempts retain the same Plan and are recorded, including unsuccessful requests and actual model calls.

F runs the trained CrysLLMGen refinement for 800 predictor/corrector steps, with two decoder evaluations per step. Its seed and batch-one execution are preserved. The raw lattice matrix is saved; the endpoint uses the original lengths/angles readout.

## Autonomous KEEP/EDIT

E receives token views of the current continuous structure. The actor predicts action scope, site count and site scores. The primary candidate uses the learned non-KEEP action; subsequent candidates use separate random streams at learned site ranks. There are up to eight candidates within a shared budget of 80 additional DLM evaluations. For one-site crystals, a coordinate-only change is a rigid translation, so candidates open the cell as well. The actual number of proposals can therefore be smaller.

An edit is bound to its continuous input once. Unchanged fields retain their original floating-point values; only changed token fields are decoded back into the structure. Periodically equivalent changes and rigid translations return KEEP. Invalid patches also retain the original structure and record their failure.

Let `x(c,p,a)` be the 8192-dimensional model feature of current `c`, proposal `p` and action `a`. The value model applies a learned 8192→128 projection and SiLU, concatenates nine geometry features, normalizes with TRAIN statistics, and applies a 137→2 readout. Its outputs estimate NS and NMS values, where N is novelty, S is strict stability, and MS is inclusive metastability.

For each candidate, the predicted gain is

\[
\Delta v = v(x(c,p,a),g(c,p,a)) - v(x(c,c,\varnothing),g(c,c,\varnothing)).
\]

KEEP has exactly zero gain. Selection maximizes `2 * ΔNS + ΔNMS`; utility ties use ΔNS, then prefer KEEP, then retain candidate order. The values are regression scores, not guaranteed calibrated probabilities.

The nine additional inputs describe site count, edited-site and token fractions, whether the cell changed, fractional/Cartesian displacement RMS and maximum, and geometry availability. The feature view uses task 1, remaining-budget context 80, and reveal fraction 1. Actual runtime calls are counted separately. The original quality-head acceptance probability is recorded for analysis; the autonomous value comparison makes the final choice.

**Inference does not receive measured energies, hull distances, forces, stresses, stability labels, novelty labels or teacher predictions.** G→F→E completes before physical evaluation begins.

## Offline updates

Each round generates feedback from separate TRAIN Plans. The same source's current state and candidate endpoints are compared after physical evaluation. E uses one actor-supervision row per source, choosing a verified useful candidate where available. Rejection labels describe that measured candidate pool and do not prove that no possible useful edit exists.

G learns from full-token structures. Candidate selection for a teacher may use the measured continuous endpoints, but the complete token teacher is constructed and **evaluated separately** before its label is used. A continuous structure's physical label is not assigned to a geometrically different quantized body.

| Component | Default settings |
|---|---|
| G | LoRA LR `5e-6`; 4 epochs; 32 source rows per device; 2 conditional mask cuts; preference β `0.1`; anchor weight `0.2`; reference KL weight `1.0` |
| E content | LR `2e-6`; 8 epochs; 16 source rows per device; complete atom-block permutations; dense typed-token teacher loss; reference KL weight `1.0` |
| E decision heads | LR `1e-4`; categorical site supervision; content features detached for the decision loss |
| Value student | 64 epochs; 32 sources per minibatch; head LR `1e-3`; hidden LR `1e-5`; no weight decay; gradient clipping at 1 |
| Offline value teacher | 64 epochs; row batch 256; Adam LR `1e-3`; ridge `0.1`; SUN ranking coefficient `0.2`; 50% teacher mixing |

The value student uses gain MSE, `0.1` times before/after level MSE, within-source SUN ranking with weight `0.2`, and output-weight ridge `0.1`. Its offline teacher may use twelve current-state physical features. Those columns and the teacher are absent from the inference API. Normalization statistics and all fitting targets come only from TRAIN sources.

G and E preserve the original IO tables and update the existing LoRA/model modules. Data passes are shuffled; each real optimizer step and source exposure is recorded. KL regularizes updates without terminating the round. The requested number of rounds is independent of metric improvements.

## Evaluation

The common evaluator uses CHGNet 0.3.0 weights, FIRE, full-cell `FrechetCellFilter`, zero external pressure, `fmax=0.1 eV/Å`, maximum stress `0.5 GPa`, and up to 1000 steps. Joint atomic force and stress determine convergence. Terminal energy is checked on stored, wrapped and rigidly shifted representations with a `0.001 eV/atom` tolerance.

Strict stability uses terminal energy minus reference hull energy ≤0; inclusive metastability uses ≤0.1 eV/atom. The main thresholds and the stricter verified subsets are reported separately. Composition and structure validity follow the retained CrysLLMGen metrics.

Novelty and uniqueness compare **the selected input structure before CHGNet relaxation**, using `StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)`. Novelty compares against the same-composition training references. Uniqueness compares each structure to every earlier same-formula structure in request order, including earlier structures that are unstable or non-novel. It does not use transitive clustering. Unknown required matches and missing references remain explicit.

Every selected Plan remains in the denominator, including downstream generation failures. Formal TRAIN data is composition-disjoint from the evaluation cohort. This condition concerns adaptive training; it does not assert that the base language/crystal pretraining corpora exclude every evaluation composition.
