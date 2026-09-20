# Seen-TRAIN feedback study, 20 September 2026

This study uses 16 selected TRAIN conditions and four fresh body-sampling streams per condition. Both the retained starting model and the feedback-trained model use the same body temperature 0.2, C1 implementation, request order and physical protocol. The learned model was fixed before this panel's physical outcomes. The result concerns this selected, seen set; it is not an unseen MP-20 benchmark claim.

The starting point is the retained round-1 draft adapter and its C1 head used when this correction began. The table is not a new measurement of the original paper B0 checkpoint. An original-B0 comparison and independent development/final panels belong to the subsequent larger study.

## Whole 64-request panel

| Output / metric | Starting model | Feedback-trained model |
|---|---:|---:|
| Raw native stability proxy | 7/64 | 35/64 |
| Raw conventional SUN | 6/64 | 13/64 |
| Raw conventional MSUN | 11/64 | 19/64 |
| Raw relaxed Stable | 32/64 | 48/64 |
| Raw relaxed MetaStable | 43/64 | 54/64 |
| Raw SUN unknown | 3/64 | 4/64 |
| F800 SUN | 10/64 | 9/64 |
| F800 MSUN | 17/64 | 16/64 |
| F800 relaxed Stable | 49/64 | 51/64 |
| F800 relaxed MetaStable | 64/64 | 64/64 |
| F800 SUN unknown | 0/64 | 0/64 |

The raw native proxy uses the unrelaxed exact output: CHGNet raw hull ≤0 eV/atom, maximum force ≤0.1 eV/Å and maximum stress ≤0.5 GPa. It increased by 28 requests with no losses and no unknowns. A bootstrap grouped by the 16 Plan sources gives a conditional 95% increase interval of 23.44–65.63 percentage points. This interval describes the observed source panel, not a population guarantee.

Native mean differences (learned minus starting) were −0.401 eV/atom in energy, −3.008 eV/Å in force RMS, and −21.231 GPa in maximum stress. The energy and force intervals cross zero; the stress interval is negative. Difficult cases remain. Exact physical-record keys decrease from 56 to 35 among 64 requests: sampling is more concentrated, and 35 successful requests must not be called 35 distinct new materials.

F800 point performance is close, with one fewer joint-panel SUN/MSUN and two more Stable outputs. This is not a formal noninferiority claim. F800 also need not preserve every already-good raw output; no oracle keep/fallback policy is credited as deployed learned behavior here.

## Separate-stream analysis

Scoring four 16-request panels separately gives raw SUN totals 16→32 and MSUN 27→38; F800 totals are SUN24→27 and MSUN39→40. These are not joint-64 Unique counts and are not used in the table above. Failures and unknowns remain in all denominators.

## What changed and what did not

The material uses physically equivalent teacher registration and independent exact-token remeasurement, followed by supervision at actual student prefixes. Training uses 512 LoRA updates with 75% complete teachers and 25% prefix corrections. C1 parameters and the continuous model are fixed. The selected checkpoint reaches 98.0% coordinate and 95.8% lattice-field top-1 accuracy on fixed teacher views; those fitting numbers are not the generation result.

Earlier T0.7 trials were substantially weaker: native proxy counts were 1→10 on one panel and 1→3 on a fresh-noise panel, with worse mean raw forces. They are not presented as robust confirmations. A subsequent 340-view probability audit found no strong suppression of correct teacher values by C1 or geometric support, motivating one fixed, matched T0.2 comparison. No temperature grid or final-test tuning was used. The temperature itself was an engineering choice, not a learned C2 prediction.

The present evidence supports a training-set mechanism for physical feedback learning under the frozen configuration. It does not establish generalization, an independent learned-verifier contribution, or fewer F steps. The continuous-to-discrete pathway uses physical tools during data preparation and evaluation; draft inference has no access to teacher records or true physical labels.

## Release validation

The curated release passes 122 server-side tests. With the actual model assets, one draft from each comparison arm reproduces the accepted body tokens and T0.2 trace. Replaying the accepted 64-input F800 batch with the same device partitioning reproduces all 64 saved output records. No new physical evaluation or training was used for this packaging check. This verifies the tested execution path and does not broaden the scientific claim above.
