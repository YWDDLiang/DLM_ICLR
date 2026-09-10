# Results and evidence boundaries

## Completed KEEP/EDIT component study

The released value head was selected with the trained minibatch editor and eight learned site-rank candidate streams. Selection used model features and geometry only. Measured current-state physical information was used for offline supervision and teacher fitting, not as an inference input.

| Panel | Requests | ΔSUN vs KEEP | ΔMSUN vs KEEP | ΔStable vs KEEP |
|---|---:|---:|---:|---:|
| Previously inspected development panel | 247 | +2 | +8 | +1 |
| Previously inspected second development panel | 246 | +1 | +6 | 0 |
| Previously unused compositions | 96 | 0 | +5 | 0 |

The two development panels were jointly used for model selection after several exploratory comparisons; neither is a blind final test. The 96-composition set was frozen before looking at its physical outcomes, using source exclusions and availability of fixed F inputs and static references. No retuning followed that result.

On those 96 inputs, SUN was **6→6**, MSUN **49→54**, and Stable **6→6**, with 40 selected edits. One Pd₂Er₂ edit crossed the strict-stability threshold by about 1.45 meV/atom; one GeTe₂ edit lost an existing SUN result. This supports useful autonomous edits and net MSUN benefit in that component test, while a robust SUN generalization gain has not been established.

The editor's content training completed 248 real minibatch updates over 492 sources, with eight visits per source. LoRA, state/numeric modules and localization parameters changed. The value model was trained on 3930 candidate rows from those TRAIN sources, for 64 passes and 1024 optimizer steps.

Earlier larger gains that used measured current stability, novelty or hull information are offline assisted comparisons. They are not autonomous inference results. Successful local edits do not establish that every difficult request is repairable; a previously investigated failure remained unresolved.

## Formal full-pipeline reporting

The formal protocol uses the same first 1050 legal H1A2 Plans in original order, including the one-site request, and a separate clean set of 1000 TRAIN conditions. Each run records S0 and three complete self-improvement rounds. Sampling/training seeds and all request outcomes are retained.

The public main table will report one complete trajectory selected from three runs by final S3 confirmed SUN count, then confirmed MSUN count, then confirmed Stable count. These are the lower bounds when a predicate remains unknown. The reference and this selection rule are fixed before the formal runs. The result will be explicitly labelled **best of three**, without combining different rounds or checkpoints from different runs. The two other trajectories are retained as private experiment records. Full-pipeline counts are added only after those runs have completed; the component results above are not substituted for them.

The frozen GGA/GGA+U reference currently lacks sufficient Yb entries for 56 of its 1835 requested chemical systems; 1779 systems have complete element coverage. The incomplete reference affects 30 of the 1050 evaluation conditions and 28 of the 1000 TRAIN conditions. A constructible phase diagram for a smaller subsystem does not establish coverage of a requested chemistry. The Materials Project database history documents Yb pseudopotential changes ([database versions](https://docs.materialsproject.org/changes/database-versions)). We do not fill gaps with energies from a different functional or deprecated entries.

All requested Plans remain in the denominator. For each metric, reports include the confirmed count, unknown count, and the interval `[confirmed, confirmed + unknown]`; a point count is `null` when that metric is unresolved. A confirmed novelty or uniqueness failure can resolve SUN to false even if stability is unknown. TRAIN endpoints without reliable physical targets are excluded from the affected supervision, with source and feedback counts retained. Reference gaps therefore limit both the reported stability precision and the available training labels.
