# Results and evidence boundaries

The latest full-pipeline evidence is **one completed S0/S1 comparison**, with
1050 requests per snapshot. The planned three runs of three updates were
cancelled by the user because of runtime cost. No S2/S3 or best-of-three result
is reported. [Published data](../data/experiments/README.md) contains the actual
structures, scores, paired changes, runtime and training records. The
[detailed Chinese report](FRAMEWORK_AND_EXPERIMENTS_ZH.md) describes the complete
method and ends with the historical development of the approach.

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

## Completed full-pipeline comparison: S0 to S1

The registered protocol uses the first 1050 legal H1A2 Plans in original order,
including the one-site request, and a separate clean set of 1000 TRAIN conditions.
The completed S0 and S1 use identical Plan files and G/F sampling streams. S0 is
the full G/F/E workflow using the already-trained initial checkpoints; S1 follows
one actual G/E/value update on separate TRAIN feedback.

No new Planner sampling was performed. Run 1 retained the saved G/F streams.
Runs 2/3 were planned with sampling seeds 314159/271828 and training base seeds
20261911/20262911, but never started. Run 1 used training base seed 20260911,
incremented by the update index. The original plan remains in
[formal_protocol.json](formal_protocol.json); actual completion and cancellation
are recorded in [experiment_status.json](../data/experiments/completed_1050/experiment_status.json).

The user stopped the job after reviewing S0/S1. Round 2 had generated TRAIN
candidates but was interrupted during physical feedback, before a second update
or S2. The original final-S3 selection rule was therefore **not applied**. The
following table reports the completed pair, without selecting a winner or
constructing an artificial complete trajectory.

| Metric | S0 confirmed | S0 unknown | S1 confirmed | S1 unknown |
|---|---:|---:|---:|---:|
| Reconstructed | 1049 | 0 | 1050 | 0 |
| Composition valid | 923 | 0 | 924 | 0 |
| Structure valid | 1048 | 0 | 1050 | 0 |
| Strict stable | 120 | 30 | 119 | 29 |
| Inclusive metastable | 618 | 30 | 616 | 29 |
| SUN | 112 | 29 | 107 | 29 |
| MSUN | 560 | 29 | 562 | 29 |
| Terminal verified | 1038 | 0 | 1034 | 0 |

The denominator is 1050 for every row. SUN count bounds are `[112,141]` and
`[107,136]`; MSUN bounds are `[560,589]` and `[562,591]`. The missing-label bounds
are not statistical confidence intervals. Complete SUN/MSUN point percentages
remain null.

| Paired predicate | Both-known requests | Gains | Losses | Net |
|---|---:|---:|---:|---:|
| Strict stable | 1020 | 26 | 27 | -1 |
| Inclusive metastable | 1020 | 102 | 104 | -2 |
| SUN | 1021 | 27 | 32 | -5 |
| MSUN | 1021 | 124 | 122 | +2 |

The SUN/MSUN unknown sets are identical across the two snapshots. E selected
495 edits at S0 and 669 at S1; more edits did not yield a net SUN improvement.
G, E and value changed together, so this comparison cannot isolate any one
component's causal contribution. It does not establish repeated self-improvement
or multi-seed generalization.

The first update completed G: 730 sources / 92 optimizer steps; E: 972 sources /
488 minibatch steps; value: 7766 rows from 972 sources / 1984 steps and 64 visits
per row. All three saved nonzero parameter changes. The different row counts
reflect feedback eligibility, not a reduced evaluation denominator.

Each full evaluation rollout required roughly 73–78 minutes for G/F/E on five
A800 GPUs, before physical evaluation. First-round TRAIN generation took about
69 minutes, and its approximately 10,000 unique candidate labels took about
108 minutes. At cancellation, total job runtime was 8h 47m 42s, including partial
round-2 work. Value fitting-loop time excludes feature extraction and teacher
preparation. Detailed timings and per-request calls are in the data directory.

This publication does not rerun full Direct fingerprints on the snapshots.
Only the two validity metrics present in the saved evaluation are reported for
S0/S1; no Wasserstein or COV values are fabricated.

The frozen GGA/GGA+U reference currently lacks sufficient Yb entries for 56 of its 1835 requested chemical systems; 1779 systems have complete element coverage. The incomplete reference affects 30 of the 1050 evaluation conditions and 28 of the 1000 TRAIN conditions. A constructible phase diagram for a smaller subsystem does not establish coverage of a requested chemistry. The Materials Project database history documents Yb pseudopotential changes ([database versions](https://docs.materialsproject.org/changes/database-versions)). We do not fill gaps with energies from a different functional or deprecated entries.

All requested Plans remain in the denominator. For each metric, reports include the confirmed count, unknown count, and the interval `[confirmed, confirmed + unknown]`; a point count is `null` when that metric is unresolved. A confirmed novelty or uniqueness failure can resolve SUN to false even if stability is unknown. TRAIN endpoints without reliable physical targets are excluded from the affected supervision, with source and feedback counts retained. Reference gaps therefore limit both the reported stability precision and the available training labels.

## Historical experiments

The [historical ledger](../data/experiments/historical/ledger.json) publishes 57
source-recorded rows and 15 source capsules. Every row identifies its comparison
group, endpoint, denominator and source pointer. Survivor-prefix views, raw and
refined endpoints, different conditional models, repeated streams and independent
cohorts are kept distinct. These rows are not a common leaderboard and were not
re-evaluated under the current 1050-request protocol.

Earlier clean-1000, T2T and current-state-assisted ranking studies are separately
archived in [development records](../data/experiments/development/index.json).
Their recorded improvements must not be substituted for autonomous inference
evidence. The detailed report's final section summarizes successful and failed
method changes chronologically.
