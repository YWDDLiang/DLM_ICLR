# Registered physical feedback learning

The objective is to transfer improvements found by continuous refinement and physical tools into the discrete draft generator. The evaluated path keeps the original Plan/prompt, C1 sampler, physical thresholds, and the continuous refiner. It updates draft LoRA parameters using physically checked targets. There is no differentiation through F or CHGNet, no post-F C2 edit, and no learned online verifier claim in this result.

## What C2 contributes, and how it cooperates with C1

C1 handles periodic dependencies while constructing the draft. C2 supplies the experience from which a better constructor is learned: continuous lattice/coordinate candidates, exact-token verification, representation registration and corrections aligned with actual student prefixes. The active C2 component is a physical feedback and supervision mechanism; it should not be described as an independently successful learned online value model.

Updating DLM LoRA changes its lattice/coordinate predictions and the hidden states consumed by C1. The fixed C1 head then uses these updated inputs to rebuild its periodic axis distribution during the next generation. This provides an offline-learning path from C2 feedback to online coordinate choices. C1 parameters are not readapted in the evaluated 75/25 recipe. See [the C2 module explanation](modules/c2.md) and [中文说明](C2_FEEDBACK_ZH.md).

The [complete method and mathematical account](C1_C2_METHOD_ZH.md) covers C1's conditional tree law, its confidence-projected commits, equivalent complete teachers, prefix-constrained completions, the native utility, and the actual learning objective. The [C1 module](modules/c1.md) also gives its probability and training equations in English.

## Why register a teacher representation?

A periodic translation or a permutation of same-element sites can change many tokens without changing the crystal. In the initial study, literal token accuracy improved while physically aligned free-generation coordinates became worse. The correction registers a complete teacher near one retained, pre-learning draft of the same Plan. It permits only integer-bin global translations and same-element permutations; the teacher lattice and relative periodic geometry remain unchanged.

`draft_loop.teacher_registration.align_teacher_to_anchor` returns the new token body and its transformation metadata. This is an offline operation, not an inference-time structure lookup. Its result is decoded and physically remeasured under its own record key before learning.

## Constructing the reviewed material

The preparation APIs are ordinary Python functions:

| Task | API |
|---|---|
| Generate original drafts and save traces | `draft_loop.backend.sample_drafts` |
| Generate continuous candidates and exact-token physical supervision | `draft_loop.backend.collect_teacher_candidates` and `measure` |
| Register an equivalent complete teacher | `draft_loop.teacher_registration.align_teacher_to_anchor` |
| Extract actual recorded coordinate commits | `draft_loop.state_replay.coordinate_replay_views` |
| Propose a completion without changing visible fields | `draft_loop.prefix_teacher.propose_completion` |
| Check the complete exact-token candidate | `_core.r03_physics_transfer.geometry_support_report`, then `draft_loop.backend.measure(..., singlepoint=True)` |
| Measure relative raw quality | `draft_loop.quality.raw_utility` |

Use the same original prompt, atom count and element slots for every teacher, draft and prefix of a Plan. Keep the actual student lattice and every observed coordinate fixed when constructing a prefix completion. A registered complete teacher is not automatically compatible with a different student prefix; the resulting completion must be checked again. Rejected proposals, failed measurements and unknown results remain in the preparation record.

The evaluated material contains one complete registered teacher per source and positive corrections at actual commit positions. The core prefix weight is `gain / (1 + gain)` for positive, known `raw_utility` gain, with source/axis-balanced sampling. Gain compares the complete proposed structure to the original completed raw that produced the recorded prefix. The label concerns that complete physically checked completion, not the isolated causal effect of a single action. Other relative improvements can be retained for subsequent work; reaching SUN/MSUN is not an admission requirement for supervision. Unknown outcomes are not negative examples.

The `train` command consumes three artifacts:

- `teachers.jsonl`: each row has `source_id`, `source_split="train"`, `body_prompt`, `plan_state`, `body_token_ids`, `exact_record_key`, `measurement`, and registration metadata including `physically_reverified=true`.
- `verified_prefix_supervision.jsonl`: rows include the actual `positive_view` (`prompt`, `input_body`, `positions`, `tokens`, `axis`, `mask_id`), `input_prompt_key`, `input_prefix_key`, `proposal_record_key`, parent/candidate measurements, `raw_utility_gain`, and positive `weight`.
- `ROOT_REVIEW.json`: an artifact review with `eligible=true`, `teachers_digest=digest(teachers)` and `feedback_digest=digest(prefix_rows)`. This binds the reviewed material; it is not a model-performance acceptance flag.

The command verifies those bindings, exact teacher token/physics keys, prompt consistency, visible-state hashes, and the physical gain calculation. A single-point record can legitimately have `verified=false`, because that field refers to completed relaxation. Its known single-point values are retained without relabeling it as relaxed verification.

Source material for a reporting panel used in training must be explicitly labeled as seen training data. It cannot later be described as independent validation or final test data.

## Frozen recipe

`configs/registered_feedback.json` records the small seen-TRAIN setup:

- LoRA only, 512 updates, learning rate `1e-5`, microbatch 4, effective batch 8.
- 75% complete registered teachers and 25% verified actual-prefix corrections.
- C1 weights, backbone and trained input/output vocabulary tables remain fixed. No DPO or extra TRAIN replay is used in this particular recipe.
- Checkpoint selection uses fixed teacher/prefix fitting scores, not the subsequent physical panel.
- Training logit temperature is 0.7; deployed body sampling temperature is 0.2. They have different roles.
- Baseline and learned-model comparisons use the same body temperature, Plan order, body seeds, F seeds and physical protocol.
- F800 uses `ordered_csr_v1`. The tested repeatability claim is limited to the tested hardware and fixed batching.

The expanded reporting-data profile separately fixes 6,656 updates for 1,244 complete teachers and 3,747 prefix views, retaining the same 75/25 mixture, LR, batch sizes and frozen C1. Its completed training consumed 39,936 teacher and 13,312 prefix views, covering all 4,976 teacher source-phase groups at least eight times. This is a study-specific profile, not a change to the 512-update reference JSON. The [completed expanded raw comparison](results/reporting-1306-raw.md) does not reproduce the small-panel native gain; F800/F400 remain pending. Improved fitting is not a physical result.

The exported `assets.json` includes `inference_protocol`. Both draft sampling and F refinement apply it. Moving only a checkpoint and accidentally retaining an old temperature is prevented by this binding. A changed model/protocol or an unversioned draft cache cannot silently reuse existing output files.

The runtime example uses two visible GPUs, three C1 workers per GPU, a maximum F batch of 1024, 24 physics workers per GPU, and 32 Direct workers. C1 currently uses independent request workers, not a true cross-request construction batch. Device strings are relative to `CUDA_VISIBLE_DEVICES`; use an appropriate runtime profile for the available hardware. Keep the shared budget ledger and its original start time across every stage.

## Mathematical learning contract

For a view $v$ with original prompt, input token body, supervised positions $A_v$ and verified targets $y^+$, `score_views` uses

$$
\ell(v;\theta)=-\frac1{|A_v|}\sum_{j\in A_v}
\log\widetilde p_\theta(y_j^+\mid v),\qquad
\mathcal L_{\rm C2}=
0.75\,\mathbb E_{\mathcal D_T}\ell+
0.25\,\mathbb E_{\mathcal D_P^w}\ell.
$$

$\widetilde p$ normalizes DLM logits over the field's typed vocabulary at training temperature 0.7, merging periodic aliases before temperature. It omits C1 edges, dynamic geometry masks and confidence projection, so this is a reconstruction surrogate, not exact C1 trajectory likelihood.

Complete-teacher views cover lattice/X/Y/Z phases. Earlier phases retain consistent teacher values; selected fields and all later phases are masked, alternating full-phase and partial-phase supervision. Actual-prefix views preserve the recorded student input and supervise only its original commit positions.

Prefix views cycle over sources and axes, then use $\Pr(v\mid\mathrm{source},a)\propto w_v$ within a pool. The gain weight affects **sampling**; it is not multiplied into the loss again. Per-view token means and the fixed teacher/feedback/teacher/teacher microbatch cycle implement the stated mixture over the full cycle. AdamW updates LoRA only, with weight decay 0 and gradient clipping at norm 1.

Checkpoint selection uses $0.75\,\mathrm{NLL}_{\rm teacher}+0.25\,\mathrm{NLL}_{\rm fixed\ prefix}$. Teacher probes cover all four phases; prefix probes choose one fixed view per source-axis group. Neither selection nor the fitting-progress gate establishes free-generation gains.

## Evaluation and interpretation

`python -m dlm_iclr.feedback evaluate` evaluates a Plan file fixed before outcomes. The same recipe must be supplied for both comparison arms. `--raw-only` omits F; otherwise the recipe's `F_steps` determines refinement, with 800 in the reference recipe. Development and seen-TRAIN reporting use `--reference-split val`; only a frozen final protocol should use `--reference-split test`. Novelty uses the full configured TRAIN reference.

The report separates:

1. Native CHGNet single-point energy, forces, stress and a stability proxy requiring raw hull ≤0, maximum force ≤0.1 eV/Å and maximum stress ≤0.5 GPa.
2. Conventional raw SUN/MSUN, whose stability labels follow the retained CHGNet relaxation protocol and whose novelty/uniqueness geometry is the saved output before relaxation.
3. Each separately configured F output and its physical/Direct evaluation.

### F400 and F800 budget comparison

The current reporting-data evaluation adds F400 to raw and F800 for both the starting and learned model. It keeps Plans, original prompts, body temperature 0.2, model checkpoints, F seeds, batching and `ordered_csr_v1` fixed across the applicable pairs. Each model generates raw only once; both refinement settings reuse those saved drafts. The comparisons include learned versus starting model at F400, F400 versus F800 within each model, and learned F400 versus starting-model F800.

For a shared-raw Python evaluation, pass separate copies of the settings to the existing APIs:

```python
from copy import deepcopy
from dlm_iclr.draft_loop.evaluation import refine_drafts, evaluate_endpoint

# config, assets, plans, saved_drafts, settings, budget and output are bound
# to one fixed model/Plan panel before outcomes are observed.
for steps in (800, 400):
    current = deepcopy(settings)
    current['teacher_steps'] = steps
    refined = refine_drafts(config, assets, plans, saved_drafts,
                            output / f'F{steps}' / 'generated', current, budget)
    evaluate_endpoint(config, [row['record'] for row in refined],
                      output / f'F{steps}' / 'evaluation', current, budget)
```

The vendored sampler uses `time_start = diff_steps`. Thus F400 begins at index 400 while F800 begins at index 800; F400 is not simply a truncation of the same F800 trajectory. Keep separate execution identities and output folders, report actual elapsed refinement time separately from physical evaluation, and record cache reuse. A matching seed does not make two different schedules the same trajectory. No shorter-step quality or noninferiority claim has yet been established for this expansion.

Unknowns and failures stay in the request denominator. A group of 16 Plans sampled four times has 16 source groups, not 64 independent conditions. Four separately scored 16-item panels and one jointly scored 64-item panel have different uniqueness calculations. Do not mix their SUN numerators.

The [current results](results/registered-feedback-train.md) establish a seen-TRAIN effect at the stated configuration. They do not establish unseen-condition generalization, shorter refinement, DFT accuracy or a learned online critic effect. The package never automatically declares scientific success, publishes a model, or trains on final evaluation outputs.

Older experimental loop, preference and prefix-verifier components are retained as dependencies and research interfaces. Their existence is not evidence that they contributed to this recipe's result.
