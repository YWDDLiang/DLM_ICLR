# Reference training and inference profile

This page retains the original 1050-Plan module/editor profile underlying the [physical-feedback story](PHYSICAL_FEEDBACK_MASTER_STORY_ZH.md). Its data, optimization and runtime rules are unchanged by the narrative update; see [implementation and reproduction](keep-edit-implementation.md). The separate [registered draft-feedback recipe](registered-feedback.md) updates draft LoRA from complete teachers and actual-prefix supervision with C1 fixed, and does not use the editor/value/risk objective below.

The profile follows the retained 1050-Plan execution. A module can be trained from the configured data or supplied as an existing compatible checkpoint.

| Module | Initialization | Optimization | Selected artifact |
|---|---|---|---|
| Planner | Meta-Llama-3-8B base | Two independent one-epoch LoRA stages, each LR 2e-5, effective batch 8 | Epoch-2 final |
| B0 | LLaDA-8B-Instruct | Two epochs, effective batch 16, LR 5e-5, rank 8 and trained IO tables | Final adapter |
| C1 | New periodic head; frozen B0 | One epoch, batch 16, Adam 1e-4, width 64, 8 harmonics | VAL64 best; reference step 800 |
| Diffusion | CrysLLMGen CSPDiffusion | Upstream MP-20 recipe, 1000 noise steps | Supplied model or new last checkpoint |
| C2 warm-up | New geometry/state modules on B0 | One epoch of crystal reconstruction, LR 1e-5 | Warmed geometric editor |
| C2 editor | Warmed editor | 8 epochs, batch 16, content LR 2e-6, head LR 1e-4, reference KL weight 1 | Final editor |
| C2 light update | Fitted editor | 1 epoch, content LR 1e-6, head LR 1e-4, 0.25 physical-teacher mixture, beta 0.1 | Final light editor |
| Value | Final light editor features | 64 epochs, 32 sources/batch, head LR 1e-3, hidden LR 1e-5 | Autonomous value |
| Risk | Source-balanced physical outcomes | Logistic regression on 13 geometric features, C=0.5 | Fitted risk coefficients |

The reference C2 training used 1000 independent Plan sources, yielding 970 editor rows and 7424 value pairs. Its editor and light update performed 488 and 61 optimizer updates, respectively. Value fitting performed 1984 updates. Risk fitting used 8640 labeled endpoints.

Inference uses generation temperature 0.7, a Q=100 periodic head, F800, eight original editor candidates, an 80-forward budget, and one additional conditional revision. The fitted revision settings are risk strength 2, KL budget 0.1, risk penalty 0.5 and minimum predicted reduction 0.18866579450426538.

The phase order is `Planner → B0/C1 → F → physical evaluation → C2 → final evaluation`. C2 keeps F-stage confirmed SUN inputs; its final output is chosen by the learned candidate/value/risk mechanism for other inputs.

The reference profile records the training procedure and artifacts used by the retained system. Changing data, numerical batching or foundation checkpoints creates a new trained instance with its own recorded settings and measurements.
